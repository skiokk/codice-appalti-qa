"""The evaluation dataset (`eval/dataset.yaml`): schema, loading, selection and checks."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

import psycopg
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from appalti_qa.keys import is_valid_key, key_label, ref_matches, sort_key

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("eval/dataset.yaml")
MAX_EXCERPT_CHARS = 300

Kind = Literal["answerable", "temporal", "unanswerable", "multi"]
Difficulty = Literal["easy", "medium", "hard"]
# "scartata" marks a question the reviewer discarded: it is kept in the file but never evaluated.
Status = Literal["bozza", "verificata", "corretta", "scartata"]
VERIFIED_STATUSES: frozenset[str] = frozenset({"verificata", "corretta"})


class DatasetError(ValueError):
    """The dataset file cannot be used; `problems` lists every issue found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__(f"{len(self.problems)} problem(s) in the dataset:\n" + "\n".join(problems))


class EvalItem(BaseModel):
    """One evaluation question with its expected answer and references."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^Q\d{3,}$")
    question: str = Field(min_length=1)
    kind: Kind
    category: str = Field(min_length=1)
    vigenza: date
    expected_answer: str
    expected_refs: list[str] = Field(default_factory=list)
    source_excerpt: str = ""
    difficulty: Difficulty
    status: Status = "bozza"
    notes: str = ""

    @field_validator("source_excerpt", "notes", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        return "" if value is None else value

    @property
    def verified(self) -> bool:
        return self.status in VERIFIED_STATUSES


FIELDS: tuple[str, ...] = tuple(EvalItem.model_fields)


def _entry_label(entry: Any, position: int) -> str:
    entry_id = entry.get("id") if isinstance(entry, dict) else None
    return f"{entry_id} (entry {position})" if entry_id else f"entry {position}"


def parse_items(raw: Any, source: str = "dataset") -> list[EvalItem]:
    """Validate a loaded YAML document; raises DatasetError listing every invalid entry."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise DatasetError([f"{source}: expected a list of questions, got {type(raw).__name__}"])
    items: list[EvalItem] = []
    problems: list[str] = []
    for position, entry in enumerate(raw, start=1):
        label = _entry_label(entry, position)
        if not isinstance(entry, dict):
            problems.append(f"{label}: expected a mapping of fields")
            continue
        try:
            items.append(EvalItem.model_validate(entry))
        except ValidationError as exc:
            for error in exc.errors():
                field = ".".join(str(part) for part in error["loc"]) or "entry"
                problems.append(f"{label}: {field}: {error['msg']}")
    if problems:
        raise DatasetError(problems)
    return items


def load_dataset(path: Path = DEFAULT_DATASET) -> list[EvalItem]:
    """Read and validate the dataset file; raises DatasetError with every problem found."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DatasetError([f"{path}: file not found"]) from None
    except yaml.YAMLError as exc:
        raise DatasetError([f"{path}: invalid YAML: {exc}"]) from None
    return parse_items(raw, str(path))


def select_items(
    items: Iterable[EvalItem],
    *,
    include_drafts: bool = False,
    ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[EvalItem]:
    """Questions to evaluate: verified ones (or drafts too), never discarded ones."""
    wanted = set(ids) if ids else None
    selected = [
        item
        for item in items
        if item.status != "scartata"
        and (include_drafts or item.verified)
        and (wanted is None or item.id in wanted)
    ]
    return selected[:limit] if limit is not None else selected


# --- consistency checks -----------------------------------------------------------------------


def _static_problems(item: EvalItem) -> list[str]:
    problems: list[str] = []
    refs = item.expected_refs
    problems.extend(
        f"{item.id}: invalid key {ref!r} in expected_refs" for ref in refs if not is_valid_key(ref)
    )
    duplicates = sorted(ref for ref, count in Counter(refs).items() if count > 1)
    if duplicates:
        problems.append(f"{item.id}: duplicate expected_refs {duplicates}")
    if item.kind == "unanswerable" and refs:
        problems.append(f"{item.id}: unanswerable questions must have no expected_refs")
    if item.kind != "unanswerable" and not refs:
        problems.append(f"{item.id}: {item.kind} questions need at least one expected_ref")
    if item.kind == "multi" and len(refs) < 2:
        problems.append(f"{item.id}: multi questions need at least two expected_refs")
    if not item.expected_answer.strip():
        problems.append(f"{item.id}: expected_answer is empty")
    if len(item.source_excerpt) > MAX_EXCERPT_CHARS:
        problems.append(
            f"{item.id}: source_excerpt has {len(item.source_excerpt)} characters"
            f" (max {MAX_EXCERPT_CHARS})"
        )
    if item.kind != "unanswerable" and not item.source_excerpt.strip():
        problems.append(f"{item.id}: source_excerpt is empty")
    return problems


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace and drop Normattiva '((' '))' markers, for excerpt matching."""
    return " ".join(text.replace("((", "").replace("))", "").split())


_IN_FORCE_SQL = """
    SELECT u.key, v.text, v.repealed
    FROM unit_versions v JOIN units u ON u.id = v.unit_id
    WHERE v.valid_from <= %(day)s AND (v.valid_to IS NULL OR v.valid_to > %(day)s)
    ORDER BY u.ordinal
"""


_EMBEDDED_SQL = """
    SELECT count(*),
        count(*) FILTER (WHERE v.embedding IS NOT NULL AND v.embedding_model = %(model)s)
    FROM unit_versions v
    WHERE NOT v.repealed AND EXISTS (
        SELECT 1 FROM unnest(%(days)s::date[]) AS d(day)
        WHERE v.valid_from <= d.day AND (v.valid_to IS NULL OR v.valid_to > d.day))
"""


def embedding_coverage(
    conn: psycopg.Connection, model: str, days: Iterable[date]
) -> tuple[int, int]:
    """(embedded, total) over the versions semantic search can return on any of `days`.

    `total` counts the versions in force and not repealed on at least one of the days;
    `embedded`, those of them with an embedding made by `model`.
    """
    row = conn.execute(_EMBEDDED_SQL, {"model": model, "days": sorted(set(days))}).fetchone()
    return (int(row[1]), int(row[0])) if row else (0, 0)


def units_in_force(conn: psycopg.Connection, day: date) -> dict[str, tuple[str, bool]]:
    """Every unit in force on `day`: key -> (text, repealed), in document order."""
    rows = conn.execute(_IN_FORCE_SQL, {"day": day}).fetchall()
    return {key: (text, repealed) for key, text, repealed in rows}


def _matching(refs: Sequence[str], units: dict[str, tuple[str, bool]]) -> dict[str, str]:
    """Texts of the units in `units` that match any of `refs`, in document order."""
    return {
        key: text for key, (text, _) in units.items() if any(ref_matches(ref, key) for ref in refs)
    }


def _corpus_problems(items: Sequence[EvalItem], conn: psycopg.Connection) -> list[str]:
    by_day: dict[date, list[EvalItem]] = defaultdict(list)
    for item in items:
        if item.expected_refs and all(is_valid_key(ref) for ref in item.expected_refs):
            by_day[item.vigenza].append(item)
    problems: list[str] = []
    for day in sorted(by_day):
        units = units_in_force(conn, day)
        for item in by_day[day]:
            for ref in item.expected_refs:
                matches = [key for key in units if ref_matches(ref, key)]
                if not matches:
                    problems.append(f"{item.id}: {ref} matches no unit in force on {day}")
                elif all(units[key][1] for key in matches):
                    problems.append(f"{item.id}: {ref} is repealed on {day}")
            texts = _matching(item.expected_refs, units)
            excerpt = normalize_whitespace(item.source_excerpt)
            if (
                texts
                and excerpt
                and not any(excerpt in normalize_whitespace(t) for t in texts.values())
            ):
                problems.append(
                    f"{item.id}: source_excerpt not found in the text of {item.expected_refs}"
                    f" in force on {day}"
                )
    return problems


def validate_dataset(
    items: Sequence[EvalItem], *, conn: psycopg.Connection | None = None
) -> list[str]:
    """Consistency problems of the dataset (empty when valid).

    Static checks: unique ids, valid keys, `unanswerable` iff no refs, `multi` with 2+ refs,
    non-empty answer, excerpt present and at most 300 characters. With `conn`, also: every ref
    matches a unit in force (and not repealed) on the item's `vigenza`, and the excerpt occurs,
    whitespace-normalised, in the text of the referenced units.
    """
    counts = Counter(item.id for item in items)
    problems = [f"{item_id}: duplicate id" for item_id, n in sorted(counts.items()) if n > 1]
    for item in items:
        problems.extend(_static_problems(item))
    if conn is not None:
        problems.extend(_corpus_problems(items, conn))
    return problems


def referenced_texts(items: Sequence[EvalItem], conn: psycopg.Connection) -> dict[str, str]:
    """Item id -> labelled text of its referenced units in force on its `vigenza`."""
    by_day: dict[date, list[EvalItem]] = defaultdict(list)
    for item in items:
        if item.expected_refs:
            by_day[item.vigenza].append(item)
    texts: dict[str, str] = {}
    for day, group in by_day.items():
        units = units_in_force(conn, day)
        for item in group:
            refs = [ref for ref in item.expected_refs if is_valid_key(ref)]
            matched = _matching(refs, units)
            texts[item.id] = "\n\n".join(
                f"[{key_label(key)}]\n{matched[key]}" for key in sorted(matched, key=sort_key)
            )
    return texts
