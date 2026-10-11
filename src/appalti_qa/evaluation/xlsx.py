"""Excel round trip for the human review of the dataset.

`export_xlsx` writes one row per question with Italian headers, an "Esito" dropdown
(OK / Da correggere / Scartare) and a free-text "Correzione" column. The reviewer may also edit
the question fields in place. `import_xlsx` merges the review back into the YAML file:

- edited fields are applied and the question becomes `corretta`;
- "OK" without edits makes it `verificata` (a `corretta` question stays `corretta`);
- "Da correggere" without edits leaves it in `bozza`;
- "Scartare" makes it `scartata` (kept in the file, never evaluated);
- a non-empty "Correzione" is appended to `notes` as "Revisione: ...".

A hidden sheet keeps the exported values, so edits are detected against what the reviewer saw and
a field changed in the YAML after the export is reported as a conflict instead of overwritten.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import ValidationError

from appalti_qa.evaluation.dataset import (
    FIELDS,
    DatasetError,
    EvalItem,
    Status,
    load_dataset,
    validate_dataset,
)
from appalti_qa.evaluation.yaml_writer import write_dataset
from appalti_qa.models import Answer

logger = logging.getLogger(__name__)

SHEET_REVIEW = "Revisione"
SHEET_HELP = "Istruzioni"
SHEET_ORIGINAL = "originale"  # hidden: the exported values
OUTCOME_OK, OUTCOME_FIX, OUTCOME_DROP = "OK", "Da correggere", "Scartare"
OUTCOMES = (OUTCOME_OK, OUTCOME_FIX, OUTCOME_DROP)
_MAX_CELL_CHARS = 32_000  # Excel's limit is 32,767


@dataclass(frozen=True, slots=True)
class Column:
    key: str  # an EvalItem field or one of the extra keys below
    header: str
    width: int


_ID = Column("id", "ID", 8)
_ITEM_COLUMNS = [
    Column("kind", "Tipo", 13),
    Column("category", "Categoria", 18),
    Column("vigenza", "Vigenza", 12),
    Column("question", "Domanda", 50),
    Column("expected_answer", "Risposta attesa", 60),
    Column("expected_refs", "Riferimenti", 18),
    Column("source_excerpt", "Estratto", 50),
]
_UNIT_TEXT = Column("unit_text", "Testo vigente", 70)
_ANSWER_COLUMNS = [
    Column("system_answer", "Risposta del sistema", 60),
    Column("cited_refs", "Riferimenti citati", 18),
]
_OUTCOME = Column("outcome", "Esito", 16)
_CORRECTION = Column("correction", "Correzione", 50)
_TAIL_COLUMNS = [
    Column("difficulty", "Difficoltà", 11),
    Column("status", "Stato", 11),
    Column("notes", "Note", 40),
]
# Columns whose edits are imported, by header; status is informational ("Esito" decides it).
EDITABLE: dict[str, str] = {
    c.header: c.key for c in [*_ITEM_COLUMNS, *_TAIL_COLUMNS] if c.key != "status"
}

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="0066CC")
_REVIEW_FILL = PatternFill("solid", fgColor="FFF4CC")
_WRAP_TOP = Alignment(wrap_text=True, vertical="top")
_OUTCOME_FILLS = {
    OUTCOME_OK: PatternFill("solid", fgColor="C6EFCE", bgColor="C6EFCE"),
    OUTCOME_FIX: PatternFill("solid", fgColor="FFEB9C", bgColor="FFEB9C"),
    OUTCOME_DROP: PatternFill("solid", fgColor="D9D9D9", bgColor="D9D9D9"),
}
_HELP_LINES = [
    "Revisione delle domande di valutazione",
    "",
    "1. Nel foglio «Revisione» confronta domanda, risposta attesa, riferimenti ed estratto con il "
    "testo del codice vigente alla data indicata nella colonna Vigenza.",
    "2. Nella colonna Esito scegli: OK (la domanda va bene così), Da correggere (c'è un errore), "
    "Scartare (la domanda non va usata).",
    "3. Puoi correggere direttamente le celle Tipo, Categoria, Vigenza, Domanda, Risposta attesa, "
    "Riferimenti, Estratto, Difficoltà e Note: le modifiche vengono importate e la domanda passa "
    "allo stato «corretta».",
    "4. Se non modifichi le celle, descrivi il problema nella colonna Correzione: la nota viene "
    "salvata e la domanda resta in bozza finché non viene corretta.",
    "5. Riferimenti: chiavi separate da virgola, ad esempio 50/1/b, 50/4, all:I.3/p1.",
    "6. Le righe senza Esito non vengono importate. Non modificare la colonna ID.",
]


# --- cell values ------------------------------------------------------------------------------


def _cell_value(key: str, value: Any) -> Any:
    if key == "expected_refs":
        return ", ".join(value)
    if key == "vigenza":
        return value.isoformat()
    return value


def _parse_cell(key: str, raw: Any) -> Any:
    """The field value of a cell; raises ValueError for an unreadable date."""
    if key == "vigenza":
        if isinstance(raw, datetime):
            return raw.date()
        if isinstance(raw, date):
            return raw
        return date.fromisoformat(str(raw or "").strip())
    text = "" if raw is None else str(raw).replace("\r\n", "\n").strip()
    if key == "expected_refs":
        return [ref for ref in re.split(r"[,;\s]+", text) if ref]
    if key in ("kind", "difficulty"):
        return text.lower()
    return text


def _normalized(key: str, value: Any) -> Any:
    return _parse_cell(key, _cell_value(key, value))


def _set_cell(ws: Worksheet, row: int, column: int, value: Any) -> None:
    cell = ws.cell(row=row, column=column)
    if isinstance(value, str):
        value = ILLEGAL_CHARACTERS_RE.sub("", value)
        if len(value) > _MAX_CELL_CHARS:
            value = value[: _MAX_CELL_CHARS - 1] + "…"
    cell.value = value
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"  # text, not a formula
    cell.alignment = _WRAP_TOP


# --- export -----------------------------------------------------------------------------------


def _columns(*, with_texts: bool, with_answers: bool) -> list[Column]:
    return [
        _ID,
        *_ITEM_COLUMNS,
        *([_UNIT_TEXT] if with_texts else []),
        *(_ANSWER_COLUMNS if with_answers else []),
        _OUTCOME,
        _CORRECTION,
        *_TAIL_COLUMNS,
    ]


def _row_values(
    item: EvalItem,
    columns: Sequence[Column],
    answers: Mapping[str, Answer] | None,
    unit_texts: Mapping[str, str] | None,
) -> list[Any]:
    answer = (answers or {}).get(item.id)
    extra = {
        "unit_text": (unit_texts or {}).get(item.id, ""),
        "system_answer": answer.text if answer else "",
        "cited_refs": ", ".join(answer.cited_keys) if answer else "",
    }
    return [
        _cell_value(c.key, getattr(item, c.key)) if c.key in FIELDS else extra.get(c.key, "")
        for c in columns
    ]


def _row_height(values: Sequence[Any], columns: Sequence[Column]) -> float:
    """Approximate height (points) for wrapped text; Excel does not auto-fit on open."""
    lines = 1
    for value, column in zip(values, columns, strict=True):
        per_line = max(1, int(column.width * 1.1))
        paragraphs = str(value or "").split("\n")
        lines = max(lines, sum(max(1, math.ceil(len(p) / per_line)) for p in paragraphs))
    return min(409.0, 15.0 * lines)


def _write_review_sheet(
    ws: Worksheet, rows: Sequence[Sequence[Any]], columns: Sequence[Column]
) -> None:
    ws.title = SHEET_REVIEW
    for index, column in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=index, value=column.header)
        cell.font, cell.fill = _HEADER_FONT, _HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(index)].width = column.width
    for row_number, values in enumerate(rows, start=2):
        for index, value in enumerate(values, start=1):
            _set_cell(ws, row_number, index, value)
        ws.row_dimensions[row_number].height = _row_height(values, columns)
    last_row = max(2, len(rows) + 1)
    for column in (_OUTCOME, _CORRECTION):
        letter = get_column_letter(columns.index(column) + 1)
        for row_number in range(2, last_row + 1):
            ws[f"{letter}{row_number}"].fill = _REVIEW_FILL
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{last_row}"
    _add_outcome_rules(ws, get_column_letter(columns.index(_OUTCOME) + 1), last_row)


def _add_outcome_rules(ws: Worksheet, letter: str, last_row: int) -> None:
    cells = f"{letter}2:{letter}{last_row}"
    validation = DataValidation(
        type="list",
        formula1='"' + ",".join(OUTCOMES) + '"',
        allow_blank=True,
        showErrorMessage=True,
        showInputMessage=True,
        errorTitle="Esito non valido",
        error="Scegli OK, Da correggere o Scartare.",
        promptTitle="Esito della revisione",
        prompt="OK, Da correggere o Scartare",
    )
    ws.add_data_validation(validation)
    validation.add(cells)
    for outcome, fill in _OUTCOME_FILLS.items():
        ws.conditional_formatting.add(
            cells, CellIsRule(operator="equal", formula=[f'"{outcome}"'], fill=fill)
        )


def export_xlsx(
    items: Sequence[EvalItem],
    path: Path,
    *,
    answers: Mapping[str, Answer] | None = None,
    unit_texts: Mapping[str, str] | None = None,
) -> None:
    """Write the review workbook. `answers` and `unit_texts` (by item id) add optional columns."""
    columns = _columns(with_texts=unit_texts is not None, with_answers=answers is not None)
    workbook = Workbook()
    rows = [_row_values(item, columns, answers, unit_texts) for item in items]
    _write_review_sheet(workbook.active, rows, columns)

    help_sheet = workbook.create_sheet(SHEET_HELP)
    help_sheet.column_dimensions["A"].width = 120
    for row_number, line in enumerate(_HELP_LINES, start=1):
        cell = help_sheet.cell(row=row_number, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    help_sheet["A1"].font = Font(bold=True, size=14)

    original = workbook.create_sheet(SHEET_ORIGINAL)
    original.sheet_state = "hidden"
    original.append(["id", "valori"])
    for item in items:
        original.append([item.id, json.dumps(item.model_dump(mode="json"), ensure_ascii=False)])

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


# --- import -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReviewRow:
    id: str
    row: int  # sheet row number, for messages
    cells: dict[str, Any]  # raw values of the editable columns, by field
    outcome: str | None
    correction: str


@dataclass(frozen=True, slots=True)
class ReviewSheet:
    rows: list[ReviewRow]
    originals: dict[str, dict[str, Any]] | None  # exported field values, by id


@dataclass(slots=True)
class ReviewResult:
    items: list[EvalItem]  # the whole dataset with the review applied, in file order
    changed: list[str] = field(default_factory=list)  # ids of the changed items
    changes: list[str] = field(default_factory=list)  # one line per changed item
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _parse_outcome(raw: Any) -> str | None:
    text = "" if raw is None else " ".join(str(raw).split())
    if not text:
        return None
    for outcome in OUTCOMES:
        if text.casefold() == outcome.casefold():
            return outcome
    raise ValueError(f"invalid Esito {text!r} (expected one of {', '.join(OUTCOMES)})")


def _at(values: Sequence[Any], position: int | None) -> Any:
    return values[position] if position is not None and position < len(values) else None


def read_review(path: Path) -> ReviewSheet:
    """Read the reviewer's sheet; raises DatasetError for missing columns or invalid outcomes."""
    workbook = load_workbook(path, data_only=True)
    ws = workbook[SHEET_REVIEW] if SHEET_REVIEW in workbook.sheetnames else workbook.active
    headers = [" ".join(str(c.value or "").split()) for c in ws[1]]
    index = {header: i for i, header in enumerate(headers) if header}
    missing = [h for h in (_ID.header, _OUTCOME.header) if h not in index]
    if missing:
        raise DatasetError([f"{path}: missing column(s) {', '.join(missing)}"])
    editable = {key: index[header] for header, key in EDITABLE.items() if header in index}
    rows: list[ReviewRow] = []
    problems: list[str] = []
    for row_number, values in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        item_id = str(_at(values, index[_ID.header]) or "").strip()
        if not item_id:
            continue
        try:
            outcome = _parse_outcome(_at(values, index[_OUTCOME.header]))
        except ValueError as exc:
            problems.append(f"{item_id} (row {row_number}): {exc}")
            continue
        correction = str(_at(values, index.get(_CORRECTION.header)) or "").strip()
        cells = {key: _at(values, position) for key, position in editable.items()}
        rows.append(ReviewRow(item_id, row_number, cells, outcome, correction))
    if problems:
        raise DatasetError(problems)
    originals = None
    if SHEET_ORIGINAL in workbook.sheetnames:
        originals = {
            str(item_id): json.loads(values)
            for item_id, values, *_ in workbook[SHEET_ORIGINAL].iter_rows(
                min_row=2, values_only=True
            )
            if item_id and values
        }
    return ReviewSheet(rows, originals)


def _new_status(current: Status, outcome: str, edited: bool) -> Status:
    if outcome == OUTCOME_DROP:
        return "scartata"
    if edited:
        return "corretta"
    if outcome == OUTCOME_OK:
        return "corretta" if current == "corretta" else "verificata"
    return "bozza"


def _with_note(notes: str, note: str) -> str:
    if note in notes:
        return notes
    return f"{notes}\n{note}" if notes else note


def _row_edits(
    row: ReviewRow, item: EvalItem, baseline: EvalItem, result: ReviewResult
) -> dict[str, Any] | None:
    """Fields the reviewer changed with respect to the exported values (None on problems)."""
    edits: dict[str, Any] = {}
    ok = True
    for key, raw in row.cells.items():
        try:
            value = _parse_cell(key, raw)
        except ValueError as exc:
            result.problems.append(f"{row.id} (row {row.row}): {key}: {exc}")
            ok = False
            continue
        exported = _normalized(key, getattr(baseline, key))
        if value == exported:
            continue
        # Notes are free text: the reviewer's version wins. Other fields must not have changed in
        # the dataset since the export (unless to the reviewer's value, on a repeated import).
        current = _normalized(key, getattr(item, key))
        if key != "notes" and current not in (exported, value):
            result.problems.append(
                f"{row.id}: {key} changed in the dataset after the export; export it again"
            )
            ok = False
        edits[key] = value
    return edits if ok else None


def apply_review(items: Sequence[EvalItem], review: ReviewSheet) -> ReviewResult:
    """Merge the reviewer's rows into `items` (no I/O); see the module docstring for the rules."""
    by_id = {item.id: item for item in items}
    result = ReviewResult(items=[])
    for row in review.rows:
        item = by_id.get(row.id)
        if item is None:
            result.warnings.append(f"row {row.row}: unknown id {row.id}, ignored")
            continue
        exported = (review.originals or {}).get(row.id)
        try:
            baseline = EvalItem.model_validate(exported) if exported else item
        except ValidationError:
            result.problems.append(f"{row.id}: unreadable exported values in the hidden sheet")
            continue
        edits = _row_edits(row, item, baseline, result)
        if edits is None:
            continue
        if row.outcome is None:
            if edits or row.correction:
                result.warnings.append(f"{row.id}: edited but Esito is empty, row ignored")
            continue
        status = _new_status(item.status, row.outcome, any(k != "notes" for k in edits))
        notes = edits.pop("notes", item.notes)
        if row.correction:
            notes = _with_note(notes, f"Revisione: {row.correction}")
        elif row.outcome == OUTCOME_FIX and not edits:
            result.warnings.append(f"{row.id}: 'Da correggere' without edits or a Correzione")
        try:
            updated = EvalItem.model_validate(
                {**item.model_dump(), **edits, "status": status, "notes": notes}
            )
        except ValidationError as exc:
            result.problems += [
                f"{row.id}: {'.'.join(map(str, error['loc']))}: {error['msg']}"
                for error in exc.errors()
            ]
            continue
        if updated != item:
            by_id[row.id] = updated
            result.changed.append(row.id)
            result.changes.append(_describe_change(item, updated))
    result.items = [by_id[item.id] for item in items]
    return result


def _describe_change(before: EvalItem, after: EvalItem) -> str:
    edited = [
        key
        for key in FIELDS
        if key not in ("status", "notes") and getattr(after, key) != getattr(before, key)
    ]
    detail = f"; edited: {', '.join(edited)}" if edited else ""
    note = "; note updated" if after.notes != before.notes else ""
    return f"{after.id}: {before.status} -> {after.status}{detail}{note}"


def review_import(
    path: Path,
    dataset_path: Path,
    *,
    conn: psycopg.Connection | None = None,
    write: bool = True,
) -> ReviewResult:
    """Apply a reviewed workbook to the dataset file; raises DatasetError and writes nothing
    when the review or the changed questions have problems (corpus checks need `conn`)."""
    items = load_dataset(dataset_path)
    result = apply_review(items, read_review(path))
    changed_ids = set(result.changed)
    changed = [item for item in result.items if item.id in changed_ids]
    problems = result.problems + validate_dataset(changed, conn=conn)
    if problems:
        raise DatasetError(problems)
    if write and changed:
        write_dataset(dataset_path, result.items)
    return result


def import_xlsx(path: Path, dataset_path: Path) -> list[EvalItem]:
    """Merge reviewer edits from `path` into `dataset_path` (rewritten in place); returns the
    updated dataset. Raises DatasetError, leaving the file untouched, on any problem."""
    result = review_import(path, dataset_path)
    for warning in result.warnings:
        logger.warning(warning)
    return result.items
