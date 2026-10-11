"""Write the dataset back to YAML, keeping the file's order, layout and comments.

Only the lines of fields whose value changed are rewritten. When the file does not have the
expected layout (one `- id: ...` entry per question, two-space indented fields), or the patched
text does not load back to the same items, the file is regenerated with its leading comment block
kept and a warning logged.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from appalti_qa.evaluation.dataset import FIELDS, DatasetError, EvalItem, parse_items

logger = logging.getLogger(__name__)

_ENTRY_RE = re.compile(r"""^-\s+id:\s*["']?(?P<id>[^"'\s#]+)""")
_FIELD_RE = re.compile(r"^  (?P<field>[a-z_]+):(?:\s|$)")
_PLAIN_FIELDS = frozenset({"id", "kind", "category", "difficulty", "status"})
_PLAIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")


def yaml_value(field: str, value: Any) -> str:
    """A field value in the dataset's style: plain enums, double-quoted strings, flow lists."""
    if field == "vigenza":
        return f'"{value.isoformat()}"'
    if field == "expected_refs":
        return json.dumps(list(value), ensure_ascii=False)
    if field in _PLAIN_FIELDS and _PLAIN_RE.fullmatch(value) and yaml.safe_load(value) == value:
        return value
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(value, ensure_ascii=False)


def _field_line(field: str, value: Any) -> str:
    return f"  {field}: {yaml_value(field, value)}\n"


def format_item(item: EvalItem) -> str:
    values = item.model_dump()
    lines = [f"- id: {yaml_value('id', item.id)}\n"]
    lines += [_field_line(field, values[field]) for field in FIELDS if field != "id"]
    return "".join(lines)


def _field_spans(block: Sequence[str]) -> dict[str, tuple[int, int]] | None:
    """Line range [start, end) of each field of an entry whose first line is '- id: ...'."""
    spans: dict[str, tuple[int, int]] = {"id": (0, 1)}
    current: str | None = None
    for index, line in enumerate(block[1:], start=1):
        match = _FIELD_RE.match(line)
        if match:
            current = match["field"]
            if current in spans:
                return None
            spans[current] = (index, index + 1)
        elif current and line.startswith("   ") and line.strip():  # continuation line
            spans[current] = (spans[current][0], index + 1)
        else:  # a blank or comment line ends the field
            current = None
    return spans


def _patch_entry(block: list[str], item: EvalItem) -> list[str] | None:
    try:
        (current,) = parse_items(yaml.safe_load("".join(block)))
    except (yaml.YAMLError, ValueError):  # DatasetError is a ValueError
        return None
    spans = _field_spans(block)
    if spans is None or current.id != item.id:
        return None
    old, new = current.model_dump(), item.model_dump()
    changed = [field for field in FIELDS if old[field] != new[field]]
    out = list(block)
    missing = [field for field in changed if field not in spans]
    if missing:
        end = max(stop for _, stop in spans.values())
        out[end:end] = [_field_line(field, new[field]) for field in missing]
    for field in sorted((f for f in changed if f in spans), key=lambda f: spans[f], reverse=True):
        start, stop = spans[field]
        out[start:stop] = [_field_line(field, new[field])]
    return out


def patch_dataset_text(text: str, items: Sequence[EvalItem]) -> str | None:
    """`text` with the changed fields of `items` rewritten, or None if the layout is unknown."""
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("-")]
    ids = [match["id"] if (match := _ENTRY_RE.match(lines[i])) else None for i in starts]
    if ids != [item.id for item in items]:
        return None
    if not starts:
        return text
    out = lines[: starts[0]]
    for start, end, item in zip(starts, [*starts[1:], len(lines)], items, strict=True):
        block = _patch_entry(lines[start:end], item)
        if block is None:
            return None
        out += block
    return "".join(out)


def _loads_back(text: str, items: Sequence[EvalItem]) -> bool:
    try:
        return parse_items(yaml.safe_load(text)) == list(items)
    except (yaml.YAMLError, DatasetError):
        return False


def _leading_comments(text: str) -> str:
    header: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.startswith("#"):
            break
        header.append(line)
    block = "".join(header)
    return block if not block or block.endswith("\n\n") else block.rstrip("\n") + "\n\n"


def write_dataset(path: Path, items: Sequence[EvalItem]) -> None:
    """Save `items` to `path`, rewriting only what changed when the layout allows it."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    text = patch_dataset_text(original, items)
    if text is None or not _loads_back(text, items):
        if original:
            logger.warning("%s: layout not recognised, rewriting the whole file", path)
        text = _leading_comments(original) + "\n".join(format_item(item) for item in items)
        if not _loads_back(text, items):
            dumped = [item.model_dump(mode="json") for item in items]
            text = _leading_comments(original) + yaml.safe_dump(
                dumped, allow_unicode=True, sort_keys=False, width=10**6
            )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
