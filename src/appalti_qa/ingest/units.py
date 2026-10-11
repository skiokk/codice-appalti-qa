"""The parsed retrieval unit and its constructor, shared by the body and annex parsers."""

from __future__ import annotations

from dataclasses import dataclass

from appalti_qa.ingest.text import has_markers, is_repealed, normalize_text
from appalti_qa.models import UnitKind


@dataclass(frozen=True, slots=True)
class ParsedUnit:
    """One retrieval unit (comma, article without commi, or annex block) of one snapshot."""

    key: str
    kind: UnitKind
    article: str | None
    comma: str | None
    allegato: str | None
    heading: str | None
    text: str
    ordinal: int
    repealed: bool = False
    amended: bool = False  # the snapshot marks (part of) the text as inserted or replaced


def make_unit(
    key: str,
    kind: UnitKind,
    raw_text: str,
    *,
    article: str | None = None,
    comma: str | None = None,
    allegato: str | None = None,
    heading: str | None = None,
) -> ParsedUnit:
    """Build a unit from raw text (markers still in); the ordinal is assigned by the caller."""
    text = normalize_text(raw_text)
    return ParsedUnit(
        key=key,
        kind=kind,
        article=article,
        comma=comma,
        allegato=allegato,
        heading=heading,
        text=text,
        ordinal=0,
        repealed=is_repealed(text),
        amended=has_markers(raw_text),
    )
