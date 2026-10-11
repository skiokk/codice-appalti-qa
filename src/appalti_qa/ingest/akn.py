"""Parse one Normattiva Akoma Ntoso snapshot into act metadata, events and retrieval units."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from lxml import etree

from appalti_qa.ingest.annexes import annex_units
from appalti_qa.ingest.articles import article_units
from appalti_qa.ingest.labels import act_label, short_label
from appalti_qa.ingest.text import collapse_whitespace, normalize_text, text_hash
from appalti_qa.ingest.units import ParsedUnit
from appalti_qa.ingest.xmlutil import AKN_NS, akn, child, children, inline_text, local_name
from appalti_qa.models import Event

__all__ = [
    "AKN_NS",
    "ActInfo",
    "ParsedSnapshot",
    "ParsedUnit",
    "normalize_text",
    "parse_snapshot",
    "read_events",
    "text_hash",
    "vigenza_from_filename",
]

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^(?P<code>[^_]+)_(?P<date>\d{8})\.xml$")
_CODE_IN_TITLE_RE = re.compile(r"\s*\((?P<code>\d{2}[A-Z]\d{5})\)\s*$")


@dataclass(frozen=True, slots=True)
class ActInfo:
    code: str  # Normattiva codice redazionale, "23G00044"
    urn: str  # "urn:nir:stato:decreto.legislativo:2023-03-31;36"
    title: str
    short_title: str  # "D.Lgs. 36/2023"
    issued_on: date


@dataclass(frozen=True, slots=True)
class ParsedSnapshot:
    vigenza_date: date  # from the file name 23G00044_YYYYMMDD.xml
    expression_date: date | None  # FRBRExpression/FRBRdate
    source_file: str
    sha256: str  # of the file bytes
    act: ActInfo
    events: list[Event]  # lifecycle of this snapshot, original act included, by date
    units: list[ParsedUnit]  # document order: code body, then allegati; ordinals from 1


def vigenza_from_filename(path: Path) -> date:
    """23G00044_20250401.xml -> 2025-04-01."""
    match = _FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"not a snapshot file name (CODE_YYYYMMDD.xml): {path.name}")
    raw = match["date"]
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))


def parse_snapshot(path: Path) -> ParsedSnapshot:
    """Parse a Normattiva AKN export; raises ValueError for files that are not AKN acts."""
    payload = path.read_bytes()
    root = _parse_xml(payload, path)
    act_element = child(root, "act")
    if act_element is None:
        raise ValueError(f"{path.name}: no <act> element")
    units = _number(_dedupe(_units(act_element)))
    return ParsedSnapshot(
        vigenza_date=vigenza_from_filename(path),
        expression_date=_expression_date(root),
        source_file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        act=_act_info(root, path),
        events=_events(root),
        units=units,
    )


def read_events(path: Path) -> list[Event]:
    """Only the lifecycle events of a snapshot (cheaper than parse_snapshot)."""
    return _events(_parse_xml(path.read_bytes(), path))


def _parse_xml(payload: bytes, path: Path) -> etree._Element:
    try:
        root = etree.fromstring(payload, parser=etree.XMLParser(huge_tree=True))
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"{path.name}: not well-formed XML ({exc})") from exc
    if etree.QName(root).namespace != AKN_NS:
        raise ValueError(f"{path.name}: not an Akoma Ntoso 3.0 document")
    return root


def _units(act: etree._Element) -> Iterable[ParsedUnit]:
    body = child(act, "body")
    if body is not None:
        for article in body.iter(akn("article")):
            yield from article_units(article)
    attachments = child(act, "attachments")
    if attachments is not None:
        yield from annex_units(attachments.iter(akn("doc")))


def _dedupe(units: Iterable[ParsedUnit]) -> list[ParsedUnit]:
    """Merge units that share a key (malformed numbering) into the first occurrence."""
    merged: dict[str, ParsedUnit] = {}
    for unit in units:
        first = merged.get(unit.key)
        if first is None:
            merged[unit.key] = unit
            continue
        logger.warning("duplicate unit key %s: texts merged", unit.key)
        merged[unit.key] = replace(
            first,
            text=f"{first.text}\n{unit.text}",
            repealed=first.repealed and unit.repealed,
            amended=first.amended or unit.amended,
        )
    return list(merged.values())


def _number(units: list[ParsedUnit]) -> list[ParsedUnit]:
    return [replace(unit, ordinal=index) for index, unit in enumerate(units, start=1)]


def _frbr(root: etree._Element, level: str) -> etree._Element | None:
    return root.find(f".//{akn('identification')}/{akn(level)}")


def _expression_date(root: etree._Element) -> date | None:
    expression = _frbr(root, "FRBRExpression")
    frbr_date = child(expression, "FRBRdate") if expression is not None else None
    value = frbr_date.get("date") if frbr_date is not None else None
    return date.fromisoformat(value) if value else None


def _act_info(root: etree._Element, path: Path) -> ActInfo:
    work = _frbr(root, "FRBRWork")
    if work is None:
        raise ValueError(f"{path.name}: no FRBRWork identification")
    aliases = {a.get("name"): a.get("value", "") for a in children(work, "FRBRalias")}
    this = child(work, "FRBRthis")
    work_href = this.get("value", "") if this is not None else ""
    raw_title = collapse_whitespace(inline_text(root.find(f".//{akn('docTitle')}")))
    code_match = _CODE_IN_TITLE_RE.search(raw_title)
    file_match = _FILENAME_RE.match(path.name)
    code = code_match["code"] if code_match else (file_match["code"] if file_match else "")
    title = _CODE_IN_TITLE_RE.sub("", raw_title).rstrip(" .")
    work_date = child(work, "FRBRdate")
    if work_date is None or not work_date.get("date"):
        raise ValueError(f"{path.name}: no FRBRWork date")
    return ActInfo(
        code=code,
        urn=aliases.get("urn:nir", work_href),
        title=title,
        short_title=short_label(work_href) or code,
        issued_on=date.fromisoformat(work_date.get("date")),
    )


def _events(root: etree._Element) -> list[Event]:
    """Lifecycle events joined with the references that name the acts behind them."""
    references = root.find(f".//{akn('references')}")
    sources: dict[str, tuple[str, str]] = {}
    for reference in children(references) if references is not None else ():
        if reference.get("eId") and reference.get("href"):
            sources[reference.get("eId")] = (local_name(reference), reference.get("href"))
    events: list[Event] = []
    lifecycle = root.find(f".//{akn('lifecycle')}")
    for event_ref in children(lifecycle, "eventRef") if lifecycle is not None else ():
        source = sources.get(event_ref.get("source", "").lstrip("#"))
        if source is None or not event_ref.get("date"):
            logger.warning("eventRef %s has no matching reference", event_ref.get("eId"))
            continue
        kind, href = source
        events.append(
            Event(
                date=date.fromisoformat(event_ref.get("date")),
                kind="original" if kind == "original" else "amendment",
                amending_urn=href,
                label=act_label(href),
            )
        )
    return sorted(events, key=lambda event: event.date)
