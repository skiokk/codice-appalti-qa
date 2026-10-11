"""Units of the code body: one per comma, or one per article without commi.

Normattiva exports articles in two shapes:

* structured (snapshots from 2025): ``<paragraph>`` per comma with ``<num>``, lettere as
  ``<list>/<point>``;
* flat (snapshots up to 2024): the whole article in one ``<p>`` ("Articolo 1. Principio del
  risultato. 1. Le stazioni ... 2. La concorrenza ..."), sometimes preceded by a paragraph that
  only says "Articolo" and a ``<num>`` holding the article number.

Both produce the same keys and, for unchanged commi, the same wording.

Restarted numbering: some flat exports number a few commi from "1." again inside an article
(art. 41 commi 9-10, art. 62 commi 8-9 and art. 34 commi 5-6 read "1.", "2."). A flat article
with such a run is split twice: as exported (the run stays inside the previous comma) and with
the run read as the next commi. Which reading is right is decided across snapshots by
``alignment.align_numbering``, against the same article in a structured export.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace

from lxml import etree

from appalti_qa.ingest.numbering import RunPolicy, layout_lettere, split_numbered
from appalti_qa.ingest.text import (
    clean_heading,
    collapse_whitespace,
    cut_notes,
    has_markers,
    is_note,
)
from appalti_qa.ingest.units import ParsedUnit, make_unit
from appalti_qa.ingest.xmlutil import block_lines, child, children, inline_text, local_name
from appalti_qa.keys import (
    LATIN_SUFFIXES,
    article_from_num,
    comma_from_num,
    lettera_from_num,
    make_key,
)

logger = logging.getLogger(__name__)

_SUFFIXES = "|".join(sorted(LATIN_SUFFIXES, key=len, reverse=True))
_ARTICLE_PREFIX_RE = re.compile(rf"^Articolo\s+(?P<n>\d+(?:[\s-]*(?:{_SUFFIXES}))?)\b\s*\.?\s*")


@dataclass(frozen=True, slots=True)
class ParsedArticle:
    """The units of one article of the code body, in document order."""

    number: str  # "50", "225-bis"
    structured: bool  # one <paragraph> per comma: the export's own <num> numbers the commi
    units: list[ParsedUnit]  # as numbered in the export
    # Flat text with restarted numbering: the units with every run read as the next commi.
    renumbered: list[ParsedUnit] | None = None


def article_units(article: etree._Element) -> ParsedArticle:
    """Units of one ``<article>`` of the code body.

    Text between the heading and comma 1 continues the heading: art. 221 is exported as
    heading "Indirizzo, ... presso la Cabina di regia" followed by "Governance dei servizi" in
    the flat exports, as one heading in the structured ones; art. 140 marks the deletion of
    words from its heading with a "((...))" paragraph after it.
    """
    number = article_from_num(inline_text(child(article, "num")))
    raw_heading = inline_text(child(article, "heading"))
    paragraphs = [p for p in children(article, "paragraph") if not is_note(inline_text(p))]
    first = next((c for p in paragraphs if (c := _comma(p)) is not None), None)
    if first == "1":
        units = _structured(number, raw_heading, paragraphs)
        return ParsedArticle(number, True, units)
    as_exported = _flat(number, raw_heading, paragraphs, "keep")
    renumbered = _flat(number, raw_heading, paragraphs, "continue")
    differs = [u.key for u in renumbered] != [u.key for u in as_exported]
    return ParsedArticle(number, False, as_exported, renumbered if differs else None)


def _with_heading(units: list[ParsedUnit], raw_heading: str, lead: str) -> list[ParsedUnit]:
    """The units with the heading made of ``raw_heading`` and the ``lead`` text before comma 1.

    ``heading_marked`` keeps the markers of both, as evidence for a change of the heading.
    """
    raw = f"{raw_heading}. {lead}" if raw_heading.strip() and lead.strip() else raw_heading or lead
    heading = clean_heading(raw)
    marked = collapse_whitespace(raw) if heading is not None and has_markers(raw) else ""
    return [replace(unit, heading=heading, heading_marked=marked) for unit in units]


def _comma(paragraph: etree._Element) -> str | None:
    num = child(paragraph, "num")
    if num is None:
        return None
    try:
        return comma_from_num(inline_text(num))
    except ValueError:
        return None


def _structured(
    article: str, raw_heading: str, paragraphs: list[etree._Element]
) -> list[ParsedUnit]:
    commi: dict[str, list[str]] = {}
    lead: list[str] = []  # unnumbered paragraphs before comma 1: the heading or its rest
    for paragraph in paragraphs:
        comma = _comma(paragraph)
        if comma is None:
            text = collapse_whitespace(inline_text(paragraph))
            if commi:
                next(reversed(commi.values())).append(text)
            elif text:
                lead.append(text)
            continue
        if comma in commi:
            logger.warning("art. %s: duplicate comma %s merged", article, comma)
            commi[comma].extend(_render(paragraph, ""))
        else:
            commi[comma] = _render(paragraph, f"{comma}.")
    units = [
        make_unit(make_key(article, comma), "comma", "\n".join(lines), article=article, comma=comma)
        for comma, lines in commi.items()
    ]
    return _with_heading(units, raw_heading, " ".join(lead))


def _render(element: etree._Element, label: str) -> list[str]:
    """Lines of a structured paragraph or point: label + first line, then one line per point."""
    lines: list[str] = []
    for node in children(element):
        name = local_name(node)
        if name in ("num", "heading"):
            continue
        if name == "list":
            for item in children(node):
                if local_name(item) == "point":
                    lines.extend(_render(item, _lettera_label(item)))
                else:  # intro, wrapUp
                    lines.extend(block_lines(item))
        else:
            lines.extend(block_lines(node) if len(node) else [inline_text(node)])
    lines = [line for line in lines if line.strip()]
    if label:
        lines[:1] = [f"{label} {lines[0]}" if lines else label]
    return lines


def _lettera_label(point: etree._Element) -> str:
    raw = collapse_whitespace(inline_text(child(point, "num")))
    try:
        return f"{lettera_from_num(raw)})"
    except ValueError:
        return raw


def _flat(
    article: str, raw_heading: str, paragraphs: list[etree._Element], runs: RunPolicy
) -> list[ParsedUnit]:
    raw = " ".join(collapse_whitespace(inline_text(p)) for p in paragraphs)
    raw = _strip_article_prefix(cut_notes(collapse_whitespace(raw)), article)
    preamble, segments = split_numbered(raw, runs=runs)
    if not segments:
        if not raw:
            logger.warning("art. %s has no text", article)
            return []
        unit = make_unit(article, "article", raw, article=article)
        return _with_heading([unit], raw_heading, "")
    units = [
        make_unit(
            make_key(article, segment.label),
            "comma",
            f"{segment.label}. {layout_lettere(segment.body.strip())}",
            article=article,
            comma=segment.label,
        )
        for segment in segments
    ]
    return _with_heading(units, raw_heading, preamble)


def _strip_article_prefix(text: str, article: str) -> str:
    """Drop a leading "Articolo 50." when it names this article."""
    match = _ARTICLE_PREFIX_RE.match(text)
    if match and article_from_num(f"Articolo {match['n']}") == article:
        return text[match.end() :]
    return text
