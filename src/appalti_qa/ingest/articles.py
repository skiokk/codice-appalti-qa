"""Units of the code body: one per comma, or one per article without commi.

Normattiva exports articles in two shapes:

* structured (snapshots from 2025): ``<paragraph>`` per comma with ``<num>``, lettere as
  ``<list>/<point>``;
* flat (snapshots up to 2024): the whole article in one ``<p>`` ("Articolo 1. Principio del
  risultato. 1. Le stazioni ... 2. La concorrenza ..."), sometimes preceded by a paragraph that
  only says "Articolo" and a ``<num>`` holding the article number.

Both produce the same keys and, for unchanged commi, the same wording.
"""

from __future__ import annotations

import logging
import re

from lxml import etree

from appalti_qa.ingest.numbering import layout_lettere, split_numbered
from appalti_qa.ingest.text import clean_heading, collapse_whitespace, cut_notes, is_note
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


def article_units(article: etree._Element) -> list[ParsedUnit]:
    """Units of one ``<article>`` of the code body, in document order."""
    number = article_from_num(inline_text(child(article, "num")))
    heading = clean_heading(inline_text(child(article, "heading")))
    paragraphs = [p for p in children(article, "paragraph") if not is_note(inline_text(p))]
    first = next((c for p in paragraphs if (c := _comma(p)) is not None), None)
    if first == "1":
        return _structured(number, heading, paragraphs)
    return _flat(number, heading, paragraphs)


def _comma(paragraph: etree._Element) -> str | None:
    num = child(paragraph, "num")
    if num is None:
        return None
    try:
        return comma_from_num(inline_text(num))
    except ValueError:
        return None


def _structured(
    article: str, heading: str | None, paragraphs: list[etree._Element]
) -> list[ParsedUnit]:
    commi: dict[str, list[str]] = {}
    for paragraph in paragraphs:
        comma = _comma(paragraph)
        if comma is None:
            text = collapse_whitespace(inline_text(paragraph))
            if not commi and heading is None:
                heading = clean_heading(text)  # heading exported as an unnumbered paragraph
            elif commi:
                next(reversed(commi.values())).append(text)
            elif text:
                logger.debug("art. %s: ignoring unnumbered paragraph %r", article, text[:60])
            continue
        if comma in commi:
            logger.warning("art. %s: duplicate comma %s merged", article, comma)
            commi[comma].extend(_render(paragraph, ""))
        else:
            commi[comma] = _render(paragraph, f"{comma}.")
    return [
        make_unit(
            make_key(article, comma),
            "comma",
            "\n".join(lines),
            article=article,
            comma=comma,
            heading=heading,
        )
        for comma, lines in commi.items()
    ]


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


def _flat(article: str, heading: str | None, paragraphs: list[etree._Element]) -> list[ParsedUnit]:
    raw = " ".join(collapse_whitespace(inline_text(p)) for p in paragraphs)
    raw = _strip_article_prefix(cut_notes(collapse_whitespace(raw)), article)
    preamble, segments = split_numbered(raw)
    if not segments:
        if not raw:
            logger.warning("art. %s has no text", article)
            return []
        return [make_unit(article, "article", raw, article=article, heading=heading)]
    if heading is None:
        heading = clean_heading(preamble)
    return [
        make_unit(
            make_key(article, segment.label),
            "comma",
            f"{segment.label}. {layout_lettere(segment.body.strip())}",
            article=article,
            comma=segment.label,
            heading=heading,
        )
        for segment in segments
    ]


def _strip_article_prefix(text: str, article: str) -> str:
    """Drop a leading "Articolo 50." when it names this article."""
    match = _ARTICLE_PREFIX_RE.match(text)
    if match and article_from_num(f"Articolo {match['n']}") == article:
        return text[match.end() :]
    return text
