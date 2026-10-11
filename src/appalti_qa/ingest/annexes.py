"""Units of the allegati (annexes).

Each allegato is exported as one or more ``<doc>`` elements whose body is flat text:

* ``"Allegato I.1-art. 2"``: one article of a structured allegato -> one unit per comma,
  ``all:I.1/2/1`` (or ``all:I.1/2`` when the article has no numbered commi);
* ``"Allegato I.3"``: an allegato without articles -> one unit per numbered block,
  ``all:I.3/p1``, ``all:I.3/p2``... in document order (a single ``all:I.3`` unit when nothing
  is numbered); the text before the first block (references, part titles, introductions) is
  kept at the start of ``p1``;
* ``"Allegato I.4-Tabella A"``, ``"Allegato I.5-Schede"``: tables and forms -> one unit keyed by
  a slug of the part name, ``all:I.4/tabella-a``.

``"Allegati - Indice allegati"`` (the table of contents) is not content and is skipped. Docs of
an annex to an annex are named after both (``"Allegato II.2-Allegato II.2 bis art. 1"``); the
last allegato named is the one the doc belongs to (``II.2-bis``).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import replace

from lxml import etree

from appalti_qa.ingest.numbering import layout_lettere, split_numbered
from appalti_qa.ingest.text import clean_heading, collapse_whitespace, cut_notes, is_note
from appalti_qa.ingest.units import ParsedUnit, make_unit
from appalti_qa.ingest.xmlutil import block_lines, child, children, inline_text
from appalti_qa.keys import (
    ANNEX_PREFIX,
    LATIN_SUFFIXES,
    annex_from_doc_name,
    article_from_num,
    is_valid_key,
    make_key,
)

logger = logging.getLogger(__name__)

_SUFFIXES = "|".join(sorted(LATIN_SUFFIXES, key=len, reverse=True))
_ANNEX_NAME_RE = re.compile(r"Allegato\s+[IVX]+\.\d+(?:\s*(?:bis|ter))?\b")
_ARTICLE_PART_RE = re.compile(r"^art(?:icolo)?\.?\s*\d", re.IGNORECASE)
_ANNEX_TITLE_RE = re.compile(
    r"^\W*allegato\s+[IVX]+\s*\.?\s*\d+(?:[\s-]*(?:bis|ter))?\b\W*?(?=\w|\(|$)", re.IGNORECASE
)
_ARTICLE_TITLE_RE = re.compile(
    rf"(?<!\()\bArticolo\s+(?P<n>\d+(?:[\s-]*(?:{_SUFFIXES}))?)\b(?!\s*,)\s*\.?\s*"
)
_MAX_TITLE_CHARS = 250


def doc_identity(name: str) -> tuple[str, str] | None:
    """('II.2-bis', 'art. 1') for 'Allegato II.2-Allegato II.2 bis art. 1'; None if no annex."""
    start = name.rfind("Allegato")
    if start < 0:
        return None
    tail = name[start:]
    annex = annex_from_doc_name(tail)
    match = _ANNEX_NAME_RE.match(tail)
    if annex is None or match is None:
        return None
    return annex, tail[match.end() :].strip(" -")


def annex_units(docs: Iterable[etree._Element]) -> Iterator[ParsedUnit]:
    """Units of the annex ``<doc>`` elements, in document order.

    Articles without a heading of their own get the allegato title, read from the first
    article of the allegato.
    """
    titles: dict[str, str] = {}
    for doc in docs:
        identity = doc_identity(doc.get("name", ""))
        if identity is None:
            continue
        annex, part = identity
        if _ARTICLE_PART_RE.match(part):
            title, units = _article_doc(annex, article_from_num(part), _flat_text(doc))
            if title:
                titles.setdefault(annex, title)
            fallback = titles.get(annex)
            yield from (replace(u, heading=fallback) if u.heading is None else u for u in units)
        elif not part:
            yield from _blocks_doc(annex, _flat_text(doc))
        else:
            yield from _part_doc(annex, part, doc)


def _doc_lines(doc: etree._Element, *, keep_breaks: bool) -> list[str]:
    body = child(doc, "mainBody")
    lines: list[str] = []
    for paragraph in children(body) if body is not None else ():
        block: list[str] = []
        for node in children(paragraph):
            block.extend(
                block_lines(node, keep_breaks=keep_breaks) if len(node) else [inline_text(node)]
            )
        if not is_note("\n".join(block)):
            lines.extend(block)
    return lines


def _flat_text(doc: etree._Element) -> str:
    return cut_notes(collapse_whitespace(" ".join(_doc_lines(doc, keep_breaks=False))))


def _article_doc(annex: str, article: str, text: str) -> tuple[str | None, list[ParsedUnit]]:
    """(allegato title found before "Articolo N.", units of the article)."""
    title_end, body = _split_article_title(text, article)
    title = _annex_title(text[:title_end]) if title_end else None
    preamble, segments = split_numbered(body)
    if not segments:
        key = make_key(article, annex=annex)
        return title, [make_unit(key, "allegato_block", body, article=article, allegato=annex)]
    heading = clean_heading(preamble)
    units = [
        make_unit(
            make_key(article, segment.label, annex=annex),
            "allegato_block",
            f"{segment.label}. {layout_lettere(segment.body.strip())}",
            article=article,
            comma=segment.label,
            allegato=annex,
            heading=heading,
        )
        for segment in segments
    ]
    return title, units


def _split_article_title(text: str, article: str) -> tuple[int, str]:
    """(start of "Articolo N.", text after it); the first doc of an allegato opens with the
    allegato title, which comes before."""
    for match in _ARTICLE_TITLE_RE.finditer(text):
        if article_from_num(f"Articolo {match['n']}") == article:
            return match.start(), text[match.end() :]
    logger.debug("article title %s not found in annex doc", article)
    return 0, text


def _annex_title(text: str) -> str | None:
    """'Allegato I.4 Imposta di bollo ... (Articolo 18, comma 10)' -> 'Imposta di bollo ...'."""
    match = _ANNEX_TITLE_RE.match(text)
    title, _ = _split_title(text[match.end() :] if match else text)
    return title


def _blocks_doc(annex: str, text: str) -> list[ParsedUnit]:
    title_match = _ANNEX_TITLE_RE.match(text)
    if title_match:
        text = text[title_match.end() :]
    preamble, segments = split_numbered(text, restart=True)
    heading, intro = _split_title(preamble)
    if not segments:
        if not intro.strip():
            logger.warning("allegato %s has a title but no text", annex)
            return []
        key = make_key(annex=annex)
        return [make_unit(key, "allegato_block", intro, allegato=annex, heading=heading)]
    units = []
    for index, segment in enumerate(segments, start=1):
        raw = f"{segment.label}. {layout_lettere(segment.body.strip())}"
        if index == 1 and intro.strip():
            raw = f"{intro.strip()}\n{raw}"
        key = make_key(annex=annex, block=f"p{index}")
        units.append(make_unit(key, "allegato_block", raw, allegato=annex, heading=heading))
    return units


def _split_title(preamble: str) -> tuple[str | None, str]:
    """Split "Title (Articolo 17, comma 3) intro..." into the title and the rest.

    The title ends at the first parenthesis; without one, a short preamble is all title and a
    long one is cut after its first sentence.
    """
    position = _first_parenthesis(preamble)
    if position is None and len(preamble) <= _MAX_TITLE_CHARS:
        position = len(preamble)
    elif position is None or position > _MAX_TITLE_CHARS:
        sentence_end = preamble.find(". ")
        position = sentence_end + 1 if 0 < sentence_end < _MAX_TITLE_CHARS else 0
    return clean_heading(preamble[:position]), preamble[position:]


def _first_parenthesis(text: str) -> int | None:
    """Index of the first "(" that is not part of a "((" modification marker."""
    for index, char in enumerate(text):
        if char == "(" and text[index + 1 : index + 2] != "(" and text[index - 1 : index] != "(":
            return index
    return None


def _part_doc(annex: str, part: str, doc: etree._Element) -> list[ParsedUnit]:
    key = f"{ANNEX_PREFIX}{annex}/{slugify(part)}"
    if not is_valid_key(key):
        logger.warning("skipping annex doc %r: no valid key", part)
        return []
    text = cut_notes("\n".join(_doc_lines(doc, keep_breaks=True)))
    heading = collapse_whitespace(part)
    return [make_unit(key, "allegato_block", text, allegato=annex, heading=heading)]


def slugify(label: str) -> str:
    """'Tabella A' -> 'tabella-a'."""
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_label.lower()).strip("-")
