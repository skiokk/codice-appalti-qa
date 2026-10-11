"""Sentence spans of unit texts, used to quote sources when a provider has no native citations.

Every span returned here is an exact substring of the input text, so it can be shown as a quoted
passage and checked against the source.
"""

from __future__ import annotations

import re

# A leading comma number or lettera label: "1. ", "1-bis. ", "a) ", "b-bis) ".
_ENUMERATOR = re.compile(r"(?:\d+(?:-[a-z]+)?\.|[a-z]{1,3}(?:-[a-z]+)?\))\s+")
# A sentence ends at a newline, at ';' followed by a space, at ':' closing a line, or at a full
# stop followed by a capital letter or the end of the text ("art. 50", "D.Lgs. 36", "n. 36" and
# "procedura aperta: nove mesi" are not split).
_BOUNDARY = re.compile(r"\n|;(?=\s|$)|:(?=[ \t]*(?:\n|$))|\.(?=\s+[A-ZÀ-Ý]|\s*$)")
_WORD = re.compile(r"\w+")


def sentences(text: str) -> list[str]:
    """Sentences of a unit text, without leading enumerators, as exact substrings."""
    spans: list[str] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if enumerator := _ENUMERATOR.match(text, pos):
            pos = enumerator.end()
        boundary = _BOUNDARY.search(text, pos)
        end = boundary.end() if boundary else len(text)
        if span := text[pos:end].strip():
            spans.append(span)
        pos = end
    return spans


def first_sentence(text: str, max_chars: int = 300) -> str:
    """The first sentence of a text, cut at a word boundary when longer than `max_chars`."""
    found = sentences(text)
    if not found:
        return ""
    sentence = found[0]
    if len(sentence) <= max_chars:
        return sentence
    cut = sentence.rfind(" ", 0, max_chars)
    return sentence[: cut if cut > 0 else max_chars].rstrip(" ,;:")


def _terms(text: str) -> set[str]:
    """Content words: four letters or more (skips most Italian function words), or numbers."""
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 4 or w.isdigit()}


def best_sentence(claim: str, text: str) -> str:
    """The sentence of `text` sharing the most content words with `claim` (first one on ties)."""
    found = sentences(text)
    if not found:
        return text.strip()
    claim_terms = _terms(claim)
    return max(found, key=lambda sentence: len(claim_terms & _terms(sentence)))
