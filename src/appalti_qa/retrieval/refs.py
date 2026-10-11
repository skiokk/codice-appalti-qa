"""Explicit references to the code in free Italian text, as canonical unit keys.

Recognised forms (case-insensitive, abbreviated or in full):

- "art. 50", "articolo 225-bis", "artt. 50 e 51"
- "art. 50, comma 1, lett. b)", "art. 50 c. 1", "art. 50, commi 1 e 2", "lettere a) e b)"
- "comma 2 dell'articolo 17", "lettera b) del comma 1 dell'art. 50"
- "allegato I.1", "allegato II.12, art. 2, comma 1", "art. 2 dell'allegato I.1"

References followed by another act ("art. 80 del d.lgs. 50/2016", "art. 120 c.p.a.") are skipped,
unless that act is D.Lgs. 36/2023 itself.
"""

from __future__ import annotations

import re

from appalti_qa.keys import LATIN_SUFFIXES, make_key

_SUFFIX = "|".join(LATIN_SUFFIXES)
_APOS = r"['\u2019]"  # straight or typographic apostrophe
_NUM = rf"\d+(?:\s*-?\s*(?:{_SUFFIX})\b)?"
_CONJ = r"\s*(?:,|\be\b|\bed\b|\bo\b)\s*"
_NUMS = rf"{_NUM}(?:{_CONJ}{_NUM})*"
_LABEL = rf"[a-z]{{1,3}}(?:\s*-?\s*(?:{_SUFFIX}))?\s*\)"
# The first lettera may omit the parenthesis only when it is a single letter ("lett. b").
_LABELS = rf"(?:{_LABEL}|[a-z]\b(?!{_APOS}))(?:{_CONJ}{_LABEL})*"
_ANNEX = r"[IVX]+\.\d+(?:\s*-?\s*(?:bis|ter)\b)?"

_ART = r"\b(?:articol[oi]|artt?\.?)\s*"
_COMMA = r"\b(?:comm[ai]|co\.|c\.)\s*"
_LETT = r"\b(?:letter[ae]|lett\.?|let\.)\s*"
_ALL = r"\b(?:allegat[oi]|all\.)\s*"
_SEP = r"\s*,?\s*"
_OF = rf"\s*,?\s*(?:de|ne|a)l(?:l{_APOS}|lo|la)?\s*"  # del, dello, della, dell', nel..., all'...

_TAIL = rf"(?:{_SEP}{_COMMA}(?P<commi>{_NUMS})(?:{_SEP}{_LETT}(?P<lettere>{_LABELS}))?)?"
_PATTERNS = (
    # "art. 50, comma 1, lett. b) [dell'allegato I.1]"
    re.compile(rf"{_ART}(?P<articles>{_NUMS}){_TAIL}(?:{_OF}{_ALL}(?P<annex>{_ANNEX}))?", re.I),
    # "[lettera b) del] comma 1[, lettera b),] dell'articolo 50 [dell'allegato I.1]"
    re.compile(
        rf"(?:{_LETT}(?P<lettere_first>{_LABELS}){_OF})?"
        rf"{_COMMA}(?P<commi>{_NUMS})(?:{_SEP}{_LETT}(?P<lettere>{_LABELS}))?"
        rf"{_OF}{_ART}(?P<articles>{_NUM})(?:{_OF}{_ALL}(?P<annex>{_ANNEX}))?",
        re.I,
    ),
    # "allegato I.1[, art. 2, comma 1]"
    re.compile(rf"{_ALL}(?P<annex>{_ANNEX})(?:{_SEP}{_ART}(?P<articles>{_NUMS}){_TAIL})?", re.I),
)

_OTHER_ACT = re.compile(
    rf"\s*,?\s*(?:(?:de|ne|a)l(?:l{_APOS}|lo|la|le)?|dei)?\s*"
    r"(?:d\.?\s*lgs|decreto|legge|l\.|d\.\s*l\.|d\.?\s*p\.?\s*r|r\.\s*d\.|regolamento|direttiva"
    r"|costituzione|tfue|trattato|testo\s+unico|t\.\s*u\.|c\.\s*c\.|c\.\s*p\.\s*a"
    rf"|codice\s+(?:civile|penale|di\s+procedura|del\s+processo|dell{_APOS}amministrazione))",
    re.I,
)
_THIS_ACT = re.compile(r"\b36\s*(?:/|del)\s*2023\b|\b31\s+marzo\s+2023\b", re.I)
_ITEM_SPLIT = re.compile(r"\s*,\s*|\s+(?:e|ed|o)\s+", re.I)
_ITEM = re.compile(rf"(\w+?)\s*-?\s*({_SUFFIX})?\s*\)?", re.I)


def extract_refs(text: str) -> list[str]:
    """Canonical keys of the units of D.Lgs. 36/2023 cited in `text`, distinct, in text order.

    "art. 50, comma 1, lett. b)" -> ["50/1/b"]; "artt. 50 e 51" -> ["50", "51"];
    "allegato I.1, art. 2" -> ["all:I.1/2"].
    """
    refs: list[str] = []
    pos = 0
    while True:
        found = [m for pattern in _PATTERNS if (m := pattern.search(text, pos))]
        if not found:
            break
        match = min(found, key=lambda m: (m.start(), -m.end()))
        if not _cites_other_act(text, match.end()):
            refs.extend(_keys(match))
        pos = match.end()
    return list(dict.fromkeys(refs))


def _cites_other_act(text: str, end: int) -> bool:
    other = _OTHER_ACT.match(text, end)
    return other is not None and not _THIS_ACT.search(text, other.end(), other.end() + 40)


def _keys(match: re.Match[str]) -> list[str]:
    """Expand one match; commi qualify the last article, lettere the last comma."""
    groups = match.groupdict()
    annex = _annex_code(groups["annex"]) if groups.get("annex") else None
    articles = _items(groups.get("articles"))
    commi = _items(groups.get("commi"))
    lettere = _items(groups.get("lettere") or groups.get("lettere_first"))
    if not articles:
        return [make_key(annex=annex)] if annex else []

    parts: list[tuple[str, ...]] = [(article,) for article in articles[:-1]]
    last = articles[-1]
    if not commi:
        parts.append((last,))
    else:
        parts.extend((last, comma) for comma in commi[:-1])
        if lettere:
            parts.extend((last, commi[-1], lettera) for lettera in lettere)
        else:
            parts.append((last, commi[-1]))

    keys = []
    for part in parts:
        try:
            keys.append(make_key(*part, annex=annex))
        except ValueError:
            continue
    return keys


def _items(group: str | None) -> list[str]:
    """'1, 2 e 3-bis' -> ['1', '2', '3-bis']; 'a) e b bis)' -> ['a', 'b-bis']."""
    if not group:
        return []
    items = []
    for raw in _ITEM_SPLIT.split(group.strip()):
        item = _ITEM.fullmatch(raw.strip())
        if item:
            label, suffix = item[1].lower(), item[2]
            items.append(f"{label}-{suffix.lower()}" if suffix else label)
    return items


def _annex_code(raw: str) -> str:
    """'i.1' -> 'I.1'; 'I.7 bis' -> 'I.7-bis'."""
    code, _, suffix = re.sub(r"\s*-?\s*(bis|ter)$", r" \1", raw.strip(), flags=re.I).partition(" ")
    roman, _, number = code.partition(".")
    return f"{roman.upper()}.{number}" + (f"-{suffix.lower()}" if suffix else "")
