"""Canonical unit keys shared by ingestion, retrieval, citations and the eval dataset.

Grammar (see docs/architecture.md):

    key      := code_key | annex_key
    code_key := ARTICLE [ "/" COMMA [ "/" LETTERA ] ]
    annex_key:= "all:" ANNEX ( "/" ARTICLE [ "/" COMMA [ "/" LETTERA ] ] | "/" BLOCK )?

ARTICLE and COMMA are numbers with an optional Latin suffix ("225-bis", "1-ter", "01"),
LETTERA is a lowercase label ("b", "b-bis", "aa"), ANNEX is the allegato code ("I.1", "II.12")
and BLOCK is a segment starting with a letter ("p3" for the 3rd numbered block, "tabella-a").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANNEX_PREFIX = "all:"

_NUMBER = r"\d+(?:-[a-z]+)?"
_LETTERA = r"[a-z0-9]{1,3}(?:-[a-z]+)?"
_ANNEX = r"[IVX]+\.\d+(?:-[a-z]+)?"
_BLOCK = r"[a-z][a-z0-9-]*"

_PATH = rf"(?P<article>{_NUMBER})(?:/(?P<comma>{_NUMBER})(?:/(?P<lettera>{_LETTERA}))?)?"
_CODE_KEY_RE = re.compile(rf"^{_PATH}$")
_ANNEX_KEY_RE = re.compile(
    rf"^{ANNEX_PREFIX}(?P<annex>{_ANNEX})"
    rf"(?:/(?P<block>{_BLOCK})|/{_PATH})?$"
)

LATIN_SUFFIXES = (
    "bis",
    "ter",
    "quater",
    "quinquies",
    "sexies",
    "septies",
    "octies",
    "novies",
    "nonies",
    "decies",
    "undecies",
    "duodecies",
    "terdecies",
    "quaterdecies",
    "quinquiesdecies",
    "sexiesdecies",
    "septiesdecies",
    "octiesdecies",
)
_SUFFIX = rf"(?:[\s-]*(?P<suffix>{'|'.join(LATIN_SUFFIXES)}))?"

_ARTICLE_NUM_RE = re.compile(rf"^\s*(?:articolo|art\.?)\s*(?P<n>\d+){_SUFFIX}\b", re.IGNORECASE)
_COMMA_NUM_RE = re.compile(rf"^\s*(?P<n>\d+){_SUFFIX}\s*\.?\s*$", re.IGNORECASE)
_LETTERA_NUM_RE = re.compile(rf"^\s*(?P<n>[a-z0-9]{{1,3}}){_SUFFIX}\s*\)\s*$", re.IGNORECASE)
_ANNEX_NAME_RE = re.compile(r"Allegato\s+(?P<code>[IVX]+\.\d+)(?:\s*(?P<suffix>bis|ter))?\b")


def _join(number: str, suffix: str | None) -> str:
    return f"{number}-{suffix.lower()}" if suffix else number


def article_from_num(num: str) -> str:
    """'Articolo 50.' -> '50'; 'Articolo 225-bis.' -> '225-bis'; 'Art. 4 bis' -> '4-bis'."""
    match = _ARTICLE_NUM_RE.match(num)
    if not match:
        raise ValueError(f"not an article number: {num!r}")
    return _join(match["n"], match["suffix"])


def comma_from_num(num: str) -> str:
    """'1.' -> '1'; '1-bis.' -> '1-bis'; '01.' -> '01'."""
    match = _COMMA_NUM_RE.match(num)
    if not match:
        raise ValueError(f"not a comma number: {num!r}")
    return _join(match["n"], match["suffix"])


def lettera_from_num(num: str) -> str:
    """'b)' -> 'b'; 'b-bis)' -> 'b-bis'; 'B bis)' -> 'b-bis'."""
    match = _LETTERA_NUM_RE.match(num)
    if not match:
        raise ValueError(f"not a lettera label: {num!r}")
    return _join(match["n"].lower(), match["suffix"])


def annex_from_doc_name(name: str) -> str | None:
    """'Allegato I.1-art. 2' -> 'I.1'; returns None for docs that are not an allegato.

    The last allegato named wins: 'Allegato II.2-Allegato II.2 bis art. 1' -> 'II.2-bis'.
    """
    matches = list(_ANNEX_NAME_RE.finditer(name))
    match = matches[-1] if matches else None
    return _join(match["code"], match["suffix"]) if match else None


@dataclass(frozen=True, slots=True)
class UnitKey:
    """A parsed canonical key. Exactly one of article/block is set for annex keys with a path."""

    annex: str | None = None
    article: str | None = None
    comma: str | None = None
    lettera: str | None = None
    block: str | None = None

    def __str__(self) -> str:
        parts = [p for p in (self.article, self.comma, self.lettera) if p]
        if self.block:
            parts = [self.block]
        if self.annex:
            return "/".join([f"{ANNEX_PREFIX}{self.annex}", *parts])
        return "/".join(parts)

    @property
    def unit(self) -> UnitKey:
        """The retrieval-unit key this ref falls in (drops the lettera)."""
        return UnitKey(self.annex, self.article, self.comma, None, self.block)


def make_key(
    article: str | None = None,
    comma: str | None = None,
    lettera: str | None = None,
    *,
    annex: str | None = None,
    block: str | None = None,
) -> str:
    """Build and validate a canonical key from its parts."""
    key = str(UnitKey(annex, article, comma, lettera, block))
    parse_key(key)
    return key


def parse_key(key: str) -> UnitKey:
    """Parse a canonical key; raises ValueError when it does not follow the grammar."""
    match = _ANNEX_KEY_RE.match(key) if key.startswith(ANNEX_PREFIX) else _CODE_KEY_RE.match(key)
    if not match:
        raise ValueError(f"invalid unit key: {key!r}")
    groups = match.groupdict()
    return UnitKey(
        annex=groups.get("annex"),
        article=groups.get("article"),
        comma=groups.get("comma"),
        lettera=groups.get("lettera"),
        block=groups.get("block"),
    )


def is_valid_key(key: str) -> bool:
    try:
        parse_key(key)
    except ValueError:
        return False
    return True


def ref_matches(ref: str, unit_key: str) -> bool:
    """True when a reference (any granularity) and a retrieval unit overlap.

    '50/1/b' matches unit '50/1'; '50' matches every comma of article 50; '50/1' matches an
    article-level unit '50' only if that article has no commi (the unit key is then '50').
    """
    return unit_key == ref or unit_key.startswith(ref + "/") or ref.startswith(unit_key + "/")


def key_label(key: str) -> str:
    """Human-readable Italian label: '50/1/b' -> 'Art. 50, comma 1, lett. b)'."""
    parsed = parse_key(key)
    parts: list[str] = []
    if parsed.annex:
        parts.append(f"Allegato {parsed.annex}")
    if parsed.block:
        block = parsed.block
        numbered = re.fullmatch(r"p\d+", block)
        parts.append(f"punto {block[1:]}" if numbered else block.replace("-", " "))
    if parsed.article:
        parts.append(f"{'art.' if parsed.annex else 'Art.'} {parsed.article}")
    if parsed.comma:
        parts.append(f"comma {parsed.comma}")
    if parsed.lettera:
        parts.append(f"lett. {parsed.lettera})")
    return ", ".join(parts)


def _natural(segment: str | None) -> tuple[int, int, str]:
    """Order '4' < '4-bis' < '4-ter' < '5'; non-numeric segments sort last."""
    if segment is None:
        return (-1, -1, "")
    match = re.fullmatch(r"(\d+)(?:-(\w+))?", segment)
    if not match:
        return (10**9, 0, segment)
    suffix = match[2]
    rank = LATIN_SUFFIXES.index(suffix) + 1 if suffix in LATIN_SUFFIXES else 0
    return (int(match[1]), rank, suffix or "")


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}  # the code has Parts I-V of allegati


def sort_key(key: str) -> tuple:
    """Document order: code articles before annexes, numeric order with Latin suffixes after."""
    parsed = parse_key(key)
    annex: tuple = (0, 0, (0, 0, ""))
    if parsed.annex:
        roman, _, rest = parsed.annex.partition(".")
        annex = (1, _ROMAN.get(roman, 99), _natural(rest))
    return (
        annex,
        _natural(parsed.article),
        parsed.block or "",
        _natural(parsed.comma),
        parsed.lettera or "",
    )
