"""Split flat legal text into numbered commi and lay out lettere on their own lines.

Normattiva exported the code as one line of text per article until 2024, and the allegati are
flat text in every snapshot, so commi ("1.", "2-bis.") and lettere ("a)", "b-bis)") have to be
found inside running text. A label opens a new unit only when

* it follows sentence punctuation (``.``, ``;``, ``:``) or the end of a ``))`` insertion (the
  first "1." may also follow a heading without final punctuation), and
* it continues the numbering: "2" or "1-bis" after "1", "a-bis" or "b" after "a" (Italian lists
  skip j, k, w, x and y, so a lettera may jump a few letters).

References inside the text ("ai sensi del comma 2.", "lettere c), d) ed e)") fail one of the two
checks and stay part of the running text, and so does anything quoted between « and » (amending
provisions quote whole articles of other acts). Three defects of the flat exports are repaired:

* a comma misnumbered "1." after a full stop is accepted under the expected number when the
  next label carries on from it ("7. ... 1. ... 9." reads as 7, 8, 9);
* with ``renumber_tail=True``, the last commi of an article numbered from "1." again
  ("4. ... 1. ... 2." at the end of art. 34) continue the article's numbering when they run
  without gaps to the end of the text ("1.", "2." read as 5, 6);
* an article replaced as a whole is exported as the old text followed by the new one, both
  numbered from "1."; a second "1." followed by "2." whose opening words match the first comma
  starts over, so only the last (current) copy is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from appalti_qa.ingest.text import collapse_whitespace, strip_markers
from appalti_qa.keys import LATIN_SUFFIXES

_SUFFIXES = "|".join(sorted(LATIN_SUFFIXES, key=len, reverse=True))
_NUMBER_RE = re.compile(
    rf"(?<![\w.,'/])(?P<open>\(\(\s*)?(?P<n>\d{{1,3}})"
    rf"(?:(?:[ -]?(?P<suffix>{_SUFFIXES}))\b ?\.?| ?\.)(?=\s|\(\(|[A-Z«\"(])"
)
_LETTERA_RE = re.compile(
    rf"(?<![\w'])(?:\(\(\s*)?(?P<l>[a-z]{{1,2}})(?:[ -]?(?P<suffix>{_SUFFIXES}))?\)(?=\s|\(\()"
)
_SENTENCE_END = (".", "))")
_MAX_LETTER_JUMP = 4  # v) -> z) in the Italian alphabet
_COPY_PROBE_CHARS = 200
_COPY_MIN_SIMILARITY = 0.6

Value = tuple[int, int]  # (number, rank of the Latin suffix: 0 none, 1 bis, 2 ter, ...)


@dataclass(frozen=True, slots=True)
class Segment:
    """A numbered unit found in flat text: its canonical label and its text after the label."""

    label: str  # "1", "2-bis"
    body: str  # raw text after the label; markers kept, an opening "((" included


@dataclass(frozen=True, slots=True)
class _Label:
    match: re.Match[str]
    value: Value
    boundary: str  # what precedes the label: "", ".", ";", ":", "))" or another character


def _quoted(text: str) -> list[tuple[int, int]]:
    """Spans between matching « and » (an unmatched « quotes nothing)."""
    spans, opened = [], []
    for index, char in enumerate(text):
        if char == "«":
            opened.append(index)
        elif char == "»" and opened:
            start = opened.pop()
            if not opened:
                spans.append((start, index))
    return spans


def _outside(spans: list[tuple[int, int]], position: int) -> bool:
    return not any(start < position < end for start, end in spans)


def _rank(suffix: str | None) -> int:
    return LATIN_SUFFIXES.index(suffix.lower()) + 1 if suffix else 0


def _boundary(text: str, start: int) -> str:
    before = text[:start].rstrip()
    if not before:
        return ""
    return "))" if before.endswith("))") else before[-1]


def _may_open(label: _Label, restart: bool) -> bool:
    """Labels after punctuation may open a unit; with restarts, so may a "1." after a part title
    ("PARTE II", "C - INFORMAZIONI ... (di cui all'articolo 83) 1. Nome, ...")."""
    if label.boundary in (".", ";", ":", "))"):
        return True
    title_end = label.boundary == ")" or label.boundary.isupper()
    return restart and label.value == (1, 0) and title_end


def _continues(previous: Value | None, current: Value, max_jump: int = 1) -> bool:
    if previous is None:
        return current == (1, 0)
    (base, rank), (prev_base, prev_rank) = current, previous
    if base == prev_base:
        return rank > prev_rank
    return 0 < base - prev_base <= max_jump and rank == 0


def _label_text(value: Value) -> str:
    base, rank = value
    return f"{base}-{LATIN_SUFFIXES[rank - 1]}" if rank else str(base)


def _probe(text: str, label: _Label) -> str:
    end = label.match.end()
    return collapse_whitespace(strip_markers(text[end : end + _COPY_PROBE_CHARS]))


def _starts_copy(text: str, first: _Label, label: _Label, following: _Label | None) -> bool:
    if label.value != (1, 0) or following is None or following.value != (2, 0):
        return False
    similarity = SequenceMatcher(None, _probe(text, first), _probe(text, label)).ratio()
    return similarity >= _COPY_MIN_SIMILARITY


def _is_trailing_run(labels: list[_Label]) -> bool:
    """True for "1.", "2.", ... (at least two) that are the last labels of the text."""
    return len(labels) >= 2 and all(
        label.value == (number, 0) for number, label in enumerate(labels, start=1)
    )


def split_numbered(
    text: str, *, restart: bool = False, renumber_tail: bool = False
) -> tuple[str, list[Segment]]:
    """Split flat text at its numbered labels; returns (text before "1.", segments).

    With ``restart=True`` every "1." after a punctuation mark starts a new list (annexes that
    number their parts from 1 again); otherwise the numbering must run on. ``renumber_tail``
    enables the repair of a final run of commi numbered from "1." again (see above).
    """
    quoted = _quoted(text)
    labels = [
        _Label(match, (int(match["n"]), _rank(match["suffix"])), _boundary(text, match.start()))
        for match in _NUMBER_RE.finditer(text)
        if _outside(quoted, match.start())
    ]
    first_index = next((i for i, label in enumerate(labels) if label.value == (1, 0)), None)
    if first_index is None:
        return text, []
    accepted: list[tuple[_Label, Value]] = [(labels[first_index], (1, 0))]
    strong = [label for label in labels[first_index + 1 :] if _may_open(label, restart)]
    for index, label in enumerate(strong):
        previous = accepted[-1][1]
        following = strong[index + 1] if index + 1 < len(strong) else None
        if _continues(previous, label.value) or (restart and label.value == (1, 0)):
            accepted.append((label, label.value))
        elif label.boundary not in _SENTENCE_END or label.value != (1, 0):
            continue
        elif _starts_copy(text, accepted[0][0], label, following):
            accepted = [(label, (1, 0))]
        elif following is not None and _continues((previous[0] + 1, 0), following.value):
            accepted.append((label, (previous[0] + 1, 0)))
        elif renumber_tail and _is_trailing_run(strong[index:]):
            run = strong[index:]
            accepted.extend((item, (previous[0] + offset, 0)) for offset, item in enumerate(run, 1))
            break
    segments = []
    for index, (label, value) in enumerate(accepted):
        end = accepted[index + 1][0].match.start() if index + 1 < len(accepted) else len(text)
        body = (label.match["open"] or "") + text[label.match.end() : end]
        segments.append(Segment(_label_text(value), body))
    return text[: labels[first_index].match.start()], segments


def _letter_index(letters: str) -> int:
    value = ord(letters[0]) - ord("a") + 1
    return value if len(letters) == 1 else 26 + value


def layout_lettere(text: str) -> str:
    """Put each lettera of a flat comma on its own line ("...: a) ...; b) ...")."""
    breaks: list[int] = []
    previous: Value | None = None
    quoted = _quoted(text)
    for match in _LETTERA_RE.finditer(text):
        if not _outside(quoted, match.start()):
            continue
        letters = match["l"]
        if len(letters) == 2 and letters[0] != letters[1]:
            continue  # "aa)", "bb)" follow "z)"; other pairs are not list labels
        current = (_letter_index(letters), _rank(match["suffix"]))
        boundary = _boundary(text, match.start())
        if boundary not in ("", ".", ";", ":", "))"):
            continue
        restarts = current == (1, 0) and boundary == ":"
        if not (_continues(previous, current, _MAX_LETTER_JUMP) or restarts):
            continue
        breaks.append(match.start())
        previous = current
    if not breaks:
        return text
    pieces, last = [], 0
    for position in breaks:
        pieces.append(text[last:position].rstrip())
        last = position
    pieces.append(text[last:])
    return "\n".join(piece for piece in pieces if piece)
