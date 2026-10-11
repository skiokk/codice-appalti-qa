"""Text normalisation shared by the Akoma Ntoso parser and the versioning step.

Normattiva marks text inserted or replaced by an amending act with ``((`` ... ``))`` (inside an
``<ins>`` element, in the flat and in the structured exports alike), a deletion with ``((...))``
and a repealed comma with a marked note (``((COMMA ABROGATO DAL ...))``). The markers stay in
the text once an act has added them (later exports occasionally drop some). The stored text
drops them; the text with its markers is kept separately (``ParsedUnit.marked``) as evidence of
what amending acts changed (``marked_words``).
"""

from __future__ import annotations

import hashlib
import re

_MARKER_RE = re.compile(r"\(\(|\)\)")
_SPACES_RE = re.compile(r"[ \t\r\f\v\u00a0]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Characters ignored when comparing two renderings of the same text (see content_fingerprint):
# whitespace, dots, hyphens, and quotes or apostrophes, straight (' ` ") or typographic
# (U+2018, U+2019, guillemets, U+201C-U+201E).
_LAYOUT_RE = re.compile("[\\s.\\-'`\"\u2018\u2019\u00ab\u00bb\u201c\u201d\u201e]+")
_WORD_RE = re.compile(r"\w+")
# Editorial notes appended by Normattiva after an article or annex ("------ AGGIORNAMENTO (9) ...").
_NOTE_RE = re.compile(r"-{3,}\s*(?:\)\)\s*)?AGGIORNAMENTO\b.*\Z", re.DOTALL)
_REPEAL_RE = re.compile(
    r"^(?:\d+(?:-[a-z]+)?\.)?\s*"
    r"(?:ARTICOLO|COMMA|ALLEGATO|CAPO|SEZIONE)\s+(?:ABROGAT[OA]|SOPPRESS[OA])\b"
)


# Tokens of marked text that are not words: a deletion ("((...))") and a repeal or suppression
# note ("((COMMA SOPPRESSO DAL ...))"), whose wording Normattiva sometimes rewrites.
DELETION_MARK = "<soppressione>"
NOTE_MARK = "<nota>"
_NOTE_SPAN_RE = re.compile(r"\b(?:ABROGAT[OAIE]|SOPPRESS[OAIE])\b")

Token = tuple[str, bool]  # (lower-cased word, inside a marked span)


def marked_words(raw: str) -> list[Token]:
    """The words of ``raw`` in order, each flagged when it lies inside a ``((...))`` span.

    A deletion span is one ``DELETION_MARK`` token and a repeal note one ``NOTE_MARK`` token. A
    span left open at the end of the text runs to the end; a ``))`` with no opening marker
    closes nothing. Parentheses inside a span are matched (``((art. 3 (codice)))``).
    """
    tokens: list[Token] = []
    position = 0
    for start, end, _ in _scan_spans(raw):
        tokens.extend((word, False) for word in _WORD_RE.findall(raw[position:start].lower()))
        content = raw[start:end]
        words = _WORD_RE.findall(content.lower())
        if not words:
            tokens.append((DELETION_MARK, True))
        elif _NOTE_SPAN_RE.search(content) and content.upper() == content:
            tokens.append((NOTE_MARK, True))
        else:
            tokens.extend((word, True) for word in words)
        position = end
    tokens.extend((word, False) for word in _WORD_RE.findall(raw[position:].lower()))
    return tokens


def ends_inside_span(raw: str) -> bool:
    """True when ``raw`` ends inside a ``((`` span that it does not close."""
    spans = _scan_spans(raw)
    return bool(spans) and not spans[-1][2]


def _scan_spans(raw: str) -> list[tuple[int, int, bool]]:
    """(start, end, closed) of the content of every ``((...))`` span."""
    spans: list[tuple[int, int, bool]] = []
    position = 0
    while (start := raw.find("((", position)) >= 0:
        depth, index, closed = 0, start + 2, False
        while index < len(raw):
            if raw.startswith("((", index):
                break  # a new span opens: this one was not closed
            if raw[index] == "(":
                depth += 1
            elif raw[index] == ")":
                if depth:
                    depth -= 1
                elif raw.startswith("))", index):
                    closed = True
                    break
            index += 1
        spans.append((start + 2, index, closed))
        position = index + 2 if closed else index
    return spans


def has_markers(text: str) -> bool:
    """True when the text contains Normattiva's ``((`` / ``))`` modification markers."""
    return _MARKER_RE.search(text) is not None


def strip_markers(text: str) -> str:
    return _MARKER_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace (newlines included) into one space."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Drop the ``((`` ``))`` markers and normalise whitespace line by line.

    Runs of spaces collapse to one, lines are stripped and empty lines dropped.
    """
    return normalize_lines(strip_markers(text))


def normalize_lines(text: str) -> str:
    """Normalise whitespace line by line, markers kept."""
    lines = (_SPACES_RE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def text_hash(text: str) -> str:
    """sha256 hex digest of the UTF-8 text (the ``unit_versions.text_hash`` column)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(text: str | None) -> str:
    """Hash of the wording only, ignoring markers, whitespace, dots and hyphens.

    Normattiva exported the code as flat text until 2024 and as structured XML afterwards, so
    the same comma can be rendered with different line breaks, label punctuation ("2-bis."
    vs "2 bis.") or quotes. Versioning compares fingerprints so that such layout changes do not
    count as amendments.
    """
    return text_hash(_LAYOUT_RE.sub("", strip_markers(text or "")))


def word_fingerprint(text: str | None) -> str:
    """Hash of the lower-cased words only, all punctuation ignored.

    Two texts with different content fingerprints but the same words differ only in editorial
    details (a comma, a semicolon, a capital letter), not in what they say.
    """
    words = _WORD_RE.findall(strip_markers(text or "").lower())
    return text_hash(" ".join(words))


def clean_heading(raw: str | None) -> str | None:
    """'(( (Procedura di affidamento).))' -> 'Procedura di affidamento'; blank -> None."""
    if raw is None:
        return None
    heading = collapse_whitespace(strip_markers(raw))
    previous = None
    while heading != previous:
        previous = heading
        heading = heading.rstrip(" .;").strip()
        if heading.startswith("(") and heading.endswith(")") and _balanced(heading[1:-1]):
            heading = heading[1:-1].strip()
    return heading or None


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


def cut_notes(text: str) -> str:
    """Remove a trailing Normattiva update note ("------ AGGIORNAMENTO (n) ...")."""
    return _NOTE_RE.sub("", text).rstrip()


def is_note(text: str) -> bool:
    """True for a block that is only an editorial update note."""
    stripped = collapse_whitespace(text)
    return bool(stripped) and not cut_notes(stripped)


def is_repealed(text: str) -> bool:
    """True when the unit only states that it was repealed ("3. COMMA ABROGATO DAL ...")."""
    return _REPEAL_RE.match(collapse_whitespace(strip_markers(text))) is not None
