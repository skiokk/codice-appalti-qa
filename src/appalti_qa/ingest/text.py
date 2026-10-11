"""Text normalisation shared by the Akoma Ntoso parser and the versioning step.

Normattiva marks inserted or replaced text with ``((`` ... ``))``. The stored text drops the
markers; whether a unit carried any is recorded separately (``ParsedUnit.amended``).
"""

from __future__ import annotations

import hashlib
import re

_MARKER_RE = re.compile(r"\(\(|\)\)")
_SPACES_RE = re.compile(r"[ \t\r\f\v\u00a0]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Characters ignored when comparing two renderings of the same text (see content_fingerprint).
_LAYOUT_RE = re.compile(r"[\s.\-]+")
# Editorial notes appended by Normattiva after an article or annex ("------ AGGIORNAMENTO (9) ...").
_NOTE_RE = re.compile(r"-{3,}\s*(?:\)\)\s*)?AGGIORNAMENTO\b.*\Z", re.DOTALL)
_REPEAL_RE = re.compile(
    r"^(?:\d+(?:-[a-z]+)?\.)?\s*"
    r"(?:ARTICOLO|COMMA|ALLEGATO|CAPO|SEZIONE)\s+(?:ABROGAT[OA]|SOPPRESS[OA])\b"
)


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
    lines = (_SPACES_RE.sub(" ", line).strip() for line in strip_markers(text).splitlines())
    return "\n".join(line for line in lines if line)


def text_hash(text: str) -> str:
    """sha256 hex digest of the UTF-8 text (the ``unit_versions.text_hash`` column)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(text: str | None) -> str:
    """Hash of the wording only, ignoring markers, whitespace, dots and hyphens.

    Normattiva exported the code as flat text until 2024 and as structured XML afterwards, so
    the same comma can be rendered with different line breaks or label punctuation ("2-bis."
    vs "2 bis."). Versioning compares fingerprints so that such layout changes do not count as
    amendments.
    """
    return text_hash(_LAYOUT_RE.sub("", strip_markers(text or "")))


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
