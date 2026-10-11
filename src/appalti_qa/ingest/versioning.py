"""Point-in-time versions of every unit, derived by comparing consecutive snapshots.

Rules (see docs/architecture.md, section 4, and decision 7 in docs/decisions.md):

0. The comma numbering of flat articles that restart it is first decided against the
   structured exports (``alignment.align_numbering``).
1. Snapshots are sorted by ``vigenza_date``. A snapshot whose normalised content
   (``content_digest``: keys, headings and wording of every unit, layout ignored) equals the
   previous one adds no version; it only refreshes the rendering of the open versions.
2. A version takes effect on the ``vigenza_date`` of the snapshot that introduced it, the date in
   its file name. ``appalti-qa download --refine`` keeps one file per distinct content, named by
   the first day Normattiva serves that content, so the file date is the exact date the change
   came into force. The FRBRExpression date of the export is only a sanity check: it is stale in
   several exports (20250401 still says 2024-12-31), and a mismatch is logged.
3. A unit gets a new version when its wording changes (``content_fingerprint``: whitespace,
   dots, hyphens, quotes and markers ignored), so the switch from flat (2024) to structured
   (2025) exports does not create versions. The open version is closed (``valid_to``,
   exclusive) on the new version's ``valid_from``. A unit missing from a snapshot is closed on
   that snapshot's date and gets a new version if it reappears. An unchanged version keeps the
   latest rendering of its text.
   A change of the heading alone is not a new version of the text, but it does close the row
   and open one with the same text and the new heading (``change_kind`` ``heading``): every row
   keeps the heading in force over its interval, so the passage indexed for search (heading and
   text) never changes for a date already covered. A heading rewritten without new words
   (layout only) refreshes the rendering instead.
4. ``change_kind`` says what a version is: ``original`` for the first snapshot; ``editorial``
   when only punctuation or capitalisation changed (same ``word_fingerprint``), when the
   article's text is only split into commi differently (the same words in document order), or
   when the words that changed do not touch Normattiva's amendment markers (``_marked_change``:
   re-layout, words an export dropped or duplicated, a repeal note reworded); ``amendment``
   when they do, or when the unit becomes repealed. ``changed_by`` of an amendment (and of a
   ``heading`` row whose new heading touches markers) is the amending act whose lifecycle
   event is the latest one dated on or before ``valid_from``, taken from the lifecycle of the
   export that introduced it (an act published that day but not yet listed there did not cause
   the change); other rows have none. Doubtful attributions (no act published since the
   previous change, several acts, an act missing from the export's lifecycle) are logged.
5. Versions present in the last snapshot stay open (``valid_to`` is None).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from difflib import SequenceMatcher

from appalti_qa.ingest.akn import ActInfo, ParsedSnapshot
from appalti_qa.ingest.alignment import NumberingDecision, align_numbering
from appalti_qa.ingest.text import (
    Token,
    content_fingerprint,
    marked_words,
    text_hash,
    word_fingerprint,
)
from appalti_qa.ingest.units import ParsedUnit
from appalti_qa.models import ChangeKind, Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VersionRecord:
    unit_key: str
    heading: str | None
    text: str
    text_hash: str
    valid_from: date
    valid_to: date | None
    changed_by: Event | None
    repealed: bool
    change_kind: ChangeKind = "amendment"


@dataclass(slots=True)
class VersionedCorpus:
    act: ActInfo
    snapshots: list[tuple[ParsedSnapshot, date]]  # distinct snapshots used, with effective date
    events: list[Event]  # union over all snapshots, by date
    units: list[ParsedUnit]  # latest metadata per key, final ordinals
    versions: list[VersionRecord]  # by unit ordinal, then valid_from
    numbering: list[NumberingDecision] = field(default_factory=list)  # restarted runs read


@dataclass(slots=True)
class _Open:
    unit: ParsedUnit  # latest rendering of the version's text and heading
    fingerprint: str
    valid_from: date
    changed_by: Event | None
    change_kind: ChangeKind
    valid_to: date | None = None


def content_digest(snapshot: ParsedSnapshot) -> str:
    """Hash of what a snapshot says, independent of the export's layout and metadata.

    Two exports with the same digest have the same unit keys, headings and wording; they may
    differ in line breaks, quotes, ``((...))`` markers or lifecycle metadata.
    """
    digest = hashlib.sha256()
    for unit in snapshot.units:
        for part in (unit.key, content_fingerprint(unit.heading), content_fingerprint(unit.text)):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def effective_date(snapshot: ParsedSnapshot, previous_vigenza: date | None) -> date:
    """The date from which the text of ``snapshot`` applies: its ``vigenza_date`` (rule 2).

    The expression date is compared with it and with ``previous_vigenza`` (the date of the
    previous distinct snapshot) and any disagreement is logged.
    """
    vigenza, expression = snapshot.vigenza_date, snapshot.expression_date
    if expression is None or expression == vigenza:
        return vigenza
    if expression > vigenza:
        logger.warning(
            "%s: expression date %s is after the file date", snapshot.source_file, expression
        )
    elif previous_vigenza is not None and expression <= previous_vigenza:
        logger.info(
            "%s: stale expression date %s (the text changed after %s); using the file date",
            snapshot.source_file,
            expression,
            previous_vigenza,
        )
    else:
        logger.warning(
            "%s: expression date %s precedes the file date: the change may apply earlier "
            "(run `appalti-qa download --refine`)",
            snapshot.source_file,
            expression,
        )
    return vigenza


def build_versions(snapshots: Sequence[ParsedSnapshot]) -> VersionedCorpus:
    """Merge parsed snapshots into versioned units; raises ValueError when empty."""
    if not snapshots:
        raise ValueError("no snapshots to version")
    ordered, numbering = align_numbering(sorted(snapshots, key=lambda s: s.vigenza_date))
    events = _union_events(ordered)
    amendments = [event for event in events if event.kind == "amendment"]
    kept: list[tuple[ParsedSnapshot, date]] = []
    last_digest: str | None = None
    open_versions: dict[str, _Open] = {}
    closed: list[_Open] = []
    for snapshot in ordered:
        digest = content_digest(snapshot)
        present = {unit.key: unit for unit in snapshot.units}
        if digest == last_digest:  # same content: refresh the renderings only
            for key, unit in present.items():
                open_versions[key].unit = unit
            continue
        previous = kept[-1][1] if kept else None
        effective = effective_date(snapshot, previous)
        first = not kept
        kept.append((snapshot, effective))
        last_digest = digest
        credit = _Credit(snapshot, effective, previous, amendments)
        resplit = _resplit_containers([v.unit for v in open_versions.values()], snapshot.units)
        for key, unit in present.items():
            fingerprint = content_fingerprint(unit.text)
            current = open_versions.get(key)
            if current is not None and current.fingerprint == fingerprint:
                if _same_heading(current.unit, unit):
                    current.unit = unit
                    continue
                # The heading alone changed: a new row for the passage, not a new version.
                marked = _touches_marks(
                    current.unit.heading_marked or current.unit.heading or "",
                    unit.heading_marked or unit.heading or "",
                )
                current.valid_to = effective
                closed.append(current)
                open_versions[key] = _Open(
                    unit, fingerprint, effective, credit() if marked else None, "heading"
                )
                continue
            if first:
                kind: ChangeKind = "original"
            elif _container(unit) in resplit or (
                current is not None and _same_words(current.unit, unit)
            ):
                kind = "editorial"
            elif _marked_change(current.unit if current else None, unit):
                kind = "amendment"
            else:
                kind = "editorial"
                logger.debug(
                    "%s: %s changed outside the amendment markers: editorial",
                    snapshot.source_file,
                    key,
                )
            if current is not None:
                current.valid_to = effective
                closed.append(current)
            changed_by = credit() if kind == "amendment" else None
            open_versions[key] = _Open(unit, fingerprint, effective, changed_by, kind)
        for key in [key for key in open_versions if key not in present]:
            gone = open_versions.pop(key)
            gone.valid_to = effective
            closed.append(gone)
    units = _final_units(ordered)
    ordinals = {unit.key: unit.ordinal for unit in units}
    versions = sorted(
        (_record(version) for version in [*closed, *open_versions.values()]),
        key=lambda record: (ordinals[record.unit_key], record.valid_from),
    )
    return VersionedCorpus(
        act=kept[-1][0].act,
        snapshots=kept,
        events=events,
        units=units,
        versions=versions,
        numbering=numbering,
    )


@dataclass(slots=True)
class _Credit:
    """The act credited with the amendments of one snapshot, computed on first use."""

    snapshot: ParsedSnapshot
    effective: date
    previous: date | None
    amendments: list[Event]
    event: Event | None = None
    done: bool = False

    def __call__(self) -> Event | None:
        if not self.done:
            self.event = _attribute(self.snapshot, self.effective, self.previous, self.amendments)
            self.done = True
        return self.event


def _same_heading(before: ParsedUnit, after: ParsedUnit) -> bool:
    return content_fingerprint(before.heading) == content_fingerprint(after.heading)


def _touches_marks(before: str, after: str) -> bool:
    """True when the words that differ between the two texts touch a ``((...))`` span.

    Both texts keep Normattiva's markers. The words are compared in order; a difference touches
    marked text when an inserted or replaced word is marked in ``after``, a removed or replaced
    word was marked in ``before``, or words were removed from inside a span of ``after`` (or
    added inside a span of ``before``), the span closing the text included. A deletion marker
    counts as a word; a repeal note counts as one word whatever its wording ("COMMA SOPPRESSO"
    reworded as "PERIODO SOPPRESSO").
    """
    old, new = marked_words(before), marked_words(after)
    matcher = SequenceMatcher(None, [w for w, _ in old], [w for w, _ in new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if any(marked for _, marked in (*old[i1:i2], *new[j1:j2])):
            return True
        if (tag == "delete" and _inside(new, j1)) or (tag == "insert" and _inside(old, i1)):
            return True
    return False


def _inside(tokens: list[Token], position: int) -> bool:
    """True when the gap before ``tokens[position]`` lies inside a marked span."""
    sides = [tokens[i][1] for i in (position - 1, position) if 0 <= i < len(tokens)]
    return bool(sides) and all(sides)


def _marked_change(before: ParsedUnit | None, after: ParsedUnit) -> bool:
    """True when the export marks the change from ``before`` to ``after`` as an amendment.

    The words that changed touch marked text (``_touches_marks``), or the unit becomes
    repealed. A unit that appears or reappears is an amendment when any of its text is marked.
    """
    if after.repealed and (before is None or not before.repealed):
        return True
    old = (before.marked or before.text) if before is not None else ""
    return _touches_marks(old, after.marked or after.text)


def _same_words(before: ParsedUnit, after: ParsedUnit) -> bool:
    return word_fingerprint(before.text) == word_fingerprint(after.text)


Container = tuple[str | None, str | None]  # (allegato, article): the units split from one text


def _container(unit: ParsedUnit) -> Container:
    return unit.allegato, unit.article


def _resplit_containers(before: list[ParsedUnit], after: list[ParsedUnit]) -> set[Container]:
    """Articles (or allegati) whose text is split differently but says the same words.

    The flat exports sometimes put the first sentence of a comma at the end of the previous one
    (art. 62, commi 14-15) or leave a comma inside the previous one (all. I.7, art. 13-bis); the
    structured exports split them right. Read in document order, the words are the same.
    """

    def words(units: list[ParsedUnit]) -> dict[Container, tuple[str, str]]:
        texts: dict[Container, list[ParsedUnit]] = {}
        for unit in sorted(units, key=lambda unit: unit.ordinal):
            texts.setdefault(_container(unit), []).append(unit)
        return {
            container: (
                word_fingerprint("\n".join(unit.text for unit in members)),
                content_fingerprint("\n".join(f"{unit.key}\n{unit.text}" for unit in members)),
            )
            for container, members in texts.items()
        }

    old, new = words(before), words(after)
    return {
        container
        for container, (same_words, split) in new.items()
        if container in old and old[container][0] == same_words and old[container][1] != split
    }


def _union_events(snapshots: Iterable[ParsedSnapshot]) -> list[Event]:
    events: dict[tuple[str, date], Event] = {}
    for snapshot in snapshots:
        for event in snapshot.events:
            events[(event.amending_urn, event.date)] = event
    return sorted(events.values(), key=lambda event: (event.date, event.amending_urn))


def _attribute(
    snapshot: ParsedSnapshot, valid_from: date, previous: date | None, amendments: list[Event]
) -> Event | None:
    """The amending act behind the changes of ``snapshot`` (rule 4).

    The candidates are the amendments of the export's own lifecycle dated on or before
    ``valid_from`` (all known amendments when it lists none); ``previous`` is the date of the
    previous distinct snapshot, used to flag doubtful attributions.
    """
    known = [event for event in amendments if event.date <= valid_from]
    own = [e for e in snapshot.events if e.kind == "amendment" and e.date <= valid_from]
    candidates = sorted(own, key=lambda event: (event.date, event.amending_urn)) or known
    if not candidates:
        logger.warning("%s: no amending act before %s", snapshot.source_file, valid_from)
        return None
    chosen = candidates[-1]
    if known and known[-1] != chosen:
        logger.info(
            "%s: %s is not in the export's lifecycle yet; changes credited to %s",
            snapshot.source_file,
            known[-1].label,
            chosen.label,
        )
    recent = [event for event in candidates if previous is None or event.date > previous]
    if not recent:
        logger.warning(
            "%s: no amending act published between %s and %s; changes credited to %s (%s)",
            snapshot.source_file,
            previous,
            valid_from,
            chosen.label,
            chosen.date,
        )
    elif len(recent) > 1:
        logger.info(
            "%s: %d acts published since %s; changes credited to the latest, %s",
            snapshot.source_file,
            len(recent),
            previous,
            chosen.label,
        )
    return chosen


def _final_units(snapshots: list[ParsedSnapshot]) -> list[ParsedUnit]:
    """Latest metadata of every key, numbered in document order.

    The order is the latest snapshot's; keys that no longer exist are placed after the key that
    preceded them when they were last seen.
    """
    latest: dict[str, ParsedUnit] = {}
    order: list[str] = []
    for snapshot in reversed(snapshots):
        keys = [unit.key for unit in sorted(snapshot.units, key=lambda unit: unit.ordinal)]
        inserted: dict[str | None, list[str]] = {}
        anchor: str | None = None
        for unit in snapshot.units:
            latest.setdefault(unit.key, unit)
        known = set(order)
        for key in keys:
            if key in known:
                anchor = key
            else:
                inserted.setdefault(anchor, []).append(key)
                known.add(key)
        merged = list(inserted.get(None, []))
        for key in order:
            merged.append(key)
            merged.extend(inserted.get(key, []))
        order = merged
    return [replace(latest[key], ordinal=index) for index, key in enumerate(order, start=1)]


def _record(version: _Open) -> VersionRecord:
    unit = version.unit
    return VersionRecord(
        unit_key=unit.key,
        heading=unit.heading,
        text=unit.text,
        text_hash=text_hash(unit.text),
        valid_from=version.valid_from,
        valid_to=version.valid_to,
        changed_by=version.changed_by,
        repealed=unit.repealed,
        change_kind=version.change_kind,
    )
