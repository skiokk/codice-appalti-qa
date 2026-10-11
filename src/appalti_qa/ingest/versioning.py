"""Point-in-time versions of every unit, derived by comparing consecutive snapshots.

Rules (see docs/architecture.md, section 4):

1. Snapshots are sorted by ``vigenza_date``; a file byte-identical to the previous one adds no
   information and is skipped.
2. Each kept snapshot has an *effective date*, the date its changes take effect: normally its
   ``vigenza_date``. Normattiva files are requested on the publication date of each amending
   act, while the amendment usually takes effect later; the expression date of the export
   (FRBRExpression) records when the text it contains came into force. When that date falls
   strictly after the last date on which the previous text was observed (the vigenza of the
   preceding file, duplicates included) and strictly before ``vigenza_date``, it is used
   instead: file ``20250404`` carries expression ``2025-04-02``, so its changes apply from
   2 April. A first snapshot whose expression date precedes its vigenza starts on the
   expression date.
3. A unit gets a new version when its wording or heading changes. Wording is compared with
   ``content_fingerprint``, which ignores layout, so the switch from flat (2024) to structured
   (2025) exports does not create versions; an unchanged version keeps the latest rendering of
   its text. The open version is closed (``valid_to``, exclusive) on the new version's
   ``valid_from``. A unit missing from a snapshot is closed on that snapshot's effective date
   and gets a new version if it reappears.
4. ``changed_by`` is the amending act behind a version: among the lifecycle events of the
   snapshot that introduced it, the amendment dated on ``valid_from`` or else the latest one
   dated before it (the lifecycle grows with each amendment, so this is the act the new text
   comes from). Versions from the first snapshot have none.
5. Versions present in the last snapshot stay open (``valid_to`` is None).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import date

from appalti_qa.ingest.akn import ActInfo, ParsedSnapshot
from appalti_qa.ingest.text import content_fingerprint, text_hash
from appalti_qa.ingest.units import ParsedUnit
from appalti_qa.models import Event


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


@dataclass(slots=True)
class VersionedCorpus:
    act: ActInfo
    snapshots: list[tuple[ParsedSnapshot, date]]  # distinct snapshots used, with effective date
    events: list[Event]  # union over all snapshots, by date
    units: list[ParsedUnit]  # latest metadata per key, final ordinals
    versions: list[VersionRecord]  # by unit ordinal, then valid_from


@dataclass(slots=True)
class _Open:
    unit: ParsedUnit  # latest rendering of the version's text
    signature: tuple[str, str]
    valid_from: date
    changed_by: Event | None
    valid_to: date | None = None


def effective_date(snapshot: ParsedSnapshot, previous_vigenza: date | None) -> date:
    """The date from which the text of ``snapshot`` applies (rule 2 above)."""
    expression = snapshot.expression_date
    if expression is None or expression >= snapshot.vigenza_date:
        return snapshot.vigenza_date
    if previous_vigenza is not None and expression <= previous_vigenza:
        return snapshot.vigenza_date
    return expression


def build_versions(snapshots: Sequence[ParsedSnapshot]) -> VersionedCorpus:
    """Merge parsed snapshots into versioned units; raises ValueError when empty."""
    if not snapshots:
        raise ValueError("no snapshots to version")
    kept = _distinct(snapshots)
    events = _union_events(snapshot for snapshot, _ in kept)
    amendments = [event for event in events if event.kind == "amendment"]
    open_versions: dict[str, _Open] = {}
    closed: list[_Open] = []
    for position, (snapshot, effective) in enumerate(kept):
        present = {unit.key: unit for unit in snapshot.units}
        for key, unit in present.items():
            signature = (content_fingerprint(unit.text), content_fingerprint(unit.heading))
            current = open_versions.get(key)
            if current is not None and current.signature == signature:
                current.unit = unit
                continue
            if current is not None:
                current.valid_to = effective
                closed.append(current)
            changed_by = None if position == 0 else _changed_by(effective, snapshot, amendments)
            open_versions[key] = _Open(unit, signature, effective, changed_by)
        for key in [key for key in open_versions if key not in present]:
            gone = open_versions.pop(key)
            gone.valid_to = effective
            closed.append(gone)
    units = _final_units(kept)
    ordinals = {unit.key: unit.ordinal for unit in units}
    versions = sorted(
        (_record(version) for version in [*closed, *open_versions.values()]),
        key=lambda record: (ordinals[record.unit_key], record.valid_from),
    )
    return VersionedCorpus(
        act=kept[-1][0].act, snapshots=kept, events=events, units=units, versions=versions
    )


def _distinct(snapshots: Sequence[ParsedSnapshot]) -> list[tuple[ParsedSnapshot, date]]:
    kept: list[tuple[ParsedSnapshot, date]] = []
    last_seen: date | None = None
    for snapshot in sorted(snapshots, key=lambda s: s.vigenza_date):
        if not kept or snapshot.sha256 != kept[-1][0].sha256:
            kept.append((snapshot, effective_date(snapshot, last_seen)))
        last_seen = snapshot.vigenza_date
    return kept


def _union_events(snapshots: Iterable[ParsedSnapshot]) -> list[Event]:
    events: dict[tuple[str, date], Event] = {}
    for snapshot in snapshots:
        for event in snapshot.events:
            events[(event.amending_urn, event.date)] = event
    return sorted(events.values(), key=lambda event: (event.date, event.amending_urn))


def _changed_by(valid_from: date, snapshot: ParsedSnapshot, fallback: list[Event]) -> Event | None:
    """The amendment behind a version introduced by ``snapshot`` (rule 4 above)."""
    candidates = [e for e in snapshot.events if e.kind == "amendment"] or fallback
    on_date = [event for event in candidates if event.date == valid_from]
    before = [event for event in candidates if event.date <= valid_from]
    chosen = on_date or sorted(before, key=lambda event: event.date)
    return chosen[-1] if chosen else None


def _final_units(kept: list[tuple[ParsedSnapshot, date]]) -> list[ParsedUnit]:
    """Latest metadata of every key, numbered in document order.

    The order is the latest snapshot's; keys that no longer exist are placed after the key that
    preceded them when they were last seen.
    """
    latest: dict[str, ParsedUnit] = {}
    order: list[str] = []
    for snapshot, _ in reversed(kept):
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
    )
