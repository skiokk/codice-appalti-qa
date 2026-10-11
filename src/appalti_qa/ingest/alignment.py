"""Comma numbering of flat articles, decided against the structured exports.

The flat exports (up to 2024) sometimes number a few commi of an article from "1." again: art. 41
reads "8. ... 1. ... 2. ... 11. ..." where the structured exports have commi 8, 9, 10, 11, and
art. 34 ends with "1.", "2." for commi 5 and 6. The parser cannot tell such a slip from a run
that really belongs inside the previous comma (arts. 205 and 207 quote numbered text in *both*
shapes), so it keeps both readings of such an article (``ParsedSnapshot.renumbered``).

``align_numbering`` decides with the text itself. Each flat article with a restarted run is
compared with the same article in the nearest snapshot that exports it structured (one
``<paragraph>`` per comma, numbered by Normattiva). The renumbered reading is adopted when every
comma it changes exists there and says nearly the same words (word-sequence similarity of at
least ``MIN_SIMILARITY``) and it matches better than the reading as exported. Otherwise, or when
no snapshot exports the article structured, the export's numbering is kept and the decision is
logged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from appalti_qa.ingest.akn import ParsedSnapshot, number_units
from appalti_qa.ingest.units import ParsedUnit

logger = logging.getLogger(__name__)

MIN_SIMILARITY = 0.5  # share of words in common, in order (difflib ratio on word lists)
_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class NumberingDecision:
    """How the restarted numbering of one flat article of one snapshot was read."""

    source_file: str
    article: str
    renumbered: bool  # True: the structured numbering was adopted
    keys: tuple[str, ...]  # keys the renumbered reading adds ("41/9", "41/10", ...)
    reference: str | None  # the structured snapshot compared with
    similarity: float  # lowest similarity of a changed comma (0 without a counterpart)


def align_numbering(
    snapshots: Sequence[ParsedSnapshot],
) -> tuple[list[ParsedSnapshot], list[NumberingDecision]]:
    """The snapshots with each restarted run numbered as the evidence says, and the decisions."""
    aligned: list[ParsedSnapshot] = []
    decisions: list[NumberingDecision] = []
    for snapshot in snapshots:
        if not snapshot.renumbered:
            aligned.append(snapshot)
            continue
        chosen: dict[str, list[ParsedUnit]] = {}
        for article, renumbered in snapshot.renumbered.items():
            decision = _decide(snapshot, article, renumbered, snapshots)
            decisions.append(decision)
            if decision.renumbered:
                chosen[article] = renumbered
        aligned.append(_apply(snapshot, chosen))
    return aligned, decisions


def _decide(
    snapshot: ParsedSnapshot,
    article: str,
    renumbered: list[ParsedUnit],
    snapshots: Sequence[ParsedSnapshot],
) -> NumberingDecision:
    exported = [u for u in snapshot.units if u.allegato is None and u.article == article]
    known = {u.key: u.text for u in exported}
    added = tuple(u.key for u in renumbered if u.key not in known)
    reference = _nearest_structured(snapshot, article, snapshots)
    if reference is None:
        logger.warning(
            "%s: art. %s restarts its numbering and no export has it structured; "
            "kept as exported (%s not created)",
            snapshot.source_file,
            article,
            ", ".join(added),
        )
        return NumberingDecision(snapshot.source_file, article, False, added, None, 0.0)
    counterpart = {
        u.key: u.text for u in reference.units if u.allegato is None and u.article == article
    }
    changed = [u for u in renumbered if known.get(u.key) != u.text]
    alternative = {u.key: u.text for u in renumbered}
    replaced = [u for u in exported if alternative.get(u.key) != u.text]
    scores = [_similarity(u.text, counterpart.get(u.key)) for u in changed]
    kept = [_similarity(u.text, counterpart.get(u.key)) for u in replaced]
    lowest = min(scores, default=0.0)
    adopt = lowest >= MIN_SIMILARITY and _mean(scores) > _mean(kept)
    if adopt:
        logger.info(
            "%s: art. %s numbered as in %s (%s; lowest similarity %.2f)",
            snapshot.source_file,
            article,
            reference.source_file,
            ", ".join(added),
            lowest,
        )
    else:
        logger.warning(
            "%s: art. %s restarts its numbering but %s does not confirm %s "
            "(lowest similarity %.2f); kept as exported",
            snapshot.source_file,
            article,
            reference.source_file,
            ", ".join(added),
            lowest,
        )
    return NumberingDecision(
        snapshot.source_file, article, adopt, added, reference.source_file, lowest
    )


def _nearest_structured(
    snapshot: ParsedSnapshot, article: str, snapshots: Sequence[ParsedSnapshot]
) -> ParsedSnapshot | None:
    candidates = [s for s in snapshots if article in s.structured]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda s: (abs((s.vigenza_date - snapshot.vigenza_date).days), s.vigenza_date),
    )


def _similarity(text: str, other: str | None) -> float:
    if other is None:
        return 0.0
    words = _WORD_RE.findall(text.lower())
    return SequenceMatcher(None, words, _WORD_RE.findall(other.lower()), autojunk=False).ratio()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _apply(snapshot: ParsedSnapshot, chosen: dict[str, list[ParsedUnit]]) -> ParsedSnapshot:
    """The snapshot with the chosen articles' units replaced, in the same document order."""
    units: list[ParsedUnit] = []
    done: set[str] = set()
    for unit in snapshot.units:
        article = unit.article if unit.allegato is None else None
        if article not in chosen:
            units.append(unit)
        elif article not in done:
            units.extend(chosen[article])
            done.add(article)
    return replace(snapshot, units=number_units(units), renumbered={})
