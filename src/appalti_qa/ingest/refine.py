"""Exact change dates for the snapshots, found by bisection over Normattiva's ``dataVigenza``.

The downloader requests the text in force on each amending act's *publication* date, and the
FRBRExpression date that should say when an export's text came into force is stale in several
exports. Refining finds the dates by asking Normattiva directly: for every pair of consecutive
files whose normalised content differs (``content_digest``, not the raw bytes, which also carry
the lifecycle metadata), the export in force on an intermediate day is downloaded and compared,
halving the interval until the first day of the new content is known.

* a probe equal to the older end moves the lower bound, one equal to the newer end the upper
  bound;
* a probe different from *both* ends is an intermediate version (two acts took effect between
  the files): both halves are searched, so every change point is found.

Afterwards only *boundary* snapshots are kept: the first file of each run of equal content,
named ``23G00044_YYYYMMDD.xml`` by that exact first day. Probes that are not boundaries and
files that repeat the previous boundary are deleted. ``refined.json`` in the same directory
records the boundaries already verified and the last day checked, so later runs (and
``download_snapshots``) do not probe those intervals again.

The search assumes content does not change and then change back between two equal probes;
such a round trip would go unnoticed. The number of downloads is capped; a gap left open by
the cap is logged and recorded, and its newer file is kept as the best known date.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

from appalti_qa.ingest.akn import parse_snapshot, vigenza_from_filename
from appalti_qa.ingest.download import snapshot_files, snapshot_path
from appalti_qa.ingest.manifest import Manifest, read_manifest, write_manifest
from appalti_qa.ingest.versioning import content_digest

logger = logging.getLogger(__name__)

DEFAULT_MAX_PROBES = 150

Fetch = Callable[[date, Path], Path]  # download the export in force on a day to a path
Digest = Callable[[Path], str]


@dataclass(slots=True)
class RefineReport:
    boundaries: list[date]
    probes: int  # exports downloaded
    unresolved: list[tuple[date, date]]
    deleted: list[str]


def file_digest(path: Path) -> str:
    """Normalised content of an export file (see ``content_digest``)."""
    return content_digest(parse_snapshot(path))


class _Search:
    """Bisection state: the files known by day, their digests and the download budget."""

    def __init__(self, raw_dir: Path, fetch: Fetch, digest: Digest, max_probes: int) -> None:
        self.raw_dir = raw_dir
        self.fetch = fetch
        self.digest_of = digest
        self.max_probes = max_probes
        self.files = {vigenza_from_filename(path): path for path in snapshot_files(raw_dir)}
        self.digests: dict[date, str] = {}
        self.probes = 0
        self.unresolved: list[tuple[date, date]] = []

    def digest(self, day: date) -> str:
        if day not in self.digests:
            self.digests[day] = self.digest_of(self.files[day])
        return self.digests[day]

    def probe(self, day: date) -> str | None:
        """Digest of the content in force on ``day``; None when the budget is spent."""
        if day not in self.files:
            path = snapshot_path(self.raw_dir, day)
            if not path.exists():
                if self.probes >= self.max_probes:
                    return None
                self.probes += 1
                logger.info("probe %d: dataVigenza %s", self.probes, f"{day:%Y%m%d}")
                self.fetch(day, path)
            self.files[day] = path
        return self.digest(day)

    def bisect(self, lo: date, hi: date) -> None:
        """Make every change between ``lo`` and ``hi`` (different contents) a one-day step."""
        if (hi - lo).days <= 1:
            return
        middle = lo + (hi - lo) // 2
        found = self.probe(middle)
        if found is None:
            logger.warning(
                "probe limit (%d) reached: the change between %s and %s is dated %s, not refined",
                self.max_probes,
                lo,
                hi,
                hi,
            )
            self.unresolved.append((lo, hi))
            return
        if found == self.digest(lo):
            self.bisect(middle, hi)
        elif found == self.digest(hi):
            self.bisect(lo, middle)
        else:  # an intermediate version: both halves hold a change
            logger.info("intermediate version on %s between %s and %s", middle, lo, hi)
            self.bisect(lo, middle)
            self.bisect(middle, hi)


def refine_snapshots(
    raw_dir: Path,
    fetch: Fetch,
    *,
    max_probes: int = DEFAULT_MAX_PROBES,
    digest: Digest = file_digest,
) -> RefineReport:
    """Date every content change of the files in ``raw_dir`` exactly and keep only boundaries.

    ``fetch(day, path)`` downloads the export in force on ``day`` to ``path``; probes already
    on disk (an interrupted run) are reused without counting against ``max_probes``.
    """
    raw_dir = Path(raw_dir)
    manifest = read_manifest(raw_dir)
    search = _Search(raw_dir, fetch, digest, max_probes)
    if not search.files:
        return RefineReport(boundaries=[], probes=0, unresolved=[], deleted=[])
    for lo, hi in pairwise(sorted(search.files)):
        if manifest is not None and manifest.is_exact(hi):
            continue
        if search.digest(lo) != search.digest(hi):
            search.bisect(lo, hi)

    days = sorted(search.files)
    boundaries = [days[0]]
    deleted = []
    for previous, day in pairwise(days):
        if search.digest(day) == search.digest(previous):
            search.files[day].unlink()
            deleted.append(search.files[day].name)
        else:
            boundaries.append(day)
    unresolved = sorted(set(search.unresolved))
    verified_through = days[-1]
    if manifest is not None:
        verified_through = max(verified_through, manifest.verified_through)
    write_manifest(raw_dir, Manifest(verified_through, boundaries, unresolved))
    logger.info(
        "refined %s: %d boundaries, %d probes, %d files deleted, %d gaps not refined",
        raw_dir,
        len(boundaries),
        search.probes,
        len(deleted),
        len(unresolved),
    )
    return RefineReport(boundaries, search.probes, unresolved, deleted)
