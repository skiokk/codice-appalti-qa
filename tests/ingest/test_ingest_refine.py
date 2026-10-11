"""Bisection of the change dates (``appalti-qa download --refine``), with a fake Normattiva.

The fake fetcher writes the "content in force" on a day as plain text and the digest reads it
back, so the search can be checked without parsing exports or touching the network.
"""

import json
import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest

from appalti_qa.ingest.download import snapshot_path
from appalti_qa.ingest.manifest import MANIFEST_NAME, read_manifest
from appalti_qa.ingest.refine import file_digest, refine_snapshots
from appalti_qa.ingest.versioning import content_digest

D0 = date(2025, 1, 1)


def day(offset: int) -> date:
    return D0 + timedelta(days=offset)


class FakeTimeline:
    """Content in force from each change day; records the days fetched."""

    def __init__(self, changes: dict[int, str]) -> None:
        self.changes = sorted((day(offset), content) for offset, content in changes.items())
        self.fetched: list[date] = []

    def content(self, when: date) -> str:
        return [content for start, content in self.changes if start <= when][-1]

    def __call__(self, when: date, path: Path) -> Path:
        self.fetched.append(when)
        path.write_text(self.content(when))
        return path

    def seed(self, raw_dir: Path, *offsets: int) -> None:
        """Files as the lifecycle download leaves them (no probe counted)."""
        for offset in offsets:
            snapshot_path(raw_dir, day(offset)).write_text(self.content(day(offset)))


def text_digest(path: Path) -> str:
    return path.read_text()


def files(raw_dir: Path) -> list[str]:
    return sorted(path.name for path in raw_dir.glob("*.xml"))


def names(*offsets: int) -> list[str]:
    return [f"23G00044_{day(offset):%Y%m%d}.xml" for offset in offsets]


def refine(raw_dir: Path, fetch: Callable[[date, Path], Path], **kwargs):
    return refine_snapshots(raw_dir, fetch, digest=text_digest, **kwargs)


def test_bisection_finds_a_single_change(tmp_path: Path) -> None:
    timeline = FakeTimeline({0: "A", 17: "B"})
    timeline.seed(tmp_path, 0, 40)
    report = refine(tmp_path, timeline)
    assert report.boundaries == [day(0), day(17)]
    assert report.probes == len(timeline.fetched) <= 6  # log2(40) rounded up
    assert files(tmp_path) == names(0, 17)  # the old file of B repeated the boundary
    assert report.unresolved == []
    assert snapshot_path(tmp_path, day(17)).read_text() == "B"


def test_bisection_finds_an_intermediate_version(tmp_path: Path) -> None:
    timeline = FakeTimeline({0: "A", 10: "C", 25: "B"})
    timeline.seed(tmp_path, 0, 40)
    report = refine(tmp_path, timeline)
    assert day(20) in timeline.fetched  # the first probe sees C, unlike both ends
    assert report.boundaries == [day(0), day(10), day(25)]
    assert files(tmp_path) == names(0, 10, 25)


def test_equal_files_are_not_probed_and_duplicates_are_deleted(tmp_path: Path) -> None:
    timeline = FakeTimeline({0: "A", 9: "B"})
    timeline.seed(tmp_path, 0, 5, 9, 12)
    report = refine(tmp_path, timeline)
    assert timeline.fetched == [day(7), day(8)]  # only between 5 (A) and 9 (B)
    assert report.boundaries == [day(0), day(9)]
    assert sorted(report.deleted) == names(5, 7, 8, 12)  # duplicates and probes
    assert files(tmp_path) == names(0, 9)


def test_manifest_avoids_probing_again(tmp_path: Path) -> None:
    timeline = FakeTimeline({0: "A", 17: "B", 50: "C"})
    timeline.seed(tmp_path, 0, 40)
    refine(tmp_path, timeline)
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    assert manifest.verified_through == day(40)
    assert manifest.boundaries == [day(0), day(17)]
    data = json.loads((tmp_path / MANIFEST_NAME).read_text())
    assert data["boundaries"] == ["2025-01-01", "2025-01-18"]

    timeline.fetched.clear()
    timeline.seed(tmp_path, 60)  # a later download
    report = refine(tmp_path, timeline)
    assert all(when > day(17) for when in timeline.fetched)  # 0 -> 17 is already exact
    assert report.boundaries == [day(0), day(17), day(50)]
    assert read_manifest(tmp_path).verified_through == day(60)


def test_probe_limit_is_logged_and_resumed(tmp_path: Path, caplog) -> None:
    timeline = FakeTimeline({0: "A", 17: "B"})
    timeline.seed(tmp_path, 0, 40)
    with caplog.at_level(logging.WARNING, logger="appalti_qa.ingest.refine"):
        report = refine(tmp_path, timeline, max_probes=2)
    assert report.probes == 2
    assert len(report.unresolved) == 1
    lo, hi = report.unresolved[0]
    assert lo < day(17) <= hi and (hi - lo).days > 1
    assert "probe limit (2) reached" in caplog.text
    assert read_manifest(tmp_path).unresolved == report.unresolved
    assert report.boundaries == [day(0), hi]  # best known date kept

    report = refine(tmp_path, timeline)
    assert report.boundaries == [day(0), day(17)]
    assert report.unresolved == []
    assert read_manifest(tmp_path).unresolved == []


def test_probe_files_left_by_an_interrupted_run_are_reused(tmp_path: Path) -> None:
    timeline = FakeTimeline({0: "A", 17: "B"})
    timeline.seed(tmp_path, 0, 20, 40)  # 20 is a probe written before the interruption
    report = refine(tmp_path, timeline)
    assert day(20) not in timeline.fetched
    assert report.boundaries == [day(0), day(17)]


def test_content_digest_ignores_metadata_and_layout(tmp_path: Path, fixture_paths) -> None:
    """Two exports with different bytes (lifecycle, expression date) but the same text."""
    original = fixture_paths[2].read_bytes()
    restamped = original.replace(b'FRBRdate date="2024-12-31"', b'FRBRdate date="2025-03-15"')
    assert restamped != original
    amended = original.replace(b"possono essere", b"devono essere", 1)
    assert amended != original
    paths = []
    for offset, payload in enumerate((original, restamped, amended)):
        path = snapshot_path(tmp_path, day(offset))
        path.write_bytes(payload)
        paths.append(path)
    digests = [file_digest(path) for path in paths]
    assert digests[0] == digests[1] != digests[2]


def test_flat_and_structured_fixtures_have_the_same_digest(fixture_snapshots) -> None:
    flat, structured = fixture_snapshots[1], fixture_snapshots[2]
    assert flat.sha256 != structured.sha256
    assert content_digest(flat) == content_digest(structured)


def test_empty_directory(tmp_path: Path) -> None:
    report = refine(tmp_path, FakeTimeline({0: "A"}))
    assert report.boundaries == [] and report.probes == 0
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.parametrize("payload", ["{", '{"boundaries": []}'])
def test_malformed_manifest_is_ignored(tmp_path: Path, payload: str) -> None:
    (tmp_path / MANIFEST_NAME).write_text(payload)
    assert read_manifest(tmp_path) is None
