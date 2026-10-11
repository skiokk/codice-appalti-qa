"""Parse every Normattiva export in data/raw/akn (skipped when the files are not there)."""

from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import pytest

from appalti_qa.ingest import VersionedCorpus, build_versions, parse_snapshot
from appalti_qa.ingest.commands import snapshot_files
from appalti_qa.ingest.text import content_fingerprint
from appalti_qa.keys import is_valid_key

pytestmark = pytest.mark.slow

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "akn"


@pytest.fixture(scope="module")
def real_corpus() -> VersionedCorpus:
    files = snapshot_files(RAW_DIR) if RAW_DIR.is_dir() else []
    if len(files) < 2:
        pytest.skip(f"Normattiva exports not found in {RAW_DIR}")
    return build_versions([parse_snapshot(path) for path in files])


def test_latest_snapshot_units(real_corpus: VersionedCorpus) -> None:
    latest = real_corpus.snapshots[-1][0]
    articles = {u.article for u in latest.units if u.allegato is None}
    annex_units = [u for u in latest.units if u.allegato is not None]
    assert len(articles) >= 229
    assert len(annex_units) > 1000
    assert all(is_valid_key(u.key) for u in latest.units)
    assert all(u.text and "((" not in u.text and "))" not in u.text for u in latest.units)
    assert len({u.key for u in latest.units}) == len(latest.units)


def test_versions_are_consistent(real_corpus: VersionedCorpus) -> None:
    by_key = defaultdict(list)
    for version in real_corpus.versions:
        by_key[version.unit_key].append(version)
    assert by_key.keys() == {unit.key for unit in real_corpus.units}
    for versions in by_key.values():
        for earlier, later in pairwise(versions):
            assert earlier.valid_to is not None and earlier.valid_to <= later.valid_from
    first_effective = real_corpus.snapshots[0][1]
    for version in real_corpus.versions:
        assert (version.change_kind == "original") == (version.valid_from == first_effective)
        if version.change_kind in ("original", "editorial"):
            assert version.changed_by is None
        elif version.change_kind == "amendment":
            assert version.changed_by is not None
    for versions in by_key.values():
        for earlier, later in pairwise(versions):
            if later.change_kind == "heading":  # same text, new heading, no gap
                assert earlier.valid_to == later.valid_from
                assert content_fingerprint(earlier.text) == content_fingerprint(later.text)
                assert earlier.heading != later.heading


@pytest.mark.parametrize(("article", "last"), [("34", 6), ("41", 15), ("62", 18)])
def test_restarted_numbering_is_repaired(
    real_corpus: VersionedCorpus, article: str, last: int
) -> None:
    """The flat exports up to 2024 number arts. 34 (commi 5-6), 41 (9-10) and 62 (8-9) from
    "1." again; the structured exports confirm the continued numbering."""
    expected = {f"{article}/{n}" for n in range(1, last + 1)}
    for snapshot, _ in real_corpus.snapshots:
        keys = {u.key for u in snapshot.units if u.article == article and u.allegato is None}
        assert expected <= keys, snapshot.source_file
    repaired = {d.article for d in real_corpus.numbering if d.renumbered}
    kept = {d.article for d in real_corpus.numbering if not d.renumbered}
    assert repaired == {"34", "41", "62"}
    assert kept == {"205", "207"}  # the structured exports keep their runs inside a comma


def test_labour_cost_comma_exists_from_the_original_text(real_corpus: VersionedCorpus) -> None:
    """41/13 (costo del lavoro) is in force from 2023-04-01, amended by D.Lgs. 209/2024."""
    versions = [v for v in real_corpus.versions if v.unit_key == "41/13"]
    assert [(v.valid_from.isoformat(), v.change_kind) for v in versions] == [
        ("2023-04-01", "original"),
        ("2024-12-31", "amendment"),
    ]
    assert "il costo del lavoro e' determinato annualmente" in versions[0].text
    assert versions[1].changed_by is not None
    assert versions[1].changed_by.label == "D.Lgs. 31 dicembre 2024, n. 209"


def test_headings_are_point_in_time(real_corpus: VersionedCorpus) -> None:
    """Art. 140 lost "e di protezione civile" from its heading on 2025-05-21 (D.L. 73/2025)."""
    rows = [v for v in real_corpus.versions if v.unit_key == "140/1"][:2]
    assert [(v.valid_from.isoformat(), v.change_kind, v.heading) for v in rows] == [
        ("2023-04-01", "original", "Procedure in caso di somma urgenza e di protezione civile"),
        ("2025-05-21", "heading", "Procedure in caso di somma urgenza"),
    ]
    assert rows[1].changed_by is not None and rows[1].changed_by.label.startswith("D.L. 21 maggio")


def test_export_format_switch_is_not_an_amendment(real_corpus: VersionedCorpus) -> None:
    """The flat (2024) and structured (2025) exports render unchanged commi identically."""
    pairs = list(pairwise(real_corpus.snapshots))
    for (before, _), (after, _) in pairs:
        old = {u.key: u for u in before.units}
        new = {u.key: u for u in after.units}
        shared = old.keys() & new.keys()
        changed = [
            key
            for key in shared
            if content_fingerprint(old[key].text) != content_fingerprint(new[key].text)
        ]
        assert len(shared) > 0.95 * len(old)
        # The largest amendment, the 2024 correttivo, rewrote about 13% of the units.
        assert len(changed) < 0.15 * len(shared), (before.source_file, after.source_file)
