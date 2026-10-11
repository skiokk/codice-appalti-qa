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
    assert all(
        (v.changed_by is None) == (v.valid_from == first_effective) for v in real_corpus.versions
    )


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
