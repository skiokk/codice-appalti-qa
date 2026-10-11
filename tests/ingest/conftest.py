"""Fixtures for the ingest tests.

``fixtures/`` holds three trimmed Normattiva exports of D.Lgs. 36/2023 (articles 1, 6, 50, 109,
225-bis and allegati I.3, I.4 and II.2-bis), see ``fixtures/README.md``:

* 20240302 and 20241231 use the flat format of the 2023-2024 exports (one line per article);
* 20250401 uses the structured format (``<paragraph>``/``<point>`` elements).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from appalti_qa.ingest import ParsedSnapshot, VersionedCorpus, build_versions, parse_snapshot

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_paths() -> list[Path]:
    """The three fixture exports, oldest first."""
    return sorted(FIXTURES.glob("23G00044_*.xml"))


@pytest.fixture(scope="session")
def fixture_snapshots(fixture_paths: list[Path]) -> list[ParsedSnapshot]:
    return [parse_snapshot(path) for path in fixture_paths]


@pytest.fixture(scope="session")
def fixture_corpus(fixture_snapshots: list[ParsedSnapshot]) -> VersionedCorpus:
    return build_versions(fixture_snapshots)
