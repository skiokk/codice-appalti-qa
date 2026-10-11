from collections.abc import Iterator
from datetime import date

import psycopg
import pytest

from appalti_qa.config import get_settings
from appalti_qa.db import close_pool
from appalti_qa.retrieval import available_range, history, lookup

pytestmark = pytest.mark.db

BEFORE = date(2024, 6, 1)
AFTER = date(2025, 1, 1)
AMENDMENT_LABEL = "D.Lgs. 31 dicembre 2024, n. 209"


def test_lookup_matches_refs_at_any_granularity(seeded_conn: psycopg.Connection) -> None:
    (hit,) = lookup(["50/1/b"], BEFORE, conn=seeded_conn)
    assert (hit.unit_key, hit.score) == ("50/1", 1.0)
    assert hit.valid_to == date(2024, 12, 31)
    assert hit.modified_by == []

    (amended,) = lookup(["50"], AFTER, conn=seeded_conn)
    assert amended.version_id != hit.version_id
    assert [event.label for event in amended.modified_by] == [AMENDMENT_LABEL]

    assert [h.unit_key for h in lookup(["225-bis/1"], AFTER, conn=seeded_conn)] == ["225-bis"]
    (annex,) = lookup(["all:I.3"], AFTER, conn=seeded_conn)
    assert annex.unit_key == "all:I.3/p1"
    assert annex.normattiva_url.endswith(";36!vig=2025-01-01")


def test_lookup_returns_distinct_units_in_document_order(seeded_conn: psycopg.Connection) -> None:
    hits = lookup(["50/1", "1", "not a key", "50/1/b", "999"], AFTER, conn=seeded_conn)
    assert [hit.unit_key for hit in hits] == ["1/1", "50/1"]
    assert lookup(["not a key"], AFTER, conn=seeded_conn) == []


def test_lookup_respects_validity_but_keeps_repealed_text(seeded_conn: psycopg.Connection) -> None:
    assert lookup(["225-bis"], BEFORE, conn=seeded_conn) == []
    seeded_conn.execute("UPDATE unit_versions SET repealed = true")
    seeded_conn.commit()
    assert [hit.unit_key for hit in lookup(["225-bis"], AFTER, conn=seeded_conn)] == ["225-bis"]


def test_history_lists_every_version_with_its_amendment(seeded_conn: psycopg.Connection) -> None:
    original, amended = history("50/1/b", conn=seeded_conn)
    assert (original.unit_key, original.kind, original.article, original.comma) == (
        "50/1",
        "comma",
        "50",
        "1",
    )
    assert (original.valid_from, original.valid_to) == (date(2023, 4, 1), date(2024, 12, 31))
    assert original.changed_by is None
    assert (original.change_kind, amended.change_kind) == ("original", "amendment")
    assert amended.valid_to is None and amended.in_force_at(AFTER)
    assert amended.changed_by is not None
    assert (amended.changed_by.label, amended.changed_by.date) == (
        AMENDMENT_LABEL,
        date(2024, 12, 31),
    )
    assert len(amended.text_hash) == 64

    (block,) = history("all:I.3", conn=seeded_conn)
    assert (block.kind, block.allegato, block.heading) == (
        "allegato_block",
        "I.3",
        "Termini delle procedure di appalto e di concessione",
    )
    assert history("nonsense", conn=seeded_conn) == []


def test_history_and_hits_skip_editorial_versions(seeded_conn: psycopg.Connection) -> None:
    seeded_conn.execute(
        "UPDATE unit_versions SET change_kind = 'editorial', changed_by_event = NULL"
        " WHERE valid_from = %s AND unit_id = (SELECT id FROM units WHERE key = '50/1')",
        (date(2024, 12, 31),),
    )
    seeded_conn.commit()
    _, editorial = history("50/1", conn=seeded_conn)
    assert (editorial.change_kind, editorial.changed_by) == ("editorial", None)
    (hit,) = lookup(["50/1"], AFTER, conn=seeded_conn)
    assert hit.modified_by == []


def test_available_range(seeded_conn: psycopg.Connection) -> None:
    assert available_range(conn=seeded_conn) == (date(2023, 4, 1), date(2024, 12, 31))


def test_available_range_on_an_empty_database(db_conn: psycopg.Connection) -> None:
    assert available_range(conn=db_conn) is None


@pytest.fixture
def pool_on_test_db(db_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", db_url)
    close_pool()
    get_settings.cache_clear()
    yield
    close_pool()


@pytest.mark.usefixtures("pool_on_test_db")
def test_functions_use_the_pool_without_a_connection(seeded_conn: psycopg.Connection) -> None:
    assert [hit.unit_key for hit in lookup(["50"], AFTER)] == ["50/1"]
    assert len(history("50/1")) == 2
    assert available_range() == (date(2023, 4, 1), date(2024, 12, 31))
