"""Headings are point-in-time: a row that only changes the heading (change_kind 'heading')."""

from datetime import date

import psycopg
import pytest

from appalti_qa.retrieval import history, lookup, search

pytestmark = pytest.mark.db

RENAMED = date(2025, 3, 15)
NEW_HEADING = "Principio del risultato. Governance dei servizi"


@pytest.fixture
def renamed_conn(seeded_conn: psycopg.Connection) -> psycopg.Connection:
    """1/1 keeps its text but gets a new heading on RENAMED, credited to the amendment event."""
    seeded_conn.execute(
        """
        WITH old AS (
            UPDATE unit_versions v SET valid_to = %(renamed)s
            FROM units u WHERE u.id = v.unit_id AND u.key = '1/1'
            RETURNING v.unit_id, v.text, v.text_hash
        )
        INSERT INTO unit_versions (unit_id, heading, text, text_hash, valid_from, change_kind,
                                   changed_by_event)
        SELECT unit_id, %(heading)s, text, text_hash, %(renamed)s, 'heading',
               (SELECT id FROM events WHERE kind = 'amendment')
        FROM old
        """,
        {"renamed": RENAMED, "heading": NEW_HEADING},
    )
    seeded_conn.commit()
    return seeded_conn


def test_lookup_shows_the_heading_in_force(renamed_conn: psycopg.Connection) -> None:
    (before,) = lookup(["1/1"], date(2024, 6, 1), conn=renamed_conn)
    (after,) = lookup(["1/1"], date(2025, 6, 1), conn=renamed_conn)
    assert before.heading == "Principio del risultato"
    assert after.heading == NEW_HEADING
    assert before.version_id != after.version_id
    # Both rows are the same text version: its validity is not cut by the heading change.
    for hit in (before, after):
        assert (hit.valid_from, hit.valid_to) == (date(2023, 4, 1), None)
    assert [event.label for event in after.modified_by] == ["D.Lgs. 31 dicembre 2024, n. 209"]
    assert before.modified_by == []


def test_history_folds_heading_rows_into_the_version(renamed_conn: psycopg.Connection) -> None:
    (version,) = history("1/1", conn=renamed_conn)
    assert (version.valid_from, version.valid_to, version.change_kind) == (
        date(2023, 4, 1),
        None,
        "original",
    )
    assert version.heading == "Principio del risultato"
    (change,) = version.heading_changes
    assert (change.date, change.heading) == (RENAMED, NEW_HEADING)
    assert change.changed_by is not None
    assert version.heading_at(date(2024, 6, 1)) == "Principio del risultato"
    assert version.heading_at(RENAMED) == NEW_HEADING


def test_search_matches_the_heading_in_force(renamed_conn: psycopg.Connection) -> None:
    query = "governance"
    assert search(query, date(2024, 6, 1), mode="lexical", conn=renamed_conn) == []
    (hit,) = search(query, date(2025, 6, 1), mode="lexical", conn=renamed_conn)
    assert (hit.unit_key, hit.heading) == ("1/1", NEW_HEADING)
