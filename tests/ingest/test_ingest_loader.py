from datetime import date

import numpy as np
import psycopg
import pytest

from appalti_qa.ingest import VersionedCorpus, load_corpus

pytestmark = pytest.mark.db

IN_FORCE = """
SELECT u.key, v.text, v.repealed, e.amending_label
FROM unit_versions v JOIN units u ON u.id = v.unit_id
LEFT JOIN events e ON e.id = v.changed_by_event
WHERE v.valid_from <= %s AND (v.valid_to IS NULL OR v.valid_to > %s)
ORDER BY u.ordinal
"""
ALL_VERSIONS = """
SELECT u.key, u.ordinal, v.valid_from, v.valid_to, v.text_hash, v.heading, v.repealed,
       e.amending_label
FROM unit_versions v JOIN units u ON u.id = v.unit_id
LEFT JOIN events e ON e.id = v.changed_by_event
ORDER BY u.ordinal, v.valid_from
"""


def in_force(conn: psycopg.Connection, day: date) -> dict[str, tuple]:
    return {row[0]: row[1:] for row in conn.execute(IN_FORCE, (day, day)).fetchall()}


def test_load_corpus(db_conn: psycopg.Connection, fixture_corpus: VersionedCorpus) -> None:
    stats = load_corpus(fixture_corpus, db_conn)
    assert (stats.snapshots, stats.events, stats.units, stats.versions) == (
        3,
        12,
        len(fixture_corpus.units),
        len(fixture_corpus.versions),
    )
    assert stats.embeddings_kept == 0
    act = db_conn.execute("SELECT code, short_title, issued_on FROM acts").fetchone()
    assert act == ("23G00044", "D.Lgs. 36/2023", date(2023, 3, 31))
    snapshots = db_conn.execute(
        "SELECT vigenza_date, effective_date, source_file FROM snapshots ORDER BY vigenza_date"
    ).fetchall()
    assert snapshots[-1] == (date(2025, 4, 1), date(2025, 4, 1), "23G00044_20250401.xml")
    kinds = dict(db_conn.execute("SELECT kind, count(*) FROM events GROUP BY kind").fetchall())
    assert kinds == {"original": 1, "amendment": 11}


def test_point_in_time_queries(db_conn: psycopg.Connection, fixture_corpus) -> None:
    load_corpus(fixture_corpus, db_conn)
    before, after = in_force(db_conn, date(2024, 6, 1)), in_force(db_conn, date(2025, 1, 15))
    assert before["50/3"][0].startswith("3. In sede di prima applicazione del codice")
    assert before["50/3"][1:] == (False, None)
    assert after["50/3"][1:] == (True, "D.Lgs. 31 dicembre 2024, n. 209")
    assert {"109/1", "109/2", "109/3"} <= before.keys()
    assert "109" not in before and "225-bis/1" not in before
    assert "109/1" not in after and after["109"][1] is True
    assert in_force(db_conn, date(2024, 1, 1)) == {}


def test_reload_is_idempotent(db_conn: psycopg.Connection, fixture_corpus) -> None:
    load_corpus(fixture_corpus, db_conn)
    first = db_conn.execute(ALL_VERSIONS).fetchall()
    stats = load_corpus(fixture_corpus, db_conn)
    assert db_conn.execute(ALL_VERSIONS).fetchall() == first
    assert stats.versions == len(first)
    for table in ("acts", "snapshots", "events"):
        count = db_conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count == {"acts": 1, "snapshots": 3, "events": 12}[table]


def test_reload_keeps_embeddings(db_conn: psycopg.Connection, fixture_corpus) -> None:
    load_corpus(fixture_corpus, db_conn)
    vector = np.linspace(-1, 1, 768, dtype=np.float32)
    db_conn.execute(
        "UPDATE unit_versions v SET embedding = %s, embedding_model = 'test-model'"
        " FROM units u WHERE u.id = v.unit_id AND u.key = '50/1'",
        (vector,),
    )
    db_conn.commit()
    stats = load_corpus(fixture_corpus, db_conn)
    assert stats.embeddings_kept == 1
    embedding, model = db_conn.execute(
        "SELECT v.embedding, v.embedding_model FROM unit_versions v"
        " JOIN units u ON u.id = v.unit_id WHERE u.key = '50/1'"
    ).fetchone()
    assert model == "test-model"
    np.testing.assert_allclose(embedding.to_numpy(), vector)  # pgvector loads a Vector
    assert db_conn.execute(
        "SELECT count(*) FROM unit_versions WHERE embedding IS NOT NULL"
    ).fetchone() == (1,)


def test_other_acts_are_untouched(db_conn: psycopg.Connection, fixture_corpus) -> None:
    other = db_conn.execute(
        "INSERT INTO acts (code, urn, title, issued_on) VALUES ('99X99999', 'urn:other', 'Altro',"
        " '2020-01-01') RETURNING id"
    ).fetchone()[0]
    db_conn.execute(
        "INSERT INTO units (act_id, key, kind, article, ordinal)"
        " VALUES (%s, '1', 'article', '1', 1)",
        (other,),
    )
    db_conn.commit()
    load_corpus(fixture_corpus, db_conn)
    load_corpus(fixture_corpus, db_conn)
    assert db_conn.execute("SELECT count(*) FROM units WHERE act_id = %s", (other,)).fetchone() == (
        1,
    )


def test_failed_load_rolls_back(db_conn: psycopg.Connection, fixture_corpus) -> None:
    load_corpus(fixture_corpus, db_conn)
    before = db_conn.execute(ALL_VERSIONS).fetchall()
    broken = VersionedCorpus(
        act=fixture_corpus.act,
        snapshots=fixture_corpus.snapshots,
        events=fixture_corpus.events,
        units=fixture_corpus.units[:1],  # versions reference units that are missing
        versions=fixture_corpus.versions,
    )
    with pytest.raises(KeyError):
        load_corpus(broken, db_conn)
    db_conn.rollback()
    assert db_conn.execute(ALL_VERSIONS).fetchall() == before
