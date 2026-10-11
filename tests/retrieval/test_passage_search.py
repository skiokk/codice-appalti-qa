"""Passages in the database and passage-level (max-sim) semantic search on the seeded corpus."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
import pytest

from appalti_qa.config import Settings
from appalti_qa.retrieval import FakeEmbedder, embed_missing, passage_text, search
from appalti_qa.retrieval.passages import (
    PASSAGE_SCHEME,
    build_passages,
    embed_passages,
    split_passages,
    update_hub_scores,
)

pytestmark = pytest.mark.db

AFTER = date(2025, 1, 1)
BEFORE = date(2024, 6, 1)
SEED_VERSIONS = 5


class _CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.passages: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        self.passages.extend(texts)
        return super().embed_passages(texts)


class _RoutedEmbedder(FakeEmbedder):
    """Embeds every query as a fixed passage."""

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_passages([self.target] * len(texts))


def _rows(conn: psycopg.Connection, key: str, day: date) -> tuple[int, str | None, str]:
    return conn.execute(
        "SELECT v.id, v.heading, v.text FROM unit_versions v JOIN units u ON u.id = v.unit_id"
        " WHERE u.key = %s AND v.valid_from <= %s AND (v.valid_to IS NULL OR v.valid_to > %s)",
        (key, day, day),
    ).fetchone()


@pytest.fixture
def passage_conn(seeded_conn: psycopg.Connection) -> psycopg.Connection:
    embed_missing(seeded_conn, FakeEmbedder())
    build_passages(seeded_conn)
    embed_passages(seeded_conn, FakeEmbedder())
    return seeded_conn


def _settings(settings: Settings, **update: object) -> Settings:
    return settings.model_copy(update={"semantic_granularity": "passage", **update})


def test_build_passages_splits_every_version_once(seeded_conn: psycopg.Connection) -> None:
    assert build_passages(seeded_conn) == SEED_VERSIONS
    rows = seeded_conn.execute(
        "SELECT version_id, ordinal, text, scheme FROM unit_passages ORDER BY version_id, ordinal"
    ).fetchall()
    version_id, heading, text = _rows(seeded_conn, "50/1", AFTER)
    expected = split_passages(heading, text)
    assert [r[2] for r in rows if r[0] == version_id] == expected
    assert expected[0] == passage_text(heading, text) and len(expected) == 4  # intro, a), b)
    assert {r[3] for r in rows} == {PASSAGE_SCHEME}
    assert build_passages(seeded_conn) == 0  # idempotent


def test_build_passages_rebuilds_another_scheme(seeded_conn: psycopg.Connection) -> None:
    build_passages(seeded_conn)
    seeded_conn.execute("UPDATE unit_passages SET scheme = 'old' WHERE ordinal = 0")
    seeded_conn.commit()
    assert build_passages(seeded_conn) == SEED_VERSIONS
    assert seeded_conn.execute("SELECT DISTINCT scheme FROM unit_passages").fetchall() == [
        (PASSAGE_SCHEME,)
    ]


def test_embed_passages_reuses_unit_embeddings_and_dedupes_texts(
    seeded_conn: psycopg.Connection,
) -> None:
    embed_missing(seeded_conn, FakeEmbedder())
    build_passages(seeded_conn)
    embedder = _CountingEmbedder()
    embedded = embed_passages(seeded_conn, embedder, batch_size=2)

    texts = [r[0] for r in seeded_conn.execute("SELECT text FROM unit_passages").fetchall()]
    wholes = {r[0] for r in seeded_conn.execute("SELECT text FROM unit_passages WHERE ordinal = 0")}
    assert embedded == len(embedder.passages) == len(set(texts) - wholes)
    assert not wholes & set(embedder.passages)  # whole units took the unit embedding
    assert len(set(embedder.passages)) == len(embedder.passages)  # each text once
    # The two versions of 50/1 share the intro and lettera a): fewer texts than rows.
    assert len(set(texts)) < len(texts)
    missing = seeded_conn.execute(
        "SELECT count(*) FROM unit_passages WHERE embedding IS NULL OR embedding_model <> 'fake'"
    ).fetchone()[0]
    assert missing == 0
    assert embed_passages(seeded_conn, embedder) == 0


def test_passage_search_scores_a_version_by_its_best_passage(
    passage_conn: psycopg.Connection, settings: Settings
) -> None:
    version_id, heading, text = _rows(passage_conn, "50/1", AFTER)
    lettera_b = split_passages(heading, text)[-1]
    embedder = _RoutedEmbedder(lettera_b)

    by_passage = search(
        "q", AFTER, k=4, mode="semantic", conn=passage_conn, embedder=embedder,
        settings=_settings(settings),
    )  # fmt: skip
    by_unit = search(
        "q", AFTER, k=4, mode="semantic", conn=passage_conn, embedder=embedder, settings=settings
    )
    assert by_passage[0].unit_key == "50/1" and by_passage[0].version_id == version_id
    assert by_passage[0].score == pytest.approx(1.0, abs=1e-5)  # the lettera itself
    unit_score = next(hit.score for hit in by_unit if hit.unit_key == "50/1")
    assert unit_score < by_passage[0].score
    assert len({hit.unit_key for hit in by_passage}) == len(by_passage)  # one hit per version


def test_passage_search_filters_by_date(
    passage_conn: psycopg.Connection, settings: Settings
) -> None:
    old_id, heading, text = _rows(passage_conn, "50/1", BEFORE)
    embedder = _RoutedEmbedder(split_passages(heading, text)[-1])
    hits = search(
        "q", BEFORE, k=10, mode="semantic", conn=passage_conn, embedder=embedder,
        settings=_settings(settings),
    )  # fmt: skip
    assert hits[0].version_id == old_id
    assert "225-bis" not in [hit.unit_key for hit in hits]  # not in force yet


def test_hubness_weight_lowers_hubs(passage_conn: psycopg.Connection, settings: Settings) -> None:
    _, heading, text = _rows(passage_conn, "1/1", AFTER)
    embedder = _RoutedEmbedder(passage_text(heading, text))
    passage_conn.execute(
        "UPDATE unit_passages p SET hub_score = 0.9 FROM unit_versions v JOIN units u"
        " ON u.id = v.unit_id WHERE v.id = p.version_id AND u.key = '1/1'"
    )
    passage_conn.commit()
    plain = search(
        "q", AFTER, k=4, mode="semantic", conn=passage_conn, embedder=embedder,
        settings=_settings(settings),
    )  # fmt: skip
    corrected = search(
        "q", AFTER, k=4, mode="semantic", conn=passage_conn, embedder=embedder,
        settings=_settings(settings, hubness_weight=1.0),
    )  # fmt: skip
    assert plain[0].unit_key == "1/1" and plain[0].score == pytest.approx(1.0, abs=1e-5)
    score = next(hit.score for hit in corrected if hit.unit_key == "1/1")
    assert score == pytest.approx(0.1, abs=1e-5)


def test_update_hub_scores_fills_every_passage(passage_conn: psycopg.Connection) -> None:
    assert (
        update_hub_scores(passage_conn, FakeEmbedder())
        == passage_conn.execute("SELECT count(*) FROM unit_passages").fetchone()[0]
    )
    nulls = passage_conn.execute(
        "SELECT count(*) FROM unit_passages WHERE hub_score IS NULL"
    ).fetchone()[0]
    assert nulls == 0


def test_passage_granularity_falls_back_to_units_without_passages(
    seeded_conn: psycopg.Connection, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    embed_missing(seeded_conn, FakeEmbedder())
    hits = search(
        "affidamento diretto lavori", AFTER, mode="semantic", conn=seeded_conn,
        embedder=FakeEmbedder(), settings=_settings(settings),
    )  # fmt: skip
    assert hits and "No passage embeddings" in caplog.text


def test_hybrid_with_zero_lexical_weight_follows_semantic_order(
    passage_conn: psycopg.Connection, settings: Settings
) -> None:
    tuned = _settings(settings, lexical_weight=0.0)
    query = "affidamento diretto dei servizi di ingegneria"
    args = {"conn": passage_conn, "settings": tuned, "embedder": FakeEmbedder()}
    semantic = search(query, AFTER, k=4, mode="semantic", **args)
    hybrid = search(query, AFTER, k=4, mode="hybrid", **args)
    assert len(semantic) == 4  # every unit in force
    assert [h.unit_key for h in hybrid] == [h.unit_key for h in semantic]
