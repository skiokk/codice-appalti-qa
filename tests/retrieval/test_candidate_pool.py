"""The candidate pool of reranked answers: three lists fused, pinned units first (decision 9)."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
import pytest

from appalti_qa.config import Settings
from appalti_qa.retrieval import PINNED_SCORE, FakeEmbedder, candidate_pool, embed_missing, search
from appalti_qa.retrieval.fusion import reciprocal_rank_fusion
from appalti_qa.retrieval.passages import build_passages, embed_passages

pytestmark = pytest.mark.db

AFTER = date(2025, 1, 1)
BEFORE = date(2024, 6, 1)


class _CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries = 0

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.queries += len(texts)
        return super().embed_queries(texts)


@pytest.fixture
def pool_conn(seeded_conn: psycopg.Connection) -> psycopg.Connection:
    embed_missing(seeded_conn, FakeEmbedder())
    build_passages(seeded_conn)
    embed_passages(seeded_conn, FakeEmbedder())
    return seeded_conn


def test_pool_fuses_lexical_unit_and_passage_lists(
    pool_conn: psycopg.Connection, settings: Settings
) -> None:
    query, size = "affidamento diretto dei servizi", 10
    embedder = FakeEmbedder()
    passages = settings.model_copy(update={"semantic_granularity": "passage", "hubness_weight": 0})

    def ranked(mode: str, tuned: Settings) -> list[int]:
        hits = search(query, AFTER, size, mode, conn=pool_conn, embedder=embedder, settings=tuned)
        return [hit.version_id for hit in hits]

    expected = reciprocal_rank_fusion(
        [ranked("lexical", settings), ranked("semantic", settings), ranked("semantic", passages)]
    )
    pool = candidate_pool(query, AFTER, size, conn=pool_conn, embedder=embedder, settings=settings)
    assert [hit.version_id for hit in pool] == list(expected)[:size]
    assert [hit.score for hit in pool] == pytest.approx(list(expected.values())[:size])


def test_pool_embeds_the_query_once_and_respects_size_and_date(
    pool_conn: psycopg.Connection, settings: Settings
) -> None:
    embedder = _CountingEmbedder()
    pool = candidate_pool(
        "affidamento", BEFORE, 2, conn=pool_conn, embedder=embedder, settings=settings
    )
    assert embedder.queries == 1
    assert 0 < len(pool) <= 2
    assert all(hit.valid_from <= BEFORE for hit in pool)
    assert all(hit.valid_to is None or hit.valid_to > BEFORE for hit in pool)


def test_cited_units_come_first(pool_conn: psycopg.Connection, settings: Settings) -> None:
    pool = candidate_pool(
        "art. 1, comma 1: affidamento diretto",
        AFTER,
        5,
        conn=pool_conn,
        embedder=FakeEmbedder(),
        settings=settings,
    )
    assert pool[0].unit_key == "1/1" and pool[0].score >= PINNED_SCORE


def test_empty_query_and_bad_size(pool_conn: psycopg.Connection, settings: Settings) -> None:
    assert candidate_pool("  ", AFTER, 5, conn=pool_conn, settings=settings) == []
    with pytest.raises(ValueError):
        candidate_pool("q", AFTER, 0, conn=pool_conn, settings=settings)
