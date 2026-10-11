"""Search over the seeded corpus: 50/1 changed on 2024-12-31, 225-bis exists only from then."""

from collections.abc import Sequence
from datetime import date

import numpy as np
import psycopg
import pytest

from appalti_qa.config import Settings
from appalti_qa.models import Hit
from appalti_qa.retrieval import FakeEmbedder, embed_missing, passage_text, search

pytestmark = pytest.mark.db

BEFORE = date(2024, 6, 1)  # original text of 50/1, no 225-bis
AFTER = date(2025, 1, 1)  # 50/1 as amended by D.Lgs. 209/2024, 225-bis in force
AMENDMENT_LABEL = "D.Lgs. 31 dicembre 2024, n. 209"


def _keys(hits: Sequence[Hit]) -> list[str]:
    return [hit.unit_key for hit in hits]


@pytest.fixture
def embedded_conn(seeded_conn: psycopg.Connection) -> psycopg.Connection:
    embed_missing(seeded_conn, FakeEmbedder())
    return seeded_conn


class _RoutedEmbedder(FakeEmbedder):
    """Embeds every query as a fixed passage, to make the semantic ranking independent of words."""

    def __init__(self, target_passage: str) -> None:
        super().__init__()
        self.target_passage = target_passage

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_passages([self.target_passage] * len(texts))


def _version_ids(conn: psycopg.Connection, key: str) -> list[int]:
    rows = conn.execute(
        "SELECT v.id FROM unit_versions v JOIN units u ON u.id = v.unit_id"
        " WHERE u.key = %s ORDER BY v.valid_from",
        (key,),
    ).fetchall()
    return [row[0] for row in rows]


def test_lexical_search_returns_the_text_in_force(
    seeded_conn: psycopg.Connection, settings: Settings
) -> None:
    after = search("ingegneria e architettura", AFTER, mode="lexical", conn=seeded_conn)
    assert _keys(after) == ["50/1"]
    hit = after[0]
    assert hit.version_id == _version_ids(seeded_conn, "50/1")[1]
    assert hit.valid_from == date(2024, 12, 31) and hit.valid_to is None
    assert "ingegneria e architettura" in hit.text
    assert [event.label for event in hit.modified_by] == [AMENDMENT_LABEL]
    assert hit.score > 0

    # Before the amendment the comma did not mention these services.
    assert search("ingegneria e architettura", BEFORE, mode="lexical", conn=seeded_conn) == []


def test_units_not_yet_in_force_are_not_found(seeded_conn: psycopg.Connection) -> None:
    assert search("disposizioni transitorie", BEFORE, mode="lexical", conn=seeded_conn) == []
    after = search("disposizioni transitorie", AFTER, mode="lexical", conn=seeded_conn)
    assert _keys(after) == ["225-bis"]


def test_lexical_search_matches_questions_by_term_coverage(
    seeded_conn: psycopg.Connection,
) -> None:
    # No unit contains every word of the question ("qual", "soglia"): partial matches rank by
    # how many query terms they contain.
    question = "Qual è la soglia per l'affidamento diretto dei servizi?"
    hits = search(question, AFTER, mode="lexical", conn=seeded_conn)
    assert _keys(hits)[:2] == ["50/1", "1/1"]
    assert hits[0].score > hits[1].score


def test_lexical_search_keeps_negations(seeded_conn: psycopg.Connection) -> None:
    assert _keys(search("affidamento", AFTER, mode="lexical", conn=seeded_conn)) == ["50/1", "1/1"]
    assert _keys(search("affidamento -lavori", AFTER, mode="lexical", conn=seeded_conn)) == ["1/1"]


def test_repealed_versions_are_not_searched(seeded_conn: psycopg.Connection) -> None:
    seeded_conn.execute(
        "UPDATE unit_versions SET repealed = true"
        " WHERE unit_id = (SELECT id FROM units WHERE key = '225-bis')"
    )
    seeded_conn.commit()
    assert search("disposizioni transitorie", AFTER, mode="lexical", conn=seeded_conn) == []
    cited = search("Cosa prevede l'art. 225-bis?", AFTER, conn=seeded_conn, embedder=FakeEmbedder())
    assert cited == []


def test_semantic_search_uses_the_version_in_force(embedded_conn: psycopg.Connection) -> None:
    query = "affidamento diretto dei servizi di ingegneria e architettura"
    old_id, new_id = _version_ids(embedded_conn, "50/1")
    embedder = FakeEmbedder()

    after = search(query, AFTER, mode="semantic", conn=embedded_conn, embedder=embedder)
    before = search(query, BEFORE, mode="semantic", conn=embedded_conn, embedder=embedder)

    assert (after[0].unit_key, after[0].version_id) == ("50/1", new_id)
    assert (before[0].unit_key, before[0].version_id) == ("50/1", old_id)
    assert "225-bis" in _keys(after) and "225-bis" not in _keys(before)
    scores = [hit.score for hit in after]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)


def test_semantic_search_ignores_other_models_and_missing_embeddings(
    seeded_conn: psycopg.Connection,
) -> None:
    query = "affidamento diretto"
    assert search(query, AFTER, mode="semantic", conn=seeded_conn, embedder=FakeEmbedder()) == []

    embed_missing(seeded_conn, FakeEmbedder())
    other = FakeEmbedder()
    other.model_name = "another-model"
    assert search(query, AFTER, mode="semantic", conn=seeded_conn, embedder=other) == []


def test_hybrid_search_fuses_lexical_and_semantic_rankings(
    embedded_conn: psycopg.Connection, settings: Settings
) -> None:
    principle = embedded_conn.execute(
        "SELECT v.heading, v.text FROM unit_versions v JOIN units u ON u.id = v.unit_id"
        " WHERE u.key = '1/1'"
    ).fetchone()
    embedder = _RoutedEmbedder(passage_text(*principle))
    query = "ingegneria e architettura"
    kwargs = {"conn": embedded_conn, "embedder": embedder, "settings": settings}

    lexical = search(query, AFTER, mode="lexical", **kwargs)
    semantic = search(query, AFTER, mode="semantic", **kwargs)
    hybrid = search(query, AFTER, mode="hybrid", **kwargs)

    assert _keys(lexical) == ["50/1"]
    assert _keys(semantic)[0] == "1/1"
    # 50/1 is in both lists, so it outranks the top semantic hit; every unit is kept.
    assert _keys(hybrid)[:2] == ["50/1", "1/1"]
    assert sorted(_keys(hybrid)) == sorted(_keys(semantic))
    semantic_rank = _keys(semantic).index("50/1") + 1
    rrf_k = settings.rrf_k
    assert hybrid[0].score == pytest.approx(1 / (rrf_k + 1) + 1 / (rrf_k + semantic_rank))
    assert hybrid[1].score == pytest.approx(1 / (rrf_k + 1))


def test_hybrid_search_puts_cited_units_first(
    embedded_conn: psycopg.Connection, settings: Settings
) -> None:
    kwargs = {"conn": embedded_conn, "embedder": FakeEmbedder(), "settings": settings}

    hits = search("Cosa prevede l'art. 225-bis del codice?", AFTER, **kwargs)
    assert hits[0].unit_key == "225-bis"
    assert hits[0].score >= 1.0
    assert hits[0].normattiva_url.endswith("~art225bis!vig=2025-01-01")

    cited = search("termini previsti dal comma 1 dell'articolo 50", AFTER, **kwargs)
    assert cited[0].unit_key == "50/1"
    assert "225-bis" not in _keys(search("art. 225-bis", BEFORE, **kwargs))


def test_hit_fields_and_k(embedded_conn: psycopg.Connection, settings: Settings) -> None:
    kwargs = {"conn": embedded_conn, "embedder": FakeEmbedder(), "settings": settings}
    hits = search("affidamento diretto", AFTER, k=2, **kwargs)
    assert len(hits) == 2
    hit = next(h for h in hits if h.unit_key == "50/1")
    assert hit.heading == "Procedure per l'affidamento"
    assert hit.label == "Art. 50, comma 1"
    assert hit.normattiva_url.endswith(";36~art50!vig=2025-01-01")
    assert len(search("affidamento diretto", AFTER, k=50, **kwargs)) == 4


def test_search_leaves_the_connection_idle(seeded_conn: psycopg.Connection) -> None:
    search("affidamento", AFTER, mode="lexical", conn=seeded_conn)
    assert seeded_conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_search_input_validation(seeded_conn: psycopg.Connection) -> None:
    assert search("   ", AFTER, conn=seeded_conn) == []
    assert search("il la di", AFTER, mode="lexical", conn=seeded_conn) == []
    with pytest.raises(ValueError):
        search("affidamento", AFTER, mode="fuzzy", conn=seeded_conn)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        search("affidamento", AFTER, k=0, conn=seeded_conn)
