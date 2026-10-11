"""Reranked sources of an answer: merge order, fallback and wiring (no database, no network)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest
from pydantic import SecretStr

from appalti_qa.answer.rerank import RerankingRetriever, merge_ranking
from appalti_qa.answer.service import _default_retriever
from appalti_qa.config import Settings
from appalti_qa.llm import LLMError, ModelUsage
from appalti_qa.llm.reranker import Ranking
from appalti_qa.models import Hit
from appalti_qa.retrieval import PINNED_SCORE

AT = date(2026, 10, 11)
USAGE = ModelUsage("claude-haiku-5-5", 9000, 100)


def hit(key: str, score: float = 0.03) -> Hit:
    return Hit(
        unit_key=key,
        version_id=1,
        text="Testo.",
        score=score,
        valid_from=date(2023, 4, 1),
        normattiva_url="https://www.normattiva.it/",
    )


POOL = [hit("1/1"), hit("2/1"), hit("50/1"), hit("3/1")]


class ScriptedReranker:
    model = "claude-haiku-5-5"

    def __init__(self, order: list[int] | None) -> None:
        self.order = order  # None: fail
        self.calls: list[tuple[str, int, int]] = []

    def rank(
        self, question: str, candidates: Sequence[Hit], *, at_date: date, limit: int
    ) -> Ranking:
        self.calls.append((question, len(candidates), limit))
        if self.order is None:
            raise LLMError("Anthropic reranking failed (HTTP 529)")
        return Ranking(self.order, USAGE)


class Pool:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.sizes: list[int] = []

    def __call__(self, question: str, at_date: date, size: int) -> list[Hit]:
        self.sizes.append(size)
        return self.hits[:size]


def keys(hits: Sequence[Hit]) -> list[str]:
    return [h.unit_key for h in hits]


def test_merge_puts_pinned_then_chosen_then_the_rest() -> None:
    pinned = hit("50/2", PINNED_SCORE + 0.01)
    candidates = [pinned, *POOL]
    # pinned, then the reranker's 3 and 2 (0 is pinned already), then 1 and 4 in pool order
    assert keys(merge_ranking(candidates, [3, 0, 2])) == ["50/2", "50/1", "2/1", "1/1", "3/1"]
    assert keys(merge_ranking(POOL, [])) == keys(POOL)


def test_reranked_sources_follow_the_reranker() -> None:
    reranker = ScriptedReranker([2, 3])
    pool = Pool(POOL)
    retrieve = RerankingRetriever(pool, reranker, pool_size=40, max_ranked=15)
    assert keys(retrieve("Soglia?", AT, 3)) == ["50/1", "3/1", "1/1"]
    assert pool.sizes == [40]
    assert reranker.calls == [("Soglia?", 4, 15)]
    assert retrieve.calls == 1 and retrieve.failures == 0
    assert retrieve.take() == [USAGE] and retrieve.take() == []


def test_pool_is_at_least_k() -> None:
    pool = Pool(POOL)
    RerankingRetriever(pool, ScriptedReranker([]), pool_size=2, max_ranked=15)("q", AT, 3)
    assert pool.sizes == [3]


def test_failure_keeps_the_pool_order() -> None:
    retrieve = RerankingRetriever(Pool(POOL), ScriptedReranker(None), pool_size=40, max_ranked=15)
    assert keys(retrieve("q", AT, 2)) == ["1/1", "2/1"]
    assert retrieve.failures == 1 and retrieve.take() == []


def test_empty_pool_calls_no_model() -> None:
    reranker = ScriptedReranker([0])
    retrieve = RerankingRetriever(Pool([]), reranker, pool_size=40, max_ranked=15)
    assert retrieve("q", AT, 8) == []
    assert reranker.calls == [] and retrieve.calls == 0


@pytest.mark.parametrize(
    ("provider", "key", "mode", "reranks"),
    [
        ("anthropic", "k", "hybrid", True),
        ("anthropic", None, "hybrid", False),  # no key: plain search
        ("none", "k", "hybrid", False),
        ("anthropic", "k", "lexical", False),  # only hybrid answers are reranked
    ],
)
def test_answers_rerank_only_when_configured(
    settings: Settings, provider: str, key: str | None, mode: str, reranks: bool
) -> None:
    configured = settings.model_copy(
        update={"rerank_provider": provider, "anthropic_api_key": SecretStr(key) if key else None}
    )
    assert configured.reranks() == (provider != "none" and key is not None)
    retriever = _default_retriever(configured, mode)  # type: ignore[arg-type]
    assert isinstance(retriever, RerankingRetriever) == reranks
