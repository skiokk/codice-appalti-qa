"""Sources of an answer reordered by a reranker over a wider candidate pool (decision 9)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date

from appalti_qa.llm.base import LLMError, ModelUsage
from appalti_qa.llm.reranker import Reranker
from appalti_qa.models import Hit
from appalti_qa.retrieval.search import PINNED_SCORE

logger = logging.getLogger(__name__)

Pool = Callable[[str, date, int], list[Hit]]  # (question, at_date, size), best first


def merge_ranking(candidates: Sequence[Hit], order: Sequence[int]) -> list[Hit]:
    """Pinned candidates (cited in the question), then `order`, then the rest in pool order."""
    pinned = [i for i, hit in enumerate(candidates) if hit.score >= PINNED_SCORE]
    final = dict.fromkeys([*pinned, *order, *range(len(candidates))])
    return [candidates[i] for i in final]


class RerankingRetriever:
    """A retriever whose candidates come from `pool` and are reordered by `reranker`.

    When the reranker fails, the pool order is kept and the failure is counted. `take` returns
    the tokens the reranker consumed since the previous `take`.
    """

    def __init__(self, pool: Pool, reranker: Reranker, *, pool_size: int, max_ranked: int) -> None:
        self.pool = pool
        self.reranker = reranker
        self.pool_size = pool_size
        self.max_ranked = max_ranked
        self.calls = 0
        self.failures = 0
        self._usage: list[ModelUsage] = []

    def __call__(self, question: str, at_date: date, k: int) -> list[Hit]:
        candidates = self.pool(question, at_date, max(self.pool_size, k))
        if not candidates:
            return []
        self.calls += 1
        try:
            ranking = self.reranker.rank(
                question, candidates, at_date=at_date, limit=self.max_ranked
            )
        except LLMError as exc:
            self.failures += 1
            logger.warning("Reranking failed, keeping the retrieval order: %s", exc)
            return candidates[:k]
        self._usage.append(ranking.usage)
        return merge_ranking(candidates, ranking.order)[:k]

    def take(self) -> list[ModelUsage]:
        usage, self._usage = self._usage, []
        return usage
