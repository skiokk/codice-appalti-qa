from collections.abc import Callable
from datetime import date

import pytest

from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.judge import OverlapJudge
from appalti_qa.evaluation.metrics import (
    answer_metrics,
    article_key,
    citation_scores,
    ranking_scores,
    retrieval_metrics,
)
from appalti_qa.evaluation.pricing import cost_usd, price_for
from appalti_qa.models import Answer


@pytest.mark.parametrize(
    ("key", "article"),
    [
        ("50/1/b", "50"),
        ("225-bis", "225-bis"),
        ("all:I.1/2/1", "all:I.1/2"),
        ("all:I.3/p3", "all:I.3"),
        ("all:I.3", "all:I.3"),
    ],
)
def test_article_key(key: str, article: str) -> None:
    assert article_key(key) == article


def test_ranking_scores_comma_and_article_level() -> None:
    scores = ranking_scores(["50/1/b"], ["50/2", "50/1", "1/1"], k_values=(1, 3))
    assert scores["recall@1"] == 0.0
    assert scores["recall@3"] == 1.0
    assert scores["mrr"] == 0.5
    # a neighbouring comma of article 50 is ranked first
    assert scores["article_recall@1"] == 1.0
    assert scores["article_mrr"] == 1.0


def test_ranking_scores_article_level_counts_distinct_articles() -> None:
    scores = ranking_scores(["50/1", "50/2", "1/1"], ["50/4"], k_values=(1,))
    assert scores["recall@1"] == 0.0
    assert scores["article_recall@1"] == 0.5  # articles 50 and 1: one of two found
    assert scores["article_hit@1"] == 1.0


def test_retrieval_metrics(items: list[EvalItem], stub_retriever: Callable) -> None:
    calls: list[int] = []

    def retrieve(question: str, at_date: date, k: int) -> list:
        calls.append(k)
        return stub_retriever(question, at_date, k)

    metrics = retrieval_metrics(items, retrieve, k_values=(5, 1, 3, 10))
    assert calls == [10, 10, 10]  # Q003 (no refs) is not retrieved; depth = max k
    overall = metrics["overall"]
    assert overall["n"] == 3
    assert overall["recall@1"] == pytest.approx(0.5 / 3, abs=1e-4)
    assert overall["recall@3"] == pytest.approx(2 / 3, abs=1e-4)
    assert overall["hit@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert overall["mrr"] == 0.5
    assert overall["article_recall@1"] == 0.5
    assert list(metrics["by_kind"]) == ["answerable", "temporal", "multi"]
    assert metrics["by_kind"]["temporal"]["recall@10"] == 0.0
    assert metrics["by_category"]["principi"] == {**metrics["by_kind"]["multi"]}
    assert metrics["items"][0]["retrieved"] == ["50/2", "50/1", "1/1"]
    assert list(overall)[:5] == ["n", "recall@1", "recall@3", "recall@5", "recall@10"]


def test_retrieval_metrics_rejects_bad_k(items: list[EvalItem], stub_retriever: Callable) -> None:
    with pytest.raises(ValueError):
        retrieval_metrics(items, stub_retriever, k_values=(0, 5))


def test_citation_scores() -> None:
    assert citation_scores(["50/1/b", "1/1"], ["50/1", "2/1"]) == (0.5, 0.5)
    assert citation_scores(["50"], ["50/1", "50/2"]) == (1.0, 1.0)
    assert citation_scores(["50/1"], []) == (None, 0.0)
    assert citation_scores([], ["50/1"]) == (None, None)


def test_answer_metrics(items: list[EvalItem], stub_answer: Callable) -> None:
    metrics = answer_metrics(items, stub_answer, OverlapJudge())
    overall = metrics["overall"]
    assert overall["n"] == 4
    assert overall["errors"] == 0
    # Q001 correct, Q002 wrong, Q003 correct abstention, Q004 low overlap
    records = {r["id"]: r for r in metrics["items"]}
    assert records["Q001"]["score"] == 1.0
    assert records["Q002"]["score"] == 0.0
    assert records["Q003"]["score"] == 1.0
    assert records["Q003"]["verdict"] == "correct"
    assert records["Q004"]["citation_recall"] == 0.5
    assert overall["citation_precision"] == pytest.approx((1 + 0 + 1) / 3, abs=1e-4)
    assert overall["citation_recall"] == 0.5
    assert overall["abstention_accuracy"] == 1.0
    assert overall["abstention_rate"] == 0.25
    assert overall["latency_p50_ms"] == 200
    assert overall["latency_p95_ms"] == 400
    assert overall["input_tokens"] == 1000
    assert overall["output_tokens"] == 200
    assert overall["cost_usd"] == pytest.approx(0.008)
    assert metrics["by_kind"]["unanswerable"]["citation_precision"] is None
    assert metrics["by_category"]["affidamento_diretto"]["n"] == 2
    assert records["Q001"]["citations"] == [
        {"unit_key": "50/1", "cited_text": "Testo", "source_index": 0}
    ]


def test_answer_metrics_records_failures(
    items: list[EvalItem], stub_answer: Callable, make_answer: Callable
) -> None:
    def flaky(question: str, at_date: date) -> Answer:
        if question == items[1].question:
            raise TimeoutError("provider timed out")
        return stub_answer(question, at_date)

    class FailingJudge(OverlapJudge):
        def grade(self, item: EvalItem, answer: Answer):
            if item.id == "Q004":
                raise ValueError("no JSON")
            return super().grade(item, answer)

    metrics = answer_metrics(items, flaky, FailingJudge())
    overall = metrics["overall"]
    assert overall["errors"] == 1
    assert overall["judge_errors"] == 1
    records = {r["id"]: r for r in metrics["items"]}
    assert records["Q002"]["error"] == "TimeoutError: provider timed out"
    assert "score" not in records["Q004"] and records["Q004"]["citation_recall"] == 0.5
    assert overall["correctness"] == 1.0  # Q001 and Q003 only


def test_answer_metrics_unknown_price(items: list[EvalItem], make_answer: Callable) -> None:
    """Known costs are summed; answers by a model without a price are counted apart."""
    models = {items[0].id: "claude-opus-5-5", items[1].id: "x", items[3].id: "claude-opus-5"}

    def answer(question: str, at_date: date) -> Answer:
        item = next(item for item in items if item.question == question)
        model = models.get(item.id, "y")
        return make_answer(
            item, "Risposta [1].", ["50/1"], provider="anthropic", model=model, tokens=(10**6, 0)
        )

    overall = answer_metrics(items, answer, OverlapJudge())["overall"]
    assert overall["cost_usd"] == pytest.approx(4.0 + 5.0)
    assert overall["cost_unknown"] == 2
    assert overall["judge_cost_unknown"] == 0

    only_unknown = answer_metrics(items[1:3], answer, OverlapJudge())["overall"]
    assert only_unknown["cost_usd"] is None and only_unknown["cost_unknown"] == 2


def test_answer_metrics_all_priced(items: list[EvalItem], stub_answer: Callable) -> None:
    overall = answer_metrics(items, stub_answer, OverlapJudge())["overall"]
    assert overall["cost_unknown"] == 0


def test_pricing() -> None:
    assert cost_usd("anthropic", "claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert cost_usd("anthropic", "claude-sonnet-5-5", 2_000_000, 0) == 4.0
    assert cost_usd("anthropic", "claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.6)
    assert cost_usd("groq", "openai/gpt-oss-120b", 10**6, 10**6) == 0.0
    assert cost_usd("fake", "fake", 10, 10) == 0.0
    assert price_for("anthropic", "unknown-model") is None


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5", 30.0),
        ("claude-opus-4-8", 30.0),
        ("claude-sonnet-5", 12.0),
        ("claude-fable-5-1", 60.0),
    ],
)
def test_pricing_fallback_models(model: str, expected: float) -> None:
    """Models the server-side refusal fallback may answer with have a price."""
    assert cost_usd("anthropic", model, 1_000_000, 1_000_000) == pytest.approx(expected)
