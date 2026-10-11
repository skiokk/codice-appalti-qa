"""Retrieval and answer metrics, overall and grouped by question kind and category."""

from __future__ import annotations

import logging
import math
import operator
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date
from statistics import fmean
from typing import TYPE_CHECKING, Any, get_args

from appalti_qa.evaluation.dataset import EvalItem, Kind
from appalti_qa.evaluation.judge import Judge
from appalti_qa.evaluation.pricing import cost_usd, usage_cost
from appalti_qa.keys import make_key, parse_key, ref_matches
from appalti_qa.models import Answer, Hit

if TYPE_CHECKING:
    from appalti_qa.llm import ModelUsage

logger = logging.getLogger(__name__)

Retriever = Callable[[str, date, int], list[Hit]]  # (question, at_date, k)
AnswerFn = Callable[[str, date], Answer]
# The per-model tokens of the model calls made since the previous call (e.g. UsageMeter.take).
UsageFn = Callable[[], Sequence["ModelUsage"]]
Record = dict[str, Any]
Aggregator = Callable[[list[Record]], Any]

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
_KIND_ORDER = get_args(Kind)


# --- aggregation ------------------------------------------------------------------------------


def _mean(field: str) -> Aggregator:
    def aggregate(records: list[Record]) -> float | None:
        values = [r[field] for r in records if r.get(field) is not None]
        return round(fmean(values), 4) if values else None

    return aggregate


def _total(field: str) -> Aggregator:
    """Sum of the known values of the field.

    A None value (a cost without a known price) is left out of the sum and counted by
    `_unknown`; the total is None only when the field is set and no value is known.
    """

    def aggregate(records: list[Record]) -> float | int | None:
        values = [r[field] for r in records if field in r]
        known = [value for value in values if value is not None]
        if values and not known:
            return None
        total = sum(known)
        return round(total, 6) if isinstance(total, float) else total

    return aggregate


def _unknown(field: str) -> Aggregator:
    """How many records have the field set to None."""
    return lambda records: sum(1 for r in records if field in r and r[field] is None)


def _percentile(field: str, q: float) -> Aggregator:
    """Nearest-rank percentile."""

    def aggregate(records: list[Record]) -> int | None:
        values = sorted(r[field] for r in records if r.get(field) is not None)
        if not values:
            return None
        return values[max(0, math.ceil(q / 100 * len(values)) - 1)]

    return aggregate


def _count(field: str) -> Aggregator:
    return lambda records: sum(1 for r in records if r.get(field))


def summarize(records: list[Record], aggregators: Mapping[str, Aggregator]) -> dict[str, Any]:
    """{"overall", "by_kind", "by_category"}; each group has "n" plus one value per aggregator."""

    def group(rows: list[Record]) -> dict[str, Any]:
        return {"n": len(rows), **{name: agg(rows) for name, agg in aggregators.items()}}

    def by(field: str, order: Sequence[str] = ()) -> dict[str, Any]:
        groups: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        rank = {name: i for i, name in enumerate(order)}
        keys = sorted(groups, key=lambda name: (rank.get(name, len(rank)), name))
        return {key: group(groups[key]) for key in keys}

    return {
        "overall": group(records),
        "by_kind": by("kind", _KIND_ORDER),
        "by_category": by("category"),
    }


def _item_fields(item: EvalItem) -> Record:
    return {
        "id": item.id,
        "kind": item.kind,
        "category": item.category,
        "expected_refs": list(item.expected_refs),
    }


# --- retrieval --------------------------------------------------------------------------------


def article_key(key: str) -> str:
    """The article a unit or ref belongs to: '50/1/b' -> '50'; 'all:I.1/2/1' -> 'all:I.1/2'.

    Allegato blocks without articles map to the allegato itself ('all:I.3/p3' -> 'all:I.3').
    """
    parsed = parse_key(key)
    if parsed.article:
        return make_key(parsed.article, annex=parsed.annex)
    return make_key(annex=parsed.annex)


def _ranks(
    expected: Iterable[str], retrieved: Sequence[str], matches: Callable[[str, str], bool]
) -> list[int | None]:
    """1-based rank of the first retrieved key matching each expected ref (None if missing)."""
    return [
        next((rank for rank, key in enumerate(retrieved, start=1) if matches(ref, key)), None)
        for ref in expected
    ]


def ranking_metric_names(k_values: Sequence[int]) -> list[str]:
    names: list[str] = []
    for prefix in ("", "article_"):
        names += [f"{prefix}recall@{k}" for k in k_values]
        names += [f"{prefix}hit@{k}" for k in k_values]
        names.append(f"{prefix}mrr")
    return names


def ranking_scores(
    refs: Sequence[str], retrieved: Sequence[str], k_values: Sequence[int] = DEFAULT_K_VALUES
) -> dict[str, float]:
    """recall@k, hit@k and reciprocal rank at comma level and at article level.

    Comma level: a ref matches a retrieved unit by `keys.ref_matches` ('50/1/b' is found by unit
    '50/1'). Article level: the article of each distinct expected ref is looked up among the
    articles of the retrieved units, so a neighbouring comma of the right article counts.
    """
    levels = {
        "": _ranks(refs, retrieved, ref_matches),
        "article_": _ranks(
            dict.fromkeys(article_key(ref) for ref in refs),
            [article_key(key) for key in retrieved],
            operator.eq,
        ),
    }
    scores: dict[str, float] = {}
    for prefix, ranks in levels.items():
        found = {k: sum(1 for r in ranks if r is not None and r <= k) for k in k_values}
        scores.update({f"{prefix}recall@{k}": found[k] / len(ranks) for k in k_values})
        scores.update({f"{prefix}hit@{k}": float(found[k] > 0) for k in k_values})
        best = min((r for r in ranks if r is not None), default=None)
        scores[f"{prefix}mrr"] = 1.0 / best if best else 0.0
    return scores


def retrieval_metrics(
    items: Sequence[EvalItem],
    retrieve: Retriever,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Retrieval quality over the items with expected refs, each at its own `vigenza` date.

    Returns {"overall", "by_kind", "by_category", "items"}; metric names are listed by
    `ranking_metric_names` (comma-level `recall@k`, `hit@k`, `mrr`, then the `article_` ones).
    """
    k_values = sorted(set(k_values))
    if not k_values or k_values[0] < 1:
        raise ValueError("k values must be positive integers")
    depth = k_values[-1]
    records: list[Record] = []
    for item in items:
        if not item.expected_refs:
            continue
        hits = retrieve(item.question, item.vigenza, depth)
        retrieved = list(dict.fromkeys(hit.unit_key for hit in hits))[:depth]
        records.append(
            {
                **_item_fields(item),
                "retrieved": retrieved,
                **ranking_scores(item.expected_refs, retrieved, k_values),
            }
        )
    names = ranking_metric_names(k_values)
    return {**summarize(records, {name: _mean(name) for name in names}), "items": records}


# --- answers ----------------------------------------------------------------------------------

ANSWER_AGGREGATORS: dict[str, Aggregator] = {
    "correctness": _mean("score"),
    "citation_precision": _mean("citation_precision"),
    "citation_recall": _mean("citation_recall"),
    "abstention_accuracy": _mean("abstention_correct"),
    "abstention_rate": _mean("abstained"),
    "latency_p50_ms": _percentile("latency_ms", 50),
    "latency_p95_ms": _percentile("latency_ms", 95),
    "input_tokens": _total("input_tokens"),
    "output_tokens": _total("output_tokens"),
    "cost_usd": _total("cost_usd"),
    "cost_unknown": _unknown("cost_usd"),  # answers by a model without a known price
    "judge_input_tokens": _total("judge_input_tokens"),
    "judge_output_tokens": _total("judge_output_tokens"),
    "judge_cost_usd": _total("judge_cost_usd"),
    "judge_cost_unknown": _unknown("judge_cost_usd"),
    "errors": _count("error"),
    "judge_errors": _count("judge_error"),
}


def citation_scores(
    expected_refs: Sequence[str], cited: Sequence[str]
) -> tuple[float | None, float | None]:
    """(precision, recall) of the cited unit keys against the expected refs.

    Both are None for questions without expected refs; precision is None when nothing is cited.
    """
    if not expected_refs:
        return None, None
    recall = sum(any(ref_matches(ref, key) for key in cited) for ref in expected_refs)
    precision = sum(any(ref_matches(ref, key) for ref in expected_refs) for key in cited)
    return (precision / len(cited) if cited else None), recall / len(expected_refs)


def answer_cost(answer: Answer, usage: Sequence[ModelUsage] = ()) -> float | None:
    """Cost of an answer in USD, or None when a model it used has no known price.

    `usage` lists the tokens of each model attempt behind the answer, priced at that model's
    rate; without it, all the answer's tokens are priced at the rate of the model that wrote it.
    """
    if usage:
        return usage_cost(answer.provider, usage)
    return cost_usd(answer.provider, answer.model, answer.input_tokens, answer.output_tokens)


def _answer_fields(
    item: EvalItem, answer: Answer, elapsed_ms: int, usage: Sequence[ModelUsage]
) -> Record:
    cited = answer.cited_keys
    precision, recall = citation_scores(item.expected_refs, cited)
    return {
        "retrieved": [hit.unit_key for hit in answer.sources],
        "cited": cited,
        "abstained": answer.abstained,
        "abstention_correct": answer.abstained == (item.kind == "unanswerable"),
        "citation_precision": precision,
        "citation_recall": recall,
        "latency_ms": answer.latency_ms or elapsed_ms,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "cost_usd": answer_cost(answer, usage),
        "answer": answer.text,
        "citations": [
            {"unit_key": c.unit_key, "cited_text": c.cited_text, "source_index": c.source_index}
            for c in answer.citations
        ],
    }


def answer_metrics(
    items: Sequence[EvalItem], answer: AnswerFn, judge: Judge, *, usage: UsageFn | None = None
) -> dict[str, Any]:
    """Answer each item at its `vigenza` date and score it.

    Correctness is the judge score (0 / 0.5 / 1); citation precision and recall compare the
    cited units with the expected refs (questions with refs only); abstention accuracy checks
    that exactly the unanswerable questions are declined. A failing answer or judge call is
    recorded on the item ("error" / "judge_error") and left out of the affected averages.
    `usage`, called around each answer, returns the per-model tokens of the calls made for it, so
    that an answer served by a fallback model pays for every attempt (see `answer_cost`).
    Returns {"overall", "by_kind", "by_category", "items"}.
    """
    records: list[Record] = []
    for position, item in enumerate(items, start=1):
        record = _item_fields(item)
        records.append(record)
        if usage is not None:
            usage()  # count from this question on
        started = time.perf_counter()
        # One failing question (rate limit, timeout) must not abort a paid run.
        try:
            result = answer(item.question, item.vigenza)
        except Exception as exc:
            logger.warning("%s: answer failed: %s", item.id, exc)
            record["error"] = f"{type(exc).__name__}: {exc}"
            continue
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        calls = usage() if usage is not None else ()
        record.update(_answer_fields(item, result, elapsed_ms, calls))
        try:
            verdict = judge.grade(item, result)
        except Exception as exc:
            logger.warning("%s: judge failed: %s", item.id, exc)
            record["judge_error"] = f"{type(exc).__name__}: {exc}"
            continue
        record.update(
            score=verdict.score,
            verdict=verdict.label,
            rationale=verdict.rationale,
            judge_input_tokens=verdict.input_tokens,
            judge_output_tokens=verdict.output_tokens,
            judge_cost_usd=verdict.cost_usd,
        )
        logger.info(
            "%s (%d/%d): %s, %d ms",
            item.id,
            position,
            len(items),
            verdict.label,
            record["latency_ms"],
        )
    return {**summarize(records, ANSWER_AGGREGATORS), "items": records}
