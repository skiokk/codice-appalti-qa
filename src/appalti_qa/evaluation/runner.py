"""Evaluation runs: compute the metrics and assemble the results document."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.judge import Judge
from appalti_qa.evaluation.metrics import (
    DEFAULT_K_VALUES,
    AnswerFn,
    Retriever,
    answer_metrics,
    retrieval_metrics,
)
from appalti_qa.models import Hit


class CachedRetriever:
    """Retrieves once per (question, date) at a fixed depth and serves shorter prefixes.

    Lets the retrieval metrics and the answer generation of a run share the same ranking.
    """

    def __init__(self, retrieve: Retriever, depth: int) -> None:
        self._retrieve = retrieve
        self._depth = depth
        self._cache: dict[tuple[str, date], tuple[int, list[Hit]]] = {}

    def __call__(self, question: str, at_date: date, k: int) -> list[Hit]:
        key = (question, at_date)
        depth, hits = self._cache.get(key, (0, []))
        if k > depth:
            depth = max(k, self._depth)
            hits = self._retrieve(question, at_date, depth)
            self._cache[key] = (depth, hits)
        return hits[:k]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text).strip("-")


def new_run_id(*parts: str | None, now: datetime) -> str:
    """'20261011T120000Z_fake_fake': UTC timestamp then the slugged non-empty parts."""
    return "_".join([f"{now:%Y%m%dT%H%M%SZ}", *(_slug(p) for p in parts if p)])


def dataset_summary(
    path: Path, all_items: Sequence[EvalItem], selected: Sequence[EvalItem], *, include_drafts: bool
) -> dict[str, Any]:
    """How many questions the file has, how many are verified, and which ones were evaluated."""
    return {
        "path": str(path),
        "size": len(all_items),
        "verified": sum(item.verified for item in all_items),
        "evaluated": len(selected),
        "evaluated_verified": sum(item.verified for item in selected),
        "selection": "all" if include_drafts else "verified",
    }


def _header(run_id: str, now: datetime, dataset: dict, config: dict) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "dataset": dataset,
        "config": config,
    }


def run_retrieval_eval(
    items: Sequence[EvalItem],
    retrievers: Mapping[str, Retriever],
    *,
    primary: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Retrieval metrics for each mode; `retrieval` holds the `primary` mode's metrics."""
    now = now or datetime.now(UTC)
    by_mode: dict[str, Any] = {}
    per_item: dict[str, dict[str, Any]] = {}
    for mode, retrieve in retrievers.items():
        metrics = retrieval_metrics(items, retrieve, k_values=k_values)
        for record in metrics.pop("items"):
            fields = ("id", "kind", "category", "expected_refs")
            entry = per_item.setdefault(record["id"], {f: record[f] for f in fields})
            entry.setdefault("by_mode", {})[mode] = {
                key: value for key, value in record.items() if key not in fields
            }
        by_mode[mode] = metrics
    return {
        **_header(new_run_id("retrieval", now=now), now, dataset, config),
        "retrieval": by_mode[primary],
        "retrieval_by_mode": by_mode,
        "items": list(per_item.values()),
    }


def run_answer_eval(
    items: Sequence[EvalItem],
    answer: AnswerFn,
    judge: Judge,
    *,
    retrieve: Retriever | None,
    dataset: dict[str, Any],
    config: dict[str, Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Answer metrics, plus the retrieval metrics of `retrieve` when given.

    `config` must name the "provider" and "model"; they make up the run id. Questions are
    answered first, so each answer's latency includes its retrieval; when `retrieve` is the
    `CachedRetriever` used by `answer`, the retrieval metrics then reuse the same rankings.
    """
    now = now or datetime.now(UTC)
    answers = answer_metrics(items, answer, judge)
    records = answers.pop("items")
    results = _header(
        new_run_id(config["provider"], config["model"], now=now), now, dataset, config
    )
    if retrieve is not None:
        retrieval = retrieval_metrics(items, retrieve, k_values=k_values)
        retrieval.pop("items")
        results["retrieval"] = retrieval
    results["answer"] = answers
    results["items"] = records
    return results
