"""Retrieval variants compared on a fixed split: the grid, the selection rule and the report.

The method (decision 8 in docs/decisions.md) is fixed before any number is looked at:

1. Questions are split once by id (`dataset.split_of`): odd ids are the development set, even ids
   the test set.
2. Every variant of `default_grid` is evaluated on the development set, and `choose` picks one by
   `SELECTION_RULE`. Nothing is tuned on the test set.
3. The chosen variant and the baseline are then evaluated on the test set; the chosen one replaces
   the baseline as the default only if `beats` holds there (`ADOPTION_RULE`).

Each variant sets every retrieval parameter explicitly, so a sweep does not depend on the
configured defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.metrics import DEFAULT_K_VALUES, Retriever, retrieval_metrics
from appalti_qa.evaluation.runner import new_run_id
from appalti_qa.keys import ref_matches

LEXICAL_WEIGHTS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
FUSION_DEPTHS: tuple[int, ...] = (2, 4, 8)
HUBNESS_WEIGHTS: tuple[float, ...] = (0.0, 0.5, 1.0)  # 0.5 ranks exactly like CSLS
RRF_K = 60
MRR_TIE = 0.01
RECALL_TIE = 0.01
SELECTION_RULE = (
    "Among the hybrid variants, the highest development MRR (comma level) wins; variants within"
    f" {MRR_TIE} of it are separated by development recall@5, and those within {RECALL_TIE} of"
    " the best recall@5 by the fewest parameters changed from the baseline, then by name."
)
ADOPTION_RULE = (
    "The chosen variant becomes the default only if, on the test set, its MRR is higher than the"
    " baseline's and its recall@5 is not lower."
)

BASELINE_PARAMS: dict[str, Any] = {
    "semantic_granularity": "unit",
    "hubness_weight": 0.0,
    "lexical_weight": 1.0,
    "fusion_depth": 4,
    "rrf_k": RRF_K,
}


@dataclass(frozen=True, slots=True)
class Variant:
    """A retrieval mode with explicit settings overrides."""

    name: str
    mode: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def changes(self) -> int:
        """How many parameters differ from the baseline."""
        return sum(self.params.get(key, value) != value for key, value in BASELINE_PARAMS.items())


def variant_name(mode: str, params: Mapping[str, Any]) -> str:
    """'hybrid passage w1.5 d4 h0.5': mode, granularity, then the non-default knobs it uses."""
    parts = [mode, params["semantic_granularity"]]
    if mode == "hybrid":
        parts += [f"w{params['lexical_weight']:g}", f"d{params['fusion_depth']}"]
    if params["hubness_weight"]:
        parts.append(f"h{params['hubness_weight']:g}")
    return " ".join(parts)


def _variant(mode: str, **params: Any) -> Variant:
    full = {**BASELINE_PARAMS, **params}
    return Variant(variant_name(mode, full), mode, full)


BASELINE = _variant("hybrid")


def default_grid() -> list[Variant]:
    """Lexical, the semantic granularities, and the hybrid grid (baseline first among hybrids)."""
    semantic_options = [("unit", 0.0)] + [("passage", h) for h in HUBNESS_WEIGHTS]
    variants = [Variant("lexical", "lexical", dict(BASELINE_PARAMS))]
    variants += [
        _variant("semantic", semantic_granularity=g, hubness_weight=h) for g, h in semantic_options
    ]
    hybrids = [
        _variant(
            "hybrid",
            semantic_granularity=g,
            hubness_weight=h,
            lexical_weight=w,
            fusion_depth=d,
        )
        for g, h in semantic_options
        for w in LEXICAL_WEIGHTS
        for d in FUSION_DEPTHS
    ]
    hybrids.sort(key=lambda v: v.name != BASELINE.name)  # stable: baseline first
    return variants + hybrids


def choose(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """The name of the variant `SELECTION_RULE` picks among the hybrid rows of a sweep."""
    hybrids = [row for row in rows if row["mode"] == "hybrid"]
    if not hybrids:
        return None
    best_mrr = max(row["overall"]["mrr"] for row in hybrids)
    tied = [row for row in hybrids if row["overall"]["mrr"] >= best_mrr - MRR_TIE]
    best_recall = max(row["overall"]["recall@5"] for row in tied)
    tied = [row for row in tied if row["overall"]["recall@5"] >= best_recall - RECALL_TIE]
    return min(tied, key=lambda row: (row["changes"], row["name"]))["name"]


def beats(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    """`ADOPTION_RULE`: a higher MRR and no lower recall@5 than the baseline."""
    new, old = candidate["overall"], baseline["overall"]
    return new["mrr"] > old["mrr"] and new["recall@5"] >= old["recall@5"]


RetrieverFactory = Callable[[Variant], Retriever]


def run_sweep(
    items: Sequence[EvalItem],
    variants: Sequence[Variant],
    make_retriever: RetrieverFactory,
    *,
    split: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Retrieval metrics of every variant on `items`; on the dev split, the chosen variant."""
    now = now or datetime.now(UTC)
    rows = []
    for variant in variants:
        metrics = retrieval_metrics(items, make_retriever(variant), k_values=k_values)
        ranks = {
            record["id"]: _first_rank(record["expected_refs"], record["retrieved"])
            for record in metrics.pop("items")
        }
        rows.append(
            {
                "name": variant.name,
                "mode": variant.mode,
                "params": dict(variant.params),
                "changes": variant.changes,
                "overall": metrics["overall"],
                "by_kind": metrics["by_kind"],
                "first_rank": ranks,
            }
        )
    results: dict[str, Any] = {
        "run_id": new_run_id("sweep", split, now=now),
        "created_at": f"{now:%Y-%m-%dT%H:%M:%SZ}",
        "split": split,
        "dataset": dataset,
        "config": config,
        "selection_rule": SELECTION_RULE,
        "adoption_rule": ADOPTION_RULE,
        "baseline": BASELINE.name,
        "variants": rows,
    }
    if split == "dev":
        results["chosen"] = choose(rows)
    return results


def _first_rank(refs: Sequence[str], retrieved: Sequence[str]) -> int | None:
    return next(
        (rank for rank, key in enumerate(retrieved, 1) if any(ref_matches(r, key) for r in refs)),
        None,
    )


def render_sweep(results: Mapping[str, Any], *, chosen: str | None = None) -> str:
    """Markdown table of a sweep: one row per variant, the baseline and the chosen one marked."""
    chosen = chosen or results.get("chosen")
    rows = results["variants"]
    by_name = {row["name"]: row for row in rows}
    lines = [
        f"# Retrieval sweep `{results['run_id']}`",
        "",
        f"Split **{results['split']}**: {rows[0]['overall']['n'] if rows else 0} questions with"
        f" expected references. Baseline: `{results['baseline']}`.",
        "",
        f"Selection rule (development set only): {results['selection_rule']}",
        "",
        f"Adoption rule: {results['adoption_rule']}",
        "",
    ]
    if chosen:
        lines += [f"Chosen on the development set: `{chosen}`.", ""]
        if chosen in by_name and results["baseline"] in by_name and results["split"] != "dev":
            verdict = beats(by_name[chosen], by_name[results["baseline"]])
            lines += [f"Beats the baseline on this split: **{'yes' if verdict else 'no'}**.", ""]
    header = "| Variant | Recall@1 | Recall@5 | Recall@10 | MRR | Article MRR |"
    lines += [header, "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        o = row["overall"]
        mark = (" (baseline)" if row["name"] == results["baseline"] else "") + (
            " (chosen)" if row["name"] == chosen else ""
        )
        lines.append(
            f"| {row['name']}{mark} | {_f(o.get('recall@1'))} | {_f(o.get('recall@5'))} |"
            f" {_f(o.get('recall@10'))} | {_f(o.get('mrr'))} | {_f(o.get('article_mrr'))} |"
        )
    return "\n".join(lines) + "\n"


def _f(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"
