"""The retrieval sweep: grid, selection rule, adoption rule and the `eval sweep` command."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from appalti_qa.evaluation import cli as eval_cli
from appalti_qa.evaluation.cli import cli_main
from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.sweep import (
    BASELINE,
    BASELINE_PARAMS,
    FUSION_DEPTHS,
    HUBNESS_WEIGHTS,
    LEXICAL_WEIGHTS,
    Variant,
    beats,
    choose,
    default_grid,
    render_sweep,
    run_sweep,
)
from appalti_qa.models import Hit


def _row(name: str, mrr: float, recall5: float, changes: int, mode: str = "hybrid") -> dict:
    return {
        "name": name,
        "mode": mode,
        "changes": changes,
        "overall": {"n": 10, "mrr": mrr, "recall@5": recall5},
    }


def test_grid_has_the_baseline_and_explicit_parameters() -> None:
    grid = default_grid()
    names = [variant.name for variant in grid]
    assert len(names) == len(set(names))
    hybrids = [v for v in grid if v.mode == "hybrid"]
    assert hybrids[0] == BASELINE and BASELINE.changes == 0
    assert BASELINE.name == "hybrid unit w1 d4"
    assert len(hybrids) == (1 + len(HUBNESS_WEIGHTS)) * len(LEXICAL_WEIGHTS) * len(FUSION_DEPTHS)
    assert all(set(v.params) == set(BASELINE_PARAMS) for v in grid)
    assert "hybrid passage w1.5 d8 h0.5" in names
    assert not any(v.params["semantic_granularity"] == "unit" and v.params["hubness_weight"]
                   for v in grid)  # fmt: skip
    assert default_grid() == grid  # deterministic


def test_choose_takes_the_best_dev_mrr() -> None:
    rows = [_row("a", 0.80, 0.90, 0), _row("b", 0.85, 0.80, 2), _row("s", 0.99, 1.0, 1, "semantic")]
    assert choose(rows) == "b"  # semantic-only rows are not candidates


def test_choose_breaks_mrr_ties_by_recall_then_simplicity() -> None:
    rows = [
        _row("baseline", 0.800, 0.80, 0),
        _row("complex", 0.805, 0.85, 3),
        _row("simple", 0.798, 0.845, 1),
    ]
    assert choose(rows) == "simple"  # MRR and recall@5 within the ties, fewer changes
    rows.append(_row("better recall", 0.796, 0.90, 4))
    assert choose(rows) == "better recall"
    assert choose([]) is None


def test_beats_needs_higher_mrr_and_no_lower_recall() -> None:
    baseline = _row("b", 0.80, 0.80, 0)
    assert beats(_row("c", 0.81, 0.80, 1), baseline)
    assert not beats(_row("c", 0.81, 0.79, 1), baseline)
    assert not beats(_row("c", 0.80, 0.90, 1), baseline)


def test_run_sweep_scores_every_variant(
    items: list[EvalItem], stub_retriever: Callable[[str, date, int], list[Hit]]
) -> None:
    variants = [BASELINE, Variant("hybrid other", "hybrid", {**BASELINE_PARAMS, "fusion_depth": 2})]
    seen: list[str] = []

    def make_retriever(variant: Variant) -> Callable[[str, date, int], list[Hit]]:
        seen.append(variant.name)
        return stub_retriever

    results = run_sweep(
        items, variants, make_retriever, split="dev", dataset={}, config={}, k_values=(1, 5)
    )
    assert seen == [v.name for v in variants]
    assert results["chosen"] == BASELINE.name  # equal numbers: the fewest changes
    row = results["variants"][0]
    assert row["overall"]["n"] == 3 and row["changes"] == 0
    assert row["first_rank"] == {"Q001": 2, "Q002": None, "Q004": 1}
    report = render_sweep(results)
    assert "hybrid unit w1 d4 (baseline) (chosen)" in report
    assert "Selection rule" in report
    test_results = {**results, "split": "test"}
    test_results.pop("chosen")
    assert "Beats the baseline on this split: **no**" in render_sweep(
        test_results, chosen="hybrid other"
    )


@pytest.fixture
def sweep_backend(
    monkeypatch: pytest.MonkeyPatch, stub_retriever: Callable
) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    @contextmanager
    def open_db() -> Iterator[None]:
        yield None

    def search_retriever(mode: str, conn: object, settings: Any, embedder: Any = None) -> Callable:
        calls.append((mode, settings))
        return stub_retriever

    monkeypatch.setattr(eval_cli, "_open_db", open_db)
    monkeypatch.setattr(eval_cli, "_search_retriever", search_retriever)
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    return calls


def test_cli_sweep_on_dev_then_test(
    dataset_path: Path,
    tmp_path: Path,
    sweep_backend: list[tuple[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "sweeps"
    args = ["sweep", "--dataset", str(dataset_path), "--all", "--out", str(out), "--k", "1,5"]
    only = ["--only", BASELINE.name, "--only", "hybrid passage w2 d8"]
    assert cli_main([*args, *only]) == 0
    (dev_path,) = out.glob("*_sweep_dev.json")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    assert dev["split"] == "dev" and dev["dataset"]["split"] == "dev"
    assert [row["overall"]["n"] for row in dev["variants"]] == [1, 1]  # Q001 only
    assert dev["chosen"] == BASELINE.name
    settings = sweep_backend[1][1]
    assert (settings.semantic_granularity, settings.lexical_weight, settings.fusion_depth) == (
        "passage",
        2.0,
        8,
    )
    capsys.readouterr()

    test_args = [*args, "--split", "test", "--chosen-from", str(dev_path)]
    assert cli_main([*test_args, *only]) == 0
    (test_path,) = out.glob("*_sweep_test.json")
    test = json.loads(test_path.read_text(encoding="utf-8"))
    assert "chosen" not in test and test["chosen_on_dev"] == BASELINE.name
    assert [row["overall"]["n"] for row in test["variants"]] == [2, 2]  # Q002, Q004
    assert "Beats the baseline on this split: **no**" in capsys.readouterr().out


def test_cli_sweep_rejects_unknown_variants(
    dataset_path: Path, tmp_path: Path, sweep_backend: list, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["sweep", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*args, "--only", "hybrid nothing"]) == 2
    assert "Unknown variants" in capsys.readouterr().err


def test_retrieval_refuses_to_publish_a_split(
    dataset_path: Path, tmp_path: Path, sweep_backend: list, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*args, "--split", "dev", "--publish"]) == 2
    assert "Refusing to publish a split" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.json"))
