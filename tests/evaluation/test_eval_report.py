import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from appalti_qa.evaluation.dataset import EvalItem
from appalti_qa.evaluation.judge import OverlapJudge
from appalti_qa.evaluation.report import (
    published_names,
    render_comparison,
    render_report,
    write_results,
)
from appalti_qa.evaluation.runner import (
    dataset_summary,
    new_run_id,
    run_answer_eval,
    run_retrieval_eval,
)

NOW = datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def answer_results(items: list[EvalItem], stub_answer: Callable, stub_retriever: Callable) -> dict:
    summary = dataset_summary(Path("eval/dataset.yaml"), items, items, include_drafts=True)
    config = {"provider": "fake", "model": "fake", "retrieval_mode": "hybrid", "judge": "overlap"}
    return run_answer_eval(
        items,
        stub_answer,
        OverlapJudge(),
        retrieve=stub_retriever,
        dataset=summary,
        config=config,
        now=NOW,
    )


def test_run_ids() -> None:
    assert new_run_id("fake", "fake", now=NOW) == "20261011T120000Z_fake_fake"
    assert new_run_id("groq", "openai/gpt-oss-120b", now=NOW).endswith("_groq_openai-gpt-oss-120b")
    assert new_run_id("retrieval", None, now=NOW) == "20261011T120000Z_retrieval"


def test_dataset_summary(items: list[EvalItem]) -> None:
    summary = dataset_summary(Path("d.yaml"), items, items[:3], include_drafts=True)
    assert summary == {
        "path": "d.yaml",
        "size": 4,
        "verified": 2,
        "evaluated": 3,
        "evaluated_verified": 2,
        "selection": "all",
    }


def test_run_answer_eval_shape(answer_results: dict) -> None:
    assert answer_results["run_id"] == "20261011T120000Z_fake_fake"
    assert answer_results["created_at"] == "2026-10-11T12:00:00Z"
    assert list(answer_results) == [
        "run_id",
        "created_at",
        "dataset",
        "config",
        "retrieval",
        "answer",
        "items",
    ]
    assert set(answer_results["retrieval"]) == {"overall", "by_kind", "by_category"}
    assert answer_results["answer"]["overall"]["correctness"] == 0.5
    item = answer_results["items"][0]
    assert {"id", "retrieved", "cited", "abstained", "score", "rationale", "latency_ms"} <= set(
        item
    )


def test_run_retrieval_eval(items: list[EvalItem], stub_retriever: Callable) -> None:
    def empty(question: str, at_date: date, k: int) -> list:
        return []

    results = run_retrieval_eval(
        items,
        {"lexical": empty, "hybrid": stub_retriever},
        primary="hybrid",
        dataset={},
        config={"retrieval_mode": "hybrid"},
        k_values=(1, 5),
        now=NOW,
    )
    assert results["run_id"] == "20261011T120000Z_retrieval"
    assert results["retrieval"] is results["retrieval_by_mode"]["hybrid"]
    assert results["retrieval_by_mode"]["lexical"]["overall"]["recall@5"] == 0.0
    first = results["items"][0]
    assert first["id"] == "Q001" and first["expected_refs"] == ["50/1/b"]
    assert first["by_mode"]["hybrid"]["retrieved"] == ["50/2", "50/1", "1/1"]
    assert first["by_mode"]["lexical"]["mrr"] == 0.0
    assert "## Retrieval" in render_report(results)


def test_write_results(tmp_path: Path, answer_results: dict) -> None:
    path = write_results(answer_results, tmp_path / "results")
    assert path.name == "20261011T120000Z_fake_fake.json"
    names = sorted(p.name for p in path.parent.iterdir())
    assert names == [
        "20261011T120000Z_fake_fake.json",
        "20261011T120000Z_fake_fake.md",
        "latest.json",
        "latest.md",
    ]
    assert json.loads((path.parent / "latest.json").read_text()) == json.loads(path.read_text())

    other = write_results({**answer_results, "run_id": "x"}, tmp_path / "other", latest=False)
    assert not (other.parent / "latest.json").exists()


@pytest.fixture
def retrieval_results(items: list[EvalItem], stub_retriever: Callable) -> dict:
    return run_retrieval_eval(
        items,
        {"lexical": stub_retriever, "hybrid": stub_retriever},
        primary="hybrid",
        dataset={},
        config={"retrieval_mode": "hybrid"},
        now=NOW,
    )


def test_publish_retrieval_then_answers(
    tmp_path: Path, retrieval_results: dict, answer_results: dict
) -> None:
    """The per-mode retrieval report stays published after an answer run is published."""
    out = tmp_path / "results"
    assert published_names(retrieval_results, out) == ["latest", "latest_retrieval"]
    write_results(retrieval_results, out)
    assert json.loads((out / "latest.json").read_text())["run_id"].endswith("_retrieval")

    assert published_names(answer_results, out) == ["latest"]
    write_results(answer_results, out)
    assert json.loads((out / "latest.json").read_text())["run_id"].endswith("_fake_fake")
    kept = json.loads((out / "latest_retrieval.json").read_text())
    assert kept["retrieval_by_mode"].keys() == {"lexical", "hybrid"}
    assert "| lexical | 3 |" in (out / "latest_retrieval.md").read_text()


def test_publish_retrieval_keeps_answer_run(
    tmp_path: Path, retrieval_results: dict, answer_results: dict
) -> None:
    """A retrieval run never replaces published answers on the web page."""
    out = tmp_path / "results"
    write_results(answer_results, out)
    assert published_names(retrieval_results, out) == ["latest_retrieval"]
    write_results({**retrieval_results, "run_id": "20261012T000000Z_retrieval"}, out)
    assert json.loads((out / "latest.json").read_text())["answer"]
    published = json.loads((out / "latest_retrieval.json").read_text())
    assert published["run_id"] == "20261012T000000Z_retrieval"
    (out / "latest.json").write_text("{not json", encoding="utf-8")  # unreadable: replaced
    assert published_names(retrieval_results, out) == ["latest", "latest_retrieval"]


def test_render_report(answer_results: dict) -> None:
    answer_results["items"][1]["rationale"] = "Motivo con | barra\nsu due righe"
    report = render_report(answer_results)
    assert report.startswith("# Evaluation report `20261011T120000Z_fake_fake`\n")
    assert "**Draft questions included.** 2 of the evaluated questions" in report
    assert "deterministic fake provider" in report
    assert "lexical-overlap proxy" in report
    assert "| hybrid | 3 | 16.7% | 66.7% | 66.7% | 66.7% | 0.500 |" in report
    assert "### Answers by kind" in report
    assert "| Correctness | 50.0% |" in report
    assert "Motivo con \\| barra su due righe" in report
    assert "| Q002 | temporal | 0.000 | 50/1/b | 1/1 |" in report


def test_render_report_unknown_cost(answer_results: dict) -> None:
    overall = answer_results["answer"]["overall"]
    assert "| Cost of answers | $0.0080 |" in render_report(answer_results)
    overall.update(cost_usd=0.5, cost_unknown=3)
    report = render_report(answer_results)
    assert "| Cost of answers | $0.5000 (3 without a known price, not included) |" in report


def test_render_comparison(answer_results: dict) -> None:
    other = {**answer_results, "run_id": "second", "answer": None}
    table = render_comparison([answer_results, other])
    lines = table.strip().splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("| 20261011T120000Z_fake_fake | fake | fake | hybrid | 4 | 66.7%")
    assert lines[3].startswith("| second |") and "n/a" in lines[3]
