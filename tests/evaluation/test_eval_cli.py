import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import psycopg
import pytest
from openpyxl import load_workbook

from appalti_qa.config import Settings, get_settings
from appalti_qa.evaluation import cli as eval_cli
from appalti_qa.evaluation import cli_main
from appalti_qa.evaluation.dataset import EvalItem, load_dataset
from appalti_qa.evaluation.judge import OverlapJudge
from appalti_qa.evaluation.xlsx import SHEET_REVIEW
from appalti_qa.models import Answer, Hit


@pytest.fixture
def no_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands that need a connection get None; retrieval goes to the stub retriever."""

    @contextmanager
    def open_db() -> Iterator[None]:
        yield None

    monkeypatch.setattr(eval_cli, "_open_db", open_db)


def _run_file(out: Path) -> Path:
    """The results file of the only answer run written to `out`."""
    (path,) = (p for p in out.glob("*.json") if p.name != "latest.json")
    return path


@pytest.fixture
def stub_backends(
    monkeypatch: pytest.MonkeyPatch, no_db: None, stub_retriever: Callable, stub_answer: Callable
) -> list[str]:
    modes: list[str] = []

    def search_retriever(mode: str, conn: object, settings: object) -> Callable:
        modes.append(mode)
        return stub_retriever

    def answer_backend(settings, provider_name, model, retrieve, *, k, mode, delay_s, **_):
        return eval_cli.AnswerBackend(stub_answer, provider_name, model or "fake")

    monkeypatch.setattr(eval_cli, "_search_retriever", search_retriever)
    monkeypatch.setattr(eval_cli, "_answer_backend", answer_backend)
    return modes


@pytest.fixture
def coverage(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, set[date]]]:
    """Every version has an embedding; records the (model, days) of each check."""
    calls: list[tuple[str, set[date]]] = []

    def embedding_coverage(
        conn: object, model: str, days: Iterable[date], *, passages: bool = False
    ) -> tuple[int, int]:
        calls.append((model, set(days)))
        return 2811, 2811

    monkeypatch.setattr(eval_cli, "embedding_coverage", embedding_coverage)
    return calls


def _set_coverage(monkeypatch: pytest.MonkeyPatch, embedded: int, total: int) -> None:
    monkeypatch.setattr(
        eval_cli, "embedding_coverage", lambda conn, model, days, **_: (embedded, total)
    )


def test_validate(dataset_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert cli_main(["validate", "--dataset", str(dataset_path)]) == 0
    assert "4 questions, 2 verified; static checks passed" in capsys.readouterr().out


def test_validate_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "- id: Q001\n  question: q\n  kind: multi\n  category: c\n  vigenza: '2025-01-01'\n"
        "  expected_answer: a\n  expected_refs: ['50/1']\n  source_excerpt: x\n"
        "  difficulty: easy\n",
        encoding="utf-8",
    )
    assert cli_main(["validate", "--dataset", str(path)]) == 1
    err = capsys.readouterr().err
    assert "Q001: multi questions need at least two expected_refs" in err
    assert "1 problem(s) found." in err
    assert cli_main(["validate", "--dataset", str(tmp_path / "missing.yaml")]) == 1


@pytest.mark.db
def test_validate_with_db(
    monkeypatch: pytest.MonkeyPatch,
    seeded_conn: psycopg.Connection,
    dataset_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    @contextmanager
    def open_db() -> Iterator[psycopg.Connection]:
        yield seeded_conn

    monkeypatch.setattr(eval_cli, "_open_db", open_db)
    argv = ["validate", "--db", "--dataset", str(dataset_path)]
    assert cli_main(argv) == 0
    assert "static and corpus checks passed" in capsys.readouterr().out
    # Q002 quotes the text in force before the 2024-12-31 amendment
    text = dataset_path.read_text(encoding="utf-8")
    dataset_path.write_text(text.replace('"2024-06-01"', '"2025-06-01"'), encoding="utf-8")
    assert cli_main(argv) == 1
    err = capsys.readouterr().err
    assert "Q002: source_excerpt not found in the text of ['50/1/b'] in force on 2025-06-01" in err


@pytest.mark.db
def test_export_with_texts_in_force(
    monkeypatch: pytest.MonkeyPatch,
    seeded_conn: psycopg.Connection,
    dataset_path: Path,
    tmp_path: Path,
) -> None:
    @contextmanager
    def open_db() -> Iterator[psycopg.Connection]:
        yield seeded_conn

    monkeypatch.setattr(eval_cli, "_open_db", open_db)
    path = tmp_path / "review.xlsx"
    assert cli_main(["export-xlsx", str(path), "--dataset", str(dataset_path)]) == 0
    ws = load_workbook(path)[SHEET_REVIEW]
    column = next(cell.column for cell in ws[1] if cell.value == "Testo vigente")
    assert ws.cell(row=2, column=column).value.startswith("[Art. 50, comma 1]\n1. ")
    assert ws.cell(row=4, column=column).value in (None, "")  # unanswerable: no refs


def test_database_unreachable(
    monkeypatch: pytest.MonkeyPatch, dataset_path: Path, capsys: pytest.CaptureFixture
) -> None:
    @contextmanager
    def open_db() -> Iterator[None]:
        raise psycopg.OperationalError("connection refused\ndetails")
        yield

    monkeypatch.setattr(eval_cli, "_open_db", open_db)
    assert cli_main(["validate", "--db", "--dataset", str(dataset_path)]) == 1
    assert "Database not reachable: connection refused" in capsys.readouterr().err


def test_retrieval_command(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "results"
    args = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(out)]
    assert cli_main([*args, "--mode", "lexical", "--mode", "hybrid", "--k", "1,5"]) == 0
    assert stub_backends == ["lexical", "hybrid"]
    (path,) = out.glob("*_retrieval.json")
    results = json.loads(path.read_text())
    assert results["config"]["modes"] == ["lexical", "hybrid"]
    assert results["config"]["retrieval_mode"] == "hybrid"
    assert results["dataset"]["selection"] == "all"
    assert not (out / "latest.json").exists()
    printed = capsys.readouterr().out
    assert "retrieval [lexical] n=3 recall@1=0.167 recall@5=0.667 mrr=0.500" in printed


def test_retrieval_command_publish(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str]
) -> None:
    """Lexical search uses no embeddings: nothing to check before publishing."""
    out = tmp_path / "results"
    args = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(out)]
    assert cli_main([*args, "--mode", "lexical", "--publish"]) == 0
    latest = json.loads((out / "latest.json").read_text())
    assert latest["retrieval_by_mode"].keys() == {"lexical"}
    assert (out / "latest_retrieval.json").read_text() == (out / "latest.json").read_text()
    assert (out / "latest_retrieval.md").exists()


def test_retrieval_publish_checks_embeddings(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    coverage: list,
    capsys: pytest.CaptureFixture,
) -> None:
    out = tmp_path / "results"
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(out), "--publish"]
    assert cli_main(argv) == 0
    days = {date(2025, 1, 1), date(2024, 6, 1)}  # the dates of the evaluated questions
    assert coverage == [(get_settings().embedding_model, days)]
    assert json.loads((out / "latest_retrieval.json").read_text())["retrieval_by_mode"].keys() == {
        "lexical",
        "semantic",
        "hybrid",
    }
    printed = capsys.readouterr().out
    assert f"Published: {out / 'latest'}.json and .md" in printed
    assert f"Published: {out / 'latest_retrieval'}.json and .md" in printed


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_retrieval_publish_refused_without_embeddings(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    mode: str,
) -> None:
    out = tmp_path / "results"
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(out), "--publish"]
    _set_coverage(monkeypatch, 2780, 2811)  # 98.9%
    assert cli_main([*argv, "--mode", "lexical", "--mode", mode]) == 2
    err = capsys.readouterr().err
    assert "Refusing to publish: 2,780 of the 2,811 versions in force" in err
    assert "(98.9%) have an embedding from intfloat/multilingual-e5-base" in err
    assert "appalti-qa embed" in err
    _set_coverage(monkeypatch, 0, 0)  # empty corpus
    assert cli_main([*argv, "--mode", mode]) == 2
    assert "0 of the 0 versions" in capsys.readouterr().err
    assert stub_backends == []  # refused before any search
    assert not out.exists()

    assert cli_main([*argv[:-1], "--mode", mode]) == 0  # without --publish the run is written
    assert not (out / "latest.json").exists() and not (out / "latest_retrieval.json").exists()


def test_retrieval_publish_refused_with_fake_embedder(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    coverage: list,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    get_settings.cache_clear()
    out = tmp_path / "results"
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(out), "--publish"]
    assert cli_main(argv) == 2
    assert "Refusing to publish: EMBEDDING_MODEL is 'fake'" in capsys.readouterr().err
    assert coverage == [] and not out.exists()


def test_retrieval_publish_keeps_published_answers(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    coverage: list,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Retrieval and answer runs published in either order leave each other's report in place."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    monkeypatch.setattr(eval_cli, "get_judge", lambda settings: OverlapJudge())  # no network
    common = ["--dataset", str(dataset_path), "--out", str(tmp_path), "--publish"]
    retrieval = ["retrieval", *common, "--all"]
    answers = ["answers", *common, "--provider", "anthropic", "--judge-provider", "anthropic"]
    assert cli_main(retrieval) == 0
    assert cli_main(answers) == 0
    assert json.loads((tmp_path / "latest.json").read_text())["answer"]
    assert "retrieval_by_mode" in json.loads((tmp_path / "latest_retrieval.json").read_text())
    capsys.readouterr()
    assert cli_main(retrieval) == 0
    assert json.loads((tmp_path / "latest.json").read_text())["answer"]
    assert f"Kept {tmp_path / 'latest'}.json: it holds an answer run." in capsys.readouterr().out


class _ScriptedReranker:
    """Puts the last candidate first; records the questions it ranked."""

    model = "claude-haiku-5-5"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def rank(self, question: str, candidates, *, at_date: date, limit: int):
        from appalti_qa.llm import ModelUsage
        from appalti_qa.llm.reranker import Ranking

        self.questions.append(question)
        return Ranking([len(candidates) - 1], ModelUsage("claude-haiku-5-5", 10_000, 100))


@pytest.fixture
def stub_rerank(monkeypatch: pytest.MonkeyPatch, stub_retriever: Callable) -> _ScriptedReranker:
    from appalti_qa.answer.rerank import RerankingRetriever

    reranker = _ScriptedReranker()

    def pool(question: str, at_date: date, size: int) -> list[Hit]:
        # pool scores are RRF scores, below PINNED_SCORE: nothing is pinned
        hits = stub_retriever(question, at_date, size)
        return [hit.model_copy(update={"score": 0.03}) for hit in hits]

    def rerank_retriever(conn: object, settings: Settings, embedder: object = None):
        return RerankingRetriever(
            pool,
            reranker,
            pool_size=settings.rerank_pool,
            max_ranked=settings.rerank_max,
        )

    monkeypatch.setattr(eval_cli, "_rerank_retriever", rerank_retriever)
    return reranker


def test_retrieval_rerank_needs_a_key(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*argv, "--rerank"]) == 2
    assert "Reranking needs ANTHROPIC_API_KEY." in capsys.readouterr().err
    assert not list(tmp_path.glob("*.json"))


def test_retrieval_rerank_adds_a_mode_with_its_cost(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    stub_rerank: _ScriptedReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*argv, "--mode", "hybrid", "--rerank"]) == 0
    (path,) = tmp_path.glob("*_retrieval.json")
    results = json.loads(path.read_text())
    assert results["retrieval_by_mode"].keys() == {"hybrid", "rerank"}
    assert results["config"]["modes"] == ["hybrid", "rerank"]
    assert results["config"]["rerank_model"] == "claude-haiku-5-5"
    assert results["retrieval"] == results["retrieval_by_mode"]["hybrid"]  # primary unchanged
    assert len(stub_rerank.questions) == 3  # the questions with expected refs
    assert results["rerank"] == {
        "model": "claude-haiku-5-5",
        "pool": 40,
        "max_ranked": 15,
        "calls": 3,
        "failures": 0,
        "input_tokens": 30_000,
        "output_tokens": 300,
        "cost_usd": pytest.approx(3 * (10_000 * 0.10 + 100 * 0.50) / 1e6),
    }
    q001 = next(item for item in results["items"] if item["id"] == "Q001")
    hybrid, reranked = q001["by_mode"]["hybrid"], q001["by_mode"]["rerank"]
    assert reranked["retrieved"] == ["1/1", "50/2", "50/1"]  # the last candidate moved first
    assert hybrid["mrr"] == 0.5 and reranked["mrr"] == pytest.approx(1 / 3)


def test_retrieval_rerank_publish_checks_unit_and_passage_embeddings(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    stub_rerank: _ScriptedReranker,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    checked: list[bool] = []

    def embedding_coverage(conn, model, days, *, passages=False):
        checked.append(passages)
        return (2811, 2811) if not passages else (0, 2811)

    monkeypatch.setattr(eval_cli, "embedding_coverage", embedding_coverage)
    argv = ["retrieval", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*argv, "--mode", "lexical", "--rerank", "--publish"]) == 2
    assert checked == [False, True]
    assert "have passage embeddings from" in capsys.readouterr().err


def test_answers_rerank_their_sources_and_count_the_cost(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    stub_rerank: _ScriptedReranker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(tmp_path), "--ids", "Q001"]
    assert cli_main([*argv, "--rerank"]) == 0
    results = json.loads(_run_file(tmp_path).read_text())
    assert results["config"]["rerank_model"] == "claude-haiku-5-5"
    assert stub_backends == []  # no plain search retriever: answers rerank like the app
    assert results["rerank"]["calls"] == 1  # the stub answers do not retrieve; the metrics do
    assert results["rerank"]["failures"] == 0


def test_reranked_answers_publish_checks_passage_embeddings(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    stub_rerank: _ScriptedReranker,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    monkeypatch.setattr(eval_cli, "get_judge", lambda settings: OverlapJudge())  # no network
    monkeypatch.setattr(
        eval_cli,
        "embedding_coverage",
        lambda conn, model, days, *, passages=False: (0, 2811) if passages else (2811, 2811),
    )
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(tmp_path), "--publish"]
    argv += ["--provider", "anthropic", "--judge-provider", "anthropic", "--rerank"]
    assert cli_main(argv) == 2
    assert "have passage embeddings from" in capsys.readouterr().err
    assert not (tmp_path / "latest.json").exists()


def test_answers_command(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "results"
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(out), "--ids", "Q001,Q002"]
    assert cli_main(argv) == 0
    results = json.loads(_run_file(out).read_text())
    assert [item["id"] for item in results["items"]] == ["Q001", "Q002"]
    assert results["dataset"]["selection"] == "verified"
    assert results["config"]["provider"] == "fake" and results["config"]["judge"] == "overlap"
    assert results["retrieval"]["overall"]["n"] == 2
    assert _run_file(out).with_suffix(".md").read_text().startswith("# Evaluation report")
    assert not (out / "latest.json").exists() and not (out / "latest.md").exists()
    assert "answers n=2 correctness=0.500" in capsys.readouterr().out


def test_answers_falls_back_to_fake_without_key(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    argv = ["run", "--dataset", str(dataset_path), "--out", str(tmp_path), "--provider", "groq"]
    assert cli_main(argv) == 0
    assert "No API key for groq: answering with the fake provider." in capsys.readouterr().err
    assert not (tmp_path / "latest.json").exists()


def test_answers_publish_refused_with_fake_provider(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "results"
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(out), "--publish"]
    assert cli_main([*argv, "--provider", "anthropic"]) == 2
    err = capsys.readouterr().err
    assert "Refusing to publish: the answers and the judge would use the fake provider" in err
    assert not out.exists()  # refused before running anything


def test_answers_publish_refused_with_fake_judge(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(tmp_path), "--publish"]
    assert cli_main([*argv, "--provider", "anthropic"]) == 2
    assert "Refusing to publish: the judge would use the fake provider" in capsys.readouterr().err
    assert not (tmp_path / "latest.json").exists()


def test_answers_publish(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    coverage: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    monkeypatch.setattr(eval_cli, "get_judge", lambda settings: OverlapJudge())  # no network
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(tmp_path), "--publish"]
    assert cli_main([*argv, "--provider", "anthropic", "--judge-provider", "anthropic"]) == 0
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["config"]["provider"] == "anthropic"
    assert (tmp_path / "latest.md").exists()
    assert not (tmp_path / "latest_retrieval.json").exists()
    assert [model for model, _ in coverage] == ["intfloat/multilingual-e5-base"]

    # lexical answers need no embeddings
    coverage.clear()
    assert cli_main([*argv, "--provider", "anthropic", "--judge-provider", "anthropic"]) == 0
    assert len(coverage) == 1
    assert (
        cli_main(
            [*argv, "--provider", "anthropic", "--judge-provider", "anthropic", "--mode", "lexical"]
        )
        == 0
    )
    assert len(coverage) == 1


def test_answers_publish_refused_without_embeddings(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()
    monkeypatch.setattr(eval_cli, "get_judge", lambda settings: OverlapJudge())  # no network
    _set_coverage(monkeypatch, 100, 2811)
    out = tmp_path / "results"
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(out), "--publish"]
    assert cli_main([*argv, "--provider", "anthropic", "--judge-provider", "anthropic"]) == 2
    assert "Refusing to publish: 100 of the 2,811 versions" in capsys.readouterr().err
    assert stub_backends == [] and not out.exists()


def test_answers_retrieve_like_production(
    tmp_path: Path,
    dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_db: None,
    stub_retriever: Callable,
    items: list[EvalItem],
) -> None:
    """Answers get search(question, date, k, mode) as in the app; metrics make their own calls."""
    calls: list[tuple[str, str, int]] = []

    def search_retriever(mode: str, conn: object, settings: object) -> Callable:
        def retrieve(question: str, at_date: date, k: int) -> list[Hit]:
            calls.append((mode, question, k))
            return stub_retriever(question, at_date, k)

        return retrieve

    def answer_backend(settings, provider_name, model, retrieve, *, k, mode, delay_s, **_):
        def answer(question: str, at_date: date) -> Answer:
            hits = retrieve(question, at_date, k)
            return Answer(
                question=question,
                at_date=at_date,
                text="-",
                sources=hits,
                abstained=True,
                model="fake",
                provider="fake",
            )

        return eval_cli.AnswerBackend(answer, provider_name, "fake")

    monkeypatch.setattr(eval_cli, "_search_retriever", search_retriever)
    monkeypatch.setattr(eval_cli, "_answer_backend", answer_backend)
    argv = ["answers", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main([*argv, "--k", "2", "--mode", "lexical"]) == 0

    by_question = {item.question: item for item in items}
    answer_calls = [(by_question[q].id, k) for mode, q, k in calls if k == 2]
    metric_calls = [(by_question[q].id, k) for mode, q, k in calls if k != 2]
    assert {mode for mode, _, _ in calls} == {"lexical"}
    assert answer_calls == [("Q001", 2), ("Q002", 2), ("Q003", 2), ("Q004", 2)]
    assert metric_calls == [("Q001", 10), ("Q002", 10), ("Q004", 10)]  # Q003 has no refs
    records = {r["id"]: r for r in json.loads(_run_file(tmp_path).read_text())["items"]}
    assert records["Q001"]["retrieved"] == ["50/2", "50/1"]


def test_nothing_selected(dataset_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert cli_main(["answers", "--dataset", str(dataset_path), "--ids", "Q003"]) == 1
    assert "add --all for drafts" in capsys.readouterr().err


def test_report_and_compare(
    tmp_path: Path, dataset_path: Path, stub_backends: list[str], capsys: pytest.CaptureFixture
) -> None:
    assert cli_main(["answers", "--dataset", str(dataset_path), "--out", str(tmp_path)]) == 0
    latest = _run_file(tmp_path)
    capsys.readouterr()
    assert cli_main(["report", str(latest)]) == 0
    assert capsys.readouterr().out.startswith("# Evaluation report")
    output = tmp_path / "report.md"
    assert cli_main(["report", str(latest), "--output", str(output)]) == 0
    assert output.read_text().startswith("# Evaluation report")
    assert cli_main(["compare", str(latest), str(latest)]) == 0
    assert capsys.readouterr().out.count("| fake | fake |") == 2


def test_export_and_import(
    tmp_path: Path,
    dataset_path: Path,
    stub_backends: list[str],
    items: list[EvalItem],
    capsys: pytest.CaptureFixture,
) -> None:
    assert cli_main(["answers", "--dataset", str(dataset_path), "--out", str(tmp_path)]) == 0
    workbook_path = tmp_path / "review.xlsx"
    argv = ["export-xlsx", str(workbook_path), "--dataset", str(dataset_path), "--no-texts"]
    assert cli_main([*argv, "--with-answers", str(_run_file(tmp_path))]) == 0
    workbook = load_workbook(workbook_path)
    ws = workbook[SHEET_REVIEW]
    headers = {cell.value: cell.column for cell in ws[1]}
    assert "Testo vigente" not in headers
    assert ws.cell(row=2, column=headers["Riferimenti citati"]).value == "50/1"
    ws.cell(row=4, column=headers["Esito"], value="OK")
    workbook.save(workbook_path)

    import_argv = ["import-xlsx", str(workbook_path), "--dataset", str(dataset_path)]
    assert cli_main([*import_argv, "--dry-run"]) == 0
    assert "Q003: bozza -> verificata" in capsys.readouterr().out
    assert load_dataset(dataset_path) == items
    assert cli_main(import_argv) == 0
    assert "1 question(s) updated" in capsys.readouterr().out
    assert load_dataset(dataset_path)[2].status == "verificata"


def test_export_without_database(
    tmp_path: Path, dataset_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def open_db() -> Iterator[None]:
        raise psycopg.OperationalError("connection refused")
        yield

    monkeypatch.setattr(eval_cli, "_open_db", open_db)
    path = tmp_path / "review.xlsx"
    assert (
        cli_main(["export-xlsx", str(path), "--dataset", str(dataset_path), "--ids", "Q001"]) == 0
    )
    ws = load_workbook(path)[SHEET_REVIEW]
    assert ws.max_row == 2
    assert "Testo vigente" not in [cell.value for cell in ws[1]]


def test_usage_errors() -> None:
    with pytest.raises(SystemExit) as error:
        cli_main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        cli_main(["retrieval", "--k", "0,5"])


def test_answers_with_the_real_pipeline(
    tmp_path: Path,
    dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_db: None,
    stub_retriever: Callable,
) -> None:
    """End to end with answer_question and the FakeProvider; only retrieval is stubbed."""
    try:
        from appalti_qa.answer import answer_question  # noqa: F401
        from appalti_qa.llm import FakeProvider  # noqa: F401
    except ImportError:
        pytest.skip("answer/llm modules not available")
    monkeypatch.setattr(eval_cli, "_search_retriever", lambda mode, conn, settings: stub_retriever)
    argv = ["answers", "--dataset", str(dataset_path), "--all", "--out", str(tmp_path)]
    assert cli_main(argv) == 0
    results = json.loads(_run_file(tmp_path).read_text())
    assert results["config"]["provider"] == "fake"
    records = {item["id"]: item for item in results["items"]}
    assert "error" not in records["Q001"]
    assert records["Q001"]["cited"] == ["50/2", "50/1"]  # the fake quotes the top two sources
    assert records["Q003"]["abstained"] is True  # no sources retrieved
    assert results["answer"]["overall"]["abstention_accuracy"] == 1.0


def _fallback_reply(request: httpx2.Request) -> httpx2.Response:
    """Claude Opus 5.5 declined after some output; Claude Opus 5 served the reply.

    The top-level usage covers only the serving attempt; `usage.iterations` has both.
    """
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
                "trigger": {"type": "refusal"},
            },
            {
                "type": "text",
                "text": "Sotto i 140.000 euro.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Testo della disposizione.",
                        "document_index": 0,
                        "document_title": "titolo",
                        "start_char_index": 0,
                        "end_char_index": 25,
                    }
                ],
            },
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 80,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 200,
            "iterations": [
                {
                    "type": "message",
                    "model": "claude-opus-5-5",
                    "input_tokens": 1000,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-5",
                    "input_tokens": 1000,
                    "output_tokens": 80,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 200,
                },
            ],
        },
    }
    return httpx2.Response(200, json=body)


def test_answers_cost_every_fallback_attempt(
    tmp_path: Path,
    dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    no_db: None,
    stub_retriever: Callable,
) -> None:
    """End to end through the real answer backend: an answer served by the fallback model is
    priced for both attempts, each at its own model's rate (synthetic API responses)."""
    from appalti_qa.llm import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()

    def get_provider(settings: Settings, *, provider: str, model: str | None) -> Any:
        client = anthropic.Anthropic(
            api_key="sk-test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(_fallback_reply)
            ),
        )
        return AnthropicProvider(settings, model=model, client=client)

    monkeypatch.setattr("appalti_qa.llm.get_provider", get_provider)
    monkeypatch.setattr(eval_cli, "_search_retriever", lambda mode, conn, settings: stub_retriever)
    argv = ["answers", "--dataset", str(dataset_path), "--out", str(tmp_path), "--ids", "Q001"]
    assert cli_main([*argv, "--provider", "anthropic"]) == 0

    results = json.loads(_run_file(tmp_path).read_text())
    assert results["config"]["model"] == "claude-opus-5-5"
    (record,) = results["items"]
    assert record["cited"] == ["50/2"] and not record["abstained"]
    assert (record["input_tokens"], record["output_tokens"]) == (2400, 130)
    declined = (1200 * 4.00 + 50 * 20.00) / 1e6  # Claude Opus 5.5
    served = (1200 * 5.00 + 80 * 25.00) / 1e6  # Claude Opus 5
    assert record["cost_usd"] == pytest.approx(declined + served)
    assert results["answer"]["overall"]["cost_usd"] == pytest.approx(declined + served, abs=1e-6)
