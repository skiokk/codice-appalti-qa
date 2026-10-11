"""Results files and markdown reports."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path("eval/results")

Formatter = Callable[[Any], str]
Column = tuple[str, str, Formatter]  # header, metric name, formatter


def write_results(
    results: Mapping[str, Any], out_dir: Path = DEFAULT_RESULTS_DIR, *, latest: bool = True
) -> Path:
    """Write `<run_id>.json` and `<run_id>.md`; with `latest`, also `latest.json` / `latest.md`.

    Returns the path of the JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = results["run_id"]
    document = json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n"
    report = render_report(results)
    path = out_dir / f"{run_id}.json"
    path.write_text(document, encoding="utf-8")
    (out_dir / f"{run_id}.md").write_text(report, encoding="utf-8")
    if latest:
        (out_dir / "latest.json").write_text(document, encoding="utf-8")
        (out_dir / "latest.md").write_text(report, encoding="utf-8")
    return path


# --- formatting -------------------------------------------------------------------------------


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _dec(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _int(value: Any) -> str:
    return "n/a" if value is None else f"{value:,}"


def _usd(value: Any) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _cost(total: Any, unknown: Any) -> str:
    """A cost total, noting how many calls it leaves out for lack of a price."""
    if not unknown:
        return _usd(total)
    return f"{_usd(total)} ({unknown} without a known price, not included)"


def _cell(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def _group_table(
    label: str, groups: Mapping[str, Mapping[str, Any]], columns: Sequence[Column]
) -> str:
    headers = [label, "n", *(header for header, _, _ in columns)]
    rows = [
        [name, group["n"], *(fmt(group.get(metric)) for _, metric, fmt in columns)]
        for name, group in groups.items()
    ]
    return _table(headers, rows)


def _k_values(overall: Mapping[str, Any]) -> list[int]:
    return sorted(int(m[1]) for key in overall if (m := re.fullmatch(r"recall@(\d+)", key)))


# --- report sections --------------------------------------------------------------------------


def _intro(results: Mapping[str, Any]) -> list[str]:
    dataset = results.get("dataset", {})
    config = results.get("config", {})
    lines = [
        f"# Evaluation report `{results['run_id']}`",
        "",
        f"Created {results.get('created_at', 'n/a')}. Dataset `{dataset.get('path', 'n/a')}`:"
        f" {dataset.get('evaluated', 'n/a')} questions evaluated"
        f" (selection: {dataset.get('selection', 'n/a')});"
        f" {dataset.get('verified', 'n/a')} of {dataset.get('size', 'n/a')} questions in the file"
        " are verified.",
        "",
    ]
    drafts = dataset.get("evaluated", 0) - dataset.get("evaluated_verified", 0)
    if drafts > 0:
        lines += [
            f"> **Draft questions included.** {drafts} of the evaluated questions are not verified"
            " yet (status `bozza`): the numbers are indicative and not for publication.",
            "",
        ]
    if config.get("provider") == "fake":
        lines += [
            "> Answers come from the deterministic fake provider: this run tests the pipeline,"
            " not answer quality.",
            "",
        ]
    if config.get("judge") == "overlap":
        lines += [
            "> Correctness is a lexical-overlap proxy (no judge model configured), not a legal"
            " assessment.",
            "",
        ]
    settings = [(key, value) for key, value in config.items() if value not in (None, [], "")]
    lines += ["## Configuration", "", _table(["Setting", "Value"], settings), ""]
    return lines


def _retrieval_section(results: Mapping[str, Any]) -> list[str]:
    retrieval = results.get("retrieval")
    if not retrieval:
        return []
    ks = _k_values(retrieval["overall"])
    shown = [k for k in (5, 10) if k in ks] or ks[-2:]
    group_columns: list[Column] = [
        *((f"Recall@{k}", f"recall@{k}", _pct) for k in shown),
        ("MRR", "mrr", _dec),
        *((f"Article recall@{k}", f"article_recall@{k}", _pct) for k in shown),
        ("Article MRR", "article_mrr", _dec),
    ]
    mode_columns: list[Column] = [
        *((f"Recall@{k}", f"recall@{k}", _pct) for k in ks),
        *group_columns[len(shown) :],
    ]
    primary = results.get("config", {}).get("retrieval_mode", "n/a")
    by_mode = results.get("retrieval_by_mode") or {primary: retrieval}
    modes = {mode: metrics["overall"] for mode, metrics in by_mode.items()}
    return [
        "## Retrieval",
        "",
        "Questions with expected references only, each at its own reference date. Comma level:"
        " an expected ref is found when a retrieved unit matches it (`50/1/b` is found by"
        " `50/1`). Article level: a retrieved unit of the same article is enough.",
        "",
        _group_table("Mode", modes, mode_columns),
        "",
        f"### Retrieval by kind ({primary})",
        "",
        _group_table("Kind", retrieval["by_kind"], group_columns),
        "",
        f"### Retrieval by category ({primary})",
        "",
        _group_table("Category", retrieval["by_category"], group_columns),
        "",
    ]


_ANSWER_COLUMNS: list[Column] = [
    ("Correctness", "correctness", _pct),
    ("Citation precision", "citation_precision", _pct),
    ("Citation recall", "citation_recall", _pct),
    ("Abstention accuracy", "abstention_accuracy", _pct),
    ("Latency p50 (ms)", "latency_p50_ms", _int),
]


def _needs_review(item: Mapping[str, Any]) -> bool:
    return (
        "error" in item
        or "judge_error" in item
        or (item.get("score") is not None and item["score"] < 1)
        or item.get("abstention_correct") is False
    )


def _answer_section(results: Mapping[str, Any]) -> list[str]:
    answer = results.get("answer")
    if not answer:
        return []
    o = answer["overall"]
    overview = [
        ("Questions", o["n"]),
        ("Errors (answer / judge)", f"{o.get('errors', 0)} / {o.get('judge_errors', 0)}"),
        ("Correctness", _pct(o.get("correctness"))),
        ("Citation precision", _pct(o.get("citation_precision"))),
        ("Citation recall", _pct(o.get("citation_recall"))),
        ("Abstention accuracy", _pct(o.get("abstention_accuracy"))),
        ("Abstention rate", _pct(o.get("abstention_rate"))),
        (
            "Latency p50 / p95 (ms)",
            f"{_int(o.get('latency_p50_ms'))} / {_int(o.get('latency_p95_ms'))}",
        ),
        ("Tokens in / out", f"{_int(o.get('input_tokens'))} / {_int(o.get('output_tokens'))}"),
        ("Cost of answers", _cost(o.get("cost_usd"), o.get("cost_unknown"))),
        ("Cost of judging", _cost(o.get("judge_cost_usd"), o.get("judge_cost_unknown"))),
    ]
    review = sorted(
        (item for item in results.get("items", []) if _needs_review(item)),
        key=lambda item: (item.get("score") if item.get("score") is not None else -1, item["id"]),
    )
    review_rows = [
        [
            item["id"],
            item["kind"],
            _dec(item.get("score")),
            ", ".join(item.get("expected_refs", [])) or "-",
            ", ".join(item.get("cited", [])) or ("abstained" if item.get("abstained") else "-"),
            item.get("error") or item.get("judge_error") or item.get("rationale", ""),
        ]
        for item in review
    ]
    lines = [
        "## Answers",
        "",
        "Correctness is the judge score (correct 1, partial 0.5, wrong 0). Citation precision and"
        " recall compare cited units with the expected refs (questions with refs only); abstention"
        " accuracy checks that exactly the unanswerable questions are declined. Costs use list"
        " prices without caching discounts; Groq models count as free tier.",
        "",
        _table(["Metric", "Value"], overview),
        "",
        "### Answers by kind",
        "",
        _group_table("Kind", answer["by_kind"], _ANSWER_COLUMNS),
        "",
        "### Answers by category",
        "",
        _group_table("Category", answer["by_category"], _ANSWER_COLUMNS),
        "",
    ]
    if review_rows:
        lines += [
            "### Questions to look at",
            "",
            _table(["ID", "Kind", "Score", "Expected", "Cited", "Note"], review_rows),
            "",
        ]
    return lines


def render_report(results: Mapping[str, Any]) -> str:
    """Markdown report of one results document."""
    lines = _intro(results) + _retrieval_section(results) + _answer_section(results)
    return "\n".join(lines).rstrip() + "\n"


def render_comparison(runs: Sequence[Mapping[str, Any]]) -> str:
    """Markdown table comparing the headline numbers of several runs."""
    rows = []
    for run in runs:
        config = run.get("config", {})
        retrieval = (run.get("retrieval") or {}).get("overall", {})
        answer = (run.get("answer") or {}).get("overall", {})
        rows.append(
            [
                run["run_id"],
                config.get("provider") or "-",
                config.get("model") or "-",
                config.get("retrieval_mode") or "-",
                run.get("dataset", {}).get("evaluated", "n/a"),
                _pct(retrieval.get("recall@5")),
                _pct(retrieval.get("recall@10")),
                _dec(retrieval.get("mrr")),
                _pct(answer.get("correctness")),
                _pct(answer.get("citation_precision")),
                _pct(answer.get("citation_recall")),
                _pct(answer.get("abstention_accuracy")),
                _int(answer.get("latency_p50_ms")),
                _int(answer.get("latency_p95_ms")),
                _usd(answer.get("cost_usd")),
            ]
        )
    headers = [
        "Run",
        "Provider",
        "Model",
        "Mode",
        "n",
        "Recall@5",
        "Recall@10",
        "MRR",
        "Correctness",
        "Citation precision",
        "Citation recall",
        "Abstention accuracy",
        "p50 (ms)",
        "p95 (ms)",
        "Cost",
    ]
    return _table(headers, rows) + "\n"
