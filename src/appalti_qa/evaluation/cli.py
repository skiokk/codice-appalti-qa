"""`appalti-qa eval <command>`: validate the dataset, run evaluations, report, review via xlsx."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from datetime import date
from pathlib import Path
from typing import Any, get_args

import psycopg

from appalti_qa.config import ProviderName, RetrievalMode, Settings, get_settings
from appalti_qa.evaluation.dataset import (
    DEFAULT_DATASET,
    DatasetError,
    EvalItem,
    load_dataset,
    referenced_texts,
    select_items,
    validate_dataset,
)
from appalti_qa.evaluation.judge import get_judge
from appalti_qa.evaluation.metrics import DEFAULT_K_VALUES, AnswerFn, Retriever
from appalti_qa.evaluation.report import (
    DEFAULT_RESULTS_DIR,
    render_comparison,
    render_report,
    write_results,
)
from appalti_qa.evaluation.runner import (
    CachedRetriever,
    dataset_summary,
    run_answer_eval,
    run_retrieval_eval,
)
from appalti_qa.evaluation.xlsx import export_xlsx, review_import
from appalti_qa.models import Answer, Citation, Hit

MODES: tuple[str, ...] = get_args(RetrievalMode)
PROVIDERS: tuple[str, ...] = get_args(ProviderName)


# --- backends (patched in tests) --------------------------------------------------------------


@contextmanager
def _open_db() -> Iterator[psycopg.Connection]:
    from appalti_qa.db import connect

    conn = connect(connect_timeout=5)
    try:
        yield conn
    finally:
        conn.close()


def _search_retriever(
    mode: RetrievalMode, conn: psycopg.Connection, settings: Settings
) -> Retriever:
    from appalti_qa.retrieval import search

    def retrieve(question: str, at_date: date, k: int) -> list[Hit]:
        try:
            return search(question, at_date, k, mode, conn=conn, settings=settings)
        finally:
            conn.rollback()  # end the read-only transaction (and any SET LOCAL) per query

    return retrieve


def _answer_backend(
    settings: Settings,
    provider_name: ProviderName,
    model: str | None,
    retrieve: Retriever,
    *,
    k: int,
    mode: RetrievalMode,
    delay_s: float,
) -> tuple[AnswerFn, str, str]:
    """(answer function, provider name, model) for the run."""
    from appalti_qa.answer import answer_question
    from appalti_qa.llm import get_provider

    provider = get_provider(settings, provider=provider_name, model=model)

    def answer(question: str, at_date: date) -> Answer:
        if delay_s:
            time.sleep(delay_s)
        return answer_question(
            question, at_date, settings, provider=provider, retriever=retrieve, k=k, mode=mode
        )

    return answer, provider.name, provider.model


# --- helpers ----------------------------------------------------------------------------------


def _print_problems(problems: Sequence[str]) -> None:
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    print(f"{len(problems)} problem(s) found.", file=sys.stderr)


def _csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _k_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(part) for part in _csv_list(text)}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {text!r}") from None
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("cutoffs must be positive integers")
    return values


def _select(args: argparse.Namespace) -> tuple[list[EvalItem], dict[str, Any]] | None:
    items = load_dataset(args.dataset)
    selected = select_items(
        items, include_drafts=args.include_drafts, ids=args.ids, limit=args.limit
    )
    if not selected:
        hint = (
            "" if args.include_drafts else "; only verified questions count, add --all for drafts"
        )
        print(f"No questions selected{hint}.", file=sys.stderr)
        return None
    summary = dataset_summary(args.dataset, items, selected, include_drafts=args.include_drafts)
    return selected, summary


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _overview(results: dict[str, Any]) -> str:
    lines = []
    by_mode = results.get("retrieval_by_mode")
    if by_mode is None and results.get("retrieval"):
        by_mode = {results["config"].get("retrieval_mode"): results["retrieval"]}
    for mode, metrics in (by_mode or {}).items():
        overall = metrics["overall"]
        shown = [k for k in overall if k.startswith(("recall@", "mrr", "article_recall@"))]
        shown.append("article_mrr")
        values = " ".join(f"{name}={_fmt(overall.get(name))}" for name in shown)
        lines.append(f"retrieval [{mode}] n={overall['n']} {values}")
    if results.get("answer"):
        o = results["answer"]["overall"]
        names = (
            "correctness",
            "citation_precision",
            "citation_recall",
            "abstention_accuracy",
            "latency_p50_ms",
            "latency_p95_ms",
            "cost_usd",
            "errors",
        )
        lines.append(f"answers n={o['n']} " + " ".join(f"{n}={_fmt(o.get(n))}" for n in names))
    return "\n".join(lines)


def _answers_from_results(results: dict[str, Any], items: Sequence[EvalItem]) -> dict[str, Answer]:
    """Rebuild the answers of a results file (text and citations) for the review sheet."""
    by_id = {item.id: item for item in items}
    config = results.get("config", {})
    answers: dict[str, Answer] = {}
    for record in results.get("items", []):
        item = by_id.get(record.get("id", ""))
        if item is None or "answer" not in record:
            continue
        answers[item.id] = Answer(
            question=item.question,
            at_date=item.vigenza,
            text=record["answer"],
            citations=[Citation(**citation) for citation in record.get("citations", [])],
            abstained=record.get("abstained", False),
            model=config.get("model") or "",
            provider=config.get("provider") or "",
        )
    return answers


# --- commands ---------------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    items = load_dataset(args.dataset)
    with _open_db() if args.db else nullcontext() as conn:
        problems = validate_dataset(items, conn=conn)
    if problems:
        _print_problems(problems)
        return 1
    verified = sum(item.verified for item in items)
    scope = "static and corpus checks" if args.db else "static checks"
    print(f"{args.dataset}: {len(items)} questions, {verified} verified; {scope} passed.")
    return 0


def _cmd_retrieval(args: argparse.Namespace) -> int:
    selection = _select(args)
    if selection is None:
        return 1
    items, summary = selection
    settings = get_settings()
    modes = list(dict.fromkeys(args.mode or MODES))
    primary = settings.retrieval_mode if settings.retrieval_mode in modes else modes[0]
    config = {
        "provider": None,
        "model": None,
        "retrieval_mode": primary,
        "modes": modes,
        "k_values": list(args.k),
        "embedding_model": settings.embedding_model,
        "judge": None,
    }
    with _open_db() as conn:
        retrievers = {mode: _search_retriever(mode, conn, settings) for mode in modes}
        results = run_retrieval_eval(
            items, retrievers, primary=primary, dataset=summary, config=config, k_values=args.k
        )
    path = write_results(results, args.out, latest=args.publish)
    print(_overview(results))
    print(f"Results: {path}")
    return 0


def _cmd_answers(args: argparse.Namespace) -> int:
    selection = _select(args)
    if selection is None:
        return 1
    items, summary = selection
    settings = get_settings()
    requested = args.provider or settings.llm_provider
    provider_name = settings.effective_provider(requested)
    if provider_name != requested:
        print(f"No API key for {requested}: answering with the fake provider.", file=sys.stderr)
    judge_settings = settings
    if args.judge_provider:
        judge_settings = settings.model_copy(update={"judge_provider": args.judge_provider})
    judge = get_judge(judge_settings)
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.default_top_k
    with _open_db() as conn:
        retrieve = CachedRetriever(
            _search_retriever(mode, conn, settings), depth=max(k, *DEFAULT_K_VALUES)
        )
        answer, provider, model = _answer_backend(
            settings, provider_name, args.model, retrieve, k=k, mode=mode, delay_s=args.delay
        )
        config: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "retrieval_mode": mode,
            "k": k,
            "embedding_model": settings.embedding_model,
            "judge": judge.name,
            "judge_model": getattr(judge, "model", None),
        }
        if provider == "anthropic":
            config["effort"] = settings.llm_effort
        print(
            f"Evaluating {len(items)} questions with {provider}/{model}, judge {judge.name},"
            f" retrieval {mode} k={k}.",
            file=sys.stderr,
        )
        results = run_answer_eval(
            items, answer, judge, retrieve=retrieve, dataset=summary, config=config
        )
    path = write_results(results, args.out, latest=not args.no_latest)
    print(_overview(results))
    print(f"Results: {path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = render_report(json.loads(args.results.read_text(encoding="utf-8")))
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report: {args.output}")
    else:
        print(report, end="")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    print(render_comparison(runs), end="")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    items = load_dataset(args.dataset)
    if args.ids:
        items = [item for item in items if item.id in set(args.ids)]
    answers = None
    if args.with_answers:
        results = json.loads(args.with_answers.read_text(encoding="utf-8"))
        answers = _answers_from_results(results, items)
    texts = None
    if not args.no_texts:
        try:
            with _open_db() as conn:
                texts = referenced_texts(items, conn)
        except psycopg.OperationalError as exc:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"Database not reachable ({reason}): no 'Testo vigente' column.", file=sys.stderr)
    export_xlsx(items, args.output, answers=answers, unit_texts=texts)
    print(f"Wrote {len(items)} questions to {args.output}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    with _open_db() if args.db else nullcontext() as conn:
        result = review_import(args.input, args.dataset, conn=conn, write=not args.dry_run)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for change in result.changes:
        print(change)
    action = "would be updated (dry run)" if args.dry_run else "updated"
    print(f"{len(result.changed)} question(s) {action} in {args.dataset}.")
    return 0


# --- parser -----------------------------------------------------------------------------------


def _add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="dataset YAML")


def _add_selection(parser: argparse.ArgumentParser) -> None:
    _add_dataset(parser)
    parser.add_argument(
        "--all",
        dest="include_drafts",
        action="store_true",
        help="include draft questions (default: only verificata/corretta)",
    )
    parser.add_argument("--ids", type=_csv_list, help="comma-separated ids, e.g. Q001,Q002")
    parser.add_argument("--limit", type=int, help="evaluate at most N questions")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR, help="results directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="appalti-qa eval", description="Evaluation of retrieval and answers."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check the dataset file")
    _add_dataset(p)
    p.add_argument("--db", action="store_true", help="also check refs and excerpts in the corpus")
    p.set_defaults(handler=_cmd_validate)

    p = sub.add_parser(
        "retrieval", help="retrieval metrics per mode, no LLM (latest.json only with --publish)"
    )
    _add_selection(p)
    p.add_argument("--mode", action="append", choices=MODES, help="repeatable; default: all")
    p.add_argument("--k", type=_k_list, default=DEFAULT_K_VALUES, help="cutoffs, e.g. 1,3,5,10")
    p.add_argument(
        "--publish",
        action="store_true",
        help="also write latest.json (the web page), e.g. before any answer run exists",
    )
    p.set_defaults(handler=_cmd_retrieval)

    p = sub.add_parser("answers", aliases=["run"], help="answer and score the questions")
    _add_selection(p)
    p.add_argument("--provider", choices=PROVIDERS, help="default: LLM_PROVIDER")
    p.add_argument("--model", help="default: the provider's configured model")
    p.add_argument("--mode", choices=MODES, help="retrieval mode (default: RETRIEVAL_MODE)")
    p.add_argument("--k", type=int, help="sources given to the model (default: DEFAULT_TOP_K)")
    p.add_argument("--judge-provider", choices=PROVIDERS, help="default: JUDGE_PROVIDER")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    p.add_argument("--no-latest", action="store_true", help="do not update latest.json")
    p.set_defaults(handler=_cmd_answers)

    p = sub.add_parser("report", help="markdown report of a results file")
    p.add_argument("results", type=Path, metavar="RESULTS.json")
    p.add_argument("--output", type=Path, help="write to a file instead of stdout")
    p.set_defaults(handler=_cmd_report)

    p = sub.add_parser("compare", help="compare the headline numbers of several runs")
    p.add_argument("results", type=Path, nargs="+", metavar="RESULTS.json")
    p.set_defaults(handler=_cmd_compare)

    p = sub.add_parser("export-xlsx", help="workbook for the human review")
    p.add_argument("output", type=Path, metavar="OUT.xlsx")
    _add_dataset(p)
    p.add_argument("--ids", type=_csv_list, help="comma-separated ids")
    p.add_argument("--with-answers", type=Path, metavar="RESULTS.json", help="add system answers")
    p.add_argument("--no-texts", action="store_true", help="skip the text in force (no DB)")
    p.set_defaults(handler=_cmd_export)

    p = sub.add_parser("import-xlsx", help="merge the review back into the dataset")
    p.add_argument("input", type=Path, metavar="IN.xlsx")
    _add_dataset(p)
    p.add_argument("--db", action="store_true", help="also check changed questions in the corpus")
    p.add_argument("--dry-run", action="store_true", help="show the changes without writing")
    p.set_defaults(handler=_cmd_import)
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of `appalti-qa eval`; returns the exit code (argparse errors exit with 2)."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except DatasetError as exc:
        _print_problems(exc.problems)
        return 1
    except psycopg.OperationalError as exc:
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"Database not reachable: {reason}", file=sys.stderr)
        return 1
