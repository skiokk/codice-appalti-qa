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
from typing import TYPE_CHECKING, Any, NamedTuple, cast, get_args

import psycopg

from appalti_qa.config import ProviderName, RetrievalMode, Settings, get_settings
from appalti_qa.evaluation.dataset import (
    DEFAULT_DATASET,
    SPLITS,
    DatasetError,
    EvalItem,
    embedding_coverage,
    load_dataset,
    referenced_texts,
    select_items,
    split_items,
    validate_dataset,
)
from appalti_qa.evaluation.judge import get_judge
from appalti_qa.evaluation.metrics import DEFAULT_K_VALUES, AnswerFn, Retriever, UsageFn
from appalti_qa.evaluation.pricing import usage_cost
from appalti_qa.evaluation.report import (
    DEFAULT_RESULTS_DIR,
    LATEST,
    published_names,
    render_comparison,
    render_report,
    write_results,
)
from appalti_qa.evaluation.runner import (
    dataset_summary,
    run_answer_eval,
    run_retrieval_eval,
)
from appalti_qa.evaluation.sweep import Variant, default_grid, render_sweep, run_sweep
from appalti_qa.evaluation.xlsx import export_xlsx, review_import
from appalti_qa.models import Answer, Citation, Hit

if TYPE_CHECKING:
    from appalti_qa.answer.rerank import RerankingRetriever
    from appalti_qa.retrieval import Embedder

MODES: tuple[str, ...] = get_args(RetrievalMode)
PROVIDERS: tuple[str, ...] = get_args(ProviderName)
EMBEDDING_MODES: frozenset[str] = frozenset({"semantic", "hybrid"})
RERANK = "rerank"  # name of the reranked retrieval (decision 9) among the evaluated modes
DEFAULT_SWEEP_DIR = DEFAULT_RESULTS_DIR / "sweeps"
# The settings that shape retrieval, recorded in the configuration of every retrieval run.
RETRIEVAL_SETTINGS: tuple[str, ...] = (
    "semantic_granularity",
    "hubness_weight",
    "lexical_weight",
    "fusion_depth",
    "rrf_k",
)
# Share of the searchable versions that must carry an embedding from the configured model before
# results of the modes that use embeddings can be published.
MIN_EMBEDDING_COVERAGE = 0.99


class _CachedQueries:
    """An embedder that embeds each query once: a sweep searches the same questions many times."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, Any] = {}
        self.model_name = embedder.model_name

    @property
    def dim(self) -> int:
        return self._embedder.dim

    def embed_queries(self, texts: Sequence[str]) -> Any:
        import numpy as np

        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._embedder.embed_queries(missing), strict=True))
        return np.stack([self._cache[text] for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> Any:
        return self._embedder.embed_passages(texts)


class AnswerBackend(NamedTuple):
    answer: AnswerFn
    provider: str
    model: str
    usage: UsageFn | None = None  # per-model tokens of the calls since the previous call


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
    mode: RetrievalMode,
    conn: psycopg.Connection,
    settings: Settings,
    embedder: Embedder | None = None,
) -> Retriever:
    from appalti_qa.retrieval import search

    def retrieve(question: str, at_date: date, k: int) -> list[Hit]:
        try:
            return search(
                question, at_date, k, mode, conn=conn, settings=settings, embedder=embedder
            )
        finally:
            conn.rollback()  # end the read-only transaction (and any SET LOCAL) per query

    return retrieve


def _rerank_retriever(
    conn: psycopg.Connection, settings: Settings, embedder: Embedder | None = None
) -> RerankingRetriever:
    """The reranked retrieval of answers (decision 9) on the evaluation connection."""
    from appalti_qa.answer.rerank import RerankingRetriever
    from appalti_qa.llm.reranker import AnthropicReranker
    from appalti_qa.retrieval import candidate_pool

    def pool(question: str, at_date: date, size: int) -> list[Hit]:
        try:
            return candidate_pool(
                question, at_date, size, conn=conn, settings=settings, embedder=embedder
            )
        finally:
            conn.rollback()

    return RerankingRetriever(
        pool,
        AnthropicReranker(settings),
        pool_size=settings.rerank_pool,
        max_ranked=settings.rerank_max,
    )


def _rerank_summary(retriever: RerankingRetriever, settings: Settings) -> dict[str, Any]:
    """The reranker's configuration, calls, failures and the tokens and cost not yet taken."""
    usage = retriever.take()
    return {
        "model": settings.rerank_model,
        "pool": settings.rerank_pool,
        "max_ranked": settings.rerank_max,
        "calls": retriever.calls,
        "failures": retriever.failures,
        "input_tokens": sum(part.input_tokens for part in usage),
        "output_tokens": sum(part.output_tokens for part in usage),
        "cost_usd": usage_cost("anthropic", usage),
    }


def _with_rerank(settings: Settings, requested: bool) -> Settings | None:
    """Settings with reranking on when `requested`; None when it is requested without a key."""
    if not requested:
        return settings
    if settings.api_key("anthropic") is None:
        print("Reranking needs ANTHROPIC_API_KEY.", file=sys.stderr)
        return None
    return settings.model_copy(update={"rerank_provider": "anthropic"})


def _answer_backend(
    settings: Settings,
    provider_name: ProviderName,
    model: str | None,
    retrieve: Retriever,
    *,
    k: int,
    mode: RetrievalMode,
    delay_s: float,
    extra_usage: UsageFn | None = None,
) -> AnswerBackend:
    """The answer function of the run, with a meter of the tokens each model consumed.

    `extra_usage` adds the tokens of calls made outside the provider, such as reranking.
    """
    from appalti_qa.answer import answer_question
    from appalti_qa.llm import ModelUsage, UsageMeter, get_provider

    provider = UsageMeter(get_provider(settings, provider=provider_name, model=model))

    def answer(question: str, at_date: date) -> Answer:
        if delay_s:
            time.sleep(delay_s)
        return answer_question(
            question, at_date, settings, provider=provider, retriever=retrieve, k=k, mode=mode
        )

    def usage() -> list[ModelUsage]:
        return [*provider.take(), *(extra_usage() if extra_usage else ())]

    return AnswerBackend(answer, provider.name, provider.model, usage)


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
        split_items(items, getattr(args, "split", "all")),
        include_drafts=args.include_drafts,
        ids=args.ids,
        limit=args.limit,
    )
    if not selected:
        hint = (
            "" if args.include_drafts else "; only verified questions count, add --all for drafts"
        )
        print(f"No questions selected{hint}.", file=sys.stderr)
        return None
    summary = dataset_summary(args.dataset, items, selected, include_drafts=args.include_drafts)
    split = getattr(args, "split", "all")
    if split != "all":
        summary["split"] = split
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
            "cost_unknown",
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


def _write(results: dict[str, Any], out: Path, *, publish: bool) -> int:
    """Write the run (and with `publish`, its published copies) and print a summary."""
    published = published_names(results, out) if publish else []
    path = write_results(results, out, latest=publish)
    print(_overview(results))
    print(f"Results: {path}")
    for name in published:
        print(f"Published: {out / name}.json and .md")
    if publish and LATEST not in published:
        print(f"Kept {out / LATEST}.json: it holds an answer run.")
    return 0


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
    if args.publish and args.split != "all":
        print("Refusing to publish a split: published runs cover every question.", file=sys.stderr)
        return 2
    selection = _select(args)
    if selection is None:
        return 1
    items, summary = selection
    settings = _with_rerank(get_settings(), args.rerank)
    if settings is None:
        return 2
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
        "split": args.split,
        **{name: getattr(settings, name) for name in RETRIEVAL_SETTINGS},
    }
    evaluated = [*modes, RERANK] if args.rerank else modes
    with _open_db() as conn:
        refusal = _embedding_refusal(settings, evaluated, items, conn) if args.publish else None
        if refusal:
            print(refusal, file=sys.stderr)
            return 2
        retrievers: dict[str, Retriever] = {
            mode: _search_retriever(cast(RetrievalMode, mode), conn, settings) for mode in modes
        }
        reranked = _rerank_retriever(conn, settings) if args.rerank else None
        if reranked is not None:
            retrievers[RERANK] = reranked
        results = run_retrieval_eval(
            items, retrievers, primary=primary, dataset=summary, config=config, k_values=args.k
        )
    if reranked is not None:
        results["rerank"] = _rerank_summary(reranked, settings)
    return _write(results, args.out, publish=args.publish)


def _cmd_sweep(args: argparse.Namespace) -> int:
    selection = _select(args)
    if selection is None:
        return 1
    items, summary = selection
    settings = get_settings()
    grid = default_grid()
    variants = [v for v in grid if v.name in set(args.only)] if args.only else grid
    unknown = sorted(set(args.only or ()) - {v.name for v in grid})
    if unknown:
        print(f"Unknown variants: {unknown}", file=sys.stderr)
        return 2
    chosen = None
    if args.chosen_from:
        chosen = json.loads(args.chosen_from.read_text(encoding="utf-8")).get("chosen")
    from appalti_qa.retrieval import get_embedder

    embedder = _CachedQueries(get_embedder(settings))
    config = {"embedding_model": settings.embedding_model, "k_values": list(args.k)}
    with _open_db() as conn:

        def make_retriever(variant: Variant) -> Retriever:
            tuned = settings.model_copy(update=dict(variant.params))
            return _search_retriever(cast(RetrievalMode, variant.mode), conn, tuned, embedder)

        results = run_sweep(
            items,
            variants,
            make_retriever,
            split=args.split,
            dataset=summary,
            config=config,
            k_values=args.k,
        )
    if chosen:
        results["chosen_on_dev"] = chosen
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{results['run_id']}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = render_sweep(results, chosen=chosen)
    path.with_suffix(".md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Results: {path}")
    return 0


def _publish_refusal(provider: str, judge_provider: str) -> str | None:
    """Why an answer run must not replace the published results, or None if it may."""
    fake = [
        role for role, name in (("answers", provider), ("judge", judge_provider)) if name == "fake"
    ]
    if not fake:
        return None
    return (
        f"Refusing to publish: the {' and the '.join(fake)} would use the fake provider, so the"
        " numbers are placeholders. Set the API keys (see LLM_PROVIDER, JUDGE_PROVIDER) or run"
        " without --publish."
    )


def _embedding_refusal(
    settings: Settings,
    modes: Sequence[str],
    items: Sequence[EvalItem],
    conn: psycopg.Connection,
) -> str | None:
    """Why results of `modes` must not be published for lack of embeddings, or None if they may.

    Semantic search only returns versions embedded by the configured model, so with the fake
    embedder or a partly embedded corpus it returns little or nothing and hybrid search is in
    effect lexical: the published numbers would not describe the system.
    """
    if not EMBEDDING_MODES.union({RERANK}).intersection(modes):
        return None
    from appalti_qa.retrieval.embedding import FAKE_MODEL_NAME

    model = settings.embedding_model
    if model == FAKE_MODEL_NAME:
        return (
            "Refusing to publish: EMBEDDING_MODEL is 'fake', so semantic search does not use the"
            " embeddings of the corpus and hybrid search is in effect lexical. Set EMBEDDING_MODEL"
            " to the model the corpus was embedded with, or run without --publish."
        )
    granularities = set()
    if EMBEDDING_MODES.intersection(modes):
        granularities.add(settings.semantic_granularity == "passage")
    if RERANK in modes:  # the candidate pool searches whole units and passages
        granularities.update({False, True})
    days = {item.vigenza for item in items}
    for passages in sorted(granularities):
        embedded, total = embedding_coverage(conn, model, days, passages=passages)
        if not total or embedded / total < MIN_EMBEDDING_COVERAGE:
            return _coverage_refusal(model, embedded, total, passages=passages)
    return None


def _coverage_refusal(model: str, embedded: int, total: int, *, passages: bool) -> str:
    share = f" ({embedded / total:.1%})" if total else ""
    return (
        f"Refusing to publish: {embedded:,} of the {total:,} versions in force on the evaluated"
        f" dates{share} have {'passage embeddings' if passages else 'an embedding'} from {model};"
        f" at least {MIN_EMBEDDING_COVERAGE:.0%}"
        " are needed, otherwise semantic search misses the rest and hybrid search degrades to"
        " lexical. Run `appalti-qa embed` (or set EMBEDDING_MODEL to the model the corpus was"
        " embedded with), or run without --publish."
    )


def _cmd_answers(args: argparse.Namespace) -> int:
    settings = get_settings()
    requested = args.provider or settings.llm_provider
    provider_name = settings.effective_provider(requested)
    judge_settings = settings
    if args.judge_provider:
        judge_settings = settings.model_copy(update={"judge_provider": args.judge_provider})
    if args.publish:
        judge_provider = judge_settings.effective_provider(judge_settings.judge_provider)
        refusal = _publish_refusal(provider_name, judge_provider)
        if refusal:
            print(refusal, file=sys.stderr)
            return 2
    selection = _select(args)
    if selection is None:
        return 1
    items, summary = selection
    if provider_name != requested:
        print(f"No API key for {requested}: answering with the fake provider.", file=sys.stderr)
    reranking = _with_rerank(settings, args.rerank)
    if reranking is None:
        return 2
    settings = reranking
    judge = get_judge(judge_settings)
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.default_top_k
    with _open_db() as conn:
        refusal = _embedding_refusal(settings, [mode], items, conn) if args.publish else None
        if refusal:
            print(refusal, file=sys.stderr)
            return 2
        # Answers retrieve exactly as the app does (same k and mode, reranked when the app
        # reranks); the retrieval metrics make their own deeper calls, since the fused ranking
        # depends on the requested depth.
        reranked = (
            _rerank_retriever(conn, settings) if mode == "hybrid" and settings.reranks() else None
        )
        retrieve = reranked or _search_retriever(mode, conn, settings)
        backend = _answer_backend(
            settings,
            provider_name,
            args.model,
            retrieve,
            k=k,
            mode=mode,
            delay_s=args.delay,
            extra_usage=reranked.take if reranked else None,
        )
        provider, model = backend.provider, backend.model
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
        if reranked is not None:
            config["rerank_model"] = settings.rerank_model
        print(
            f"Evaluating {len(items)} questions with {provider}/{model}, judge {judge.name},"
            f" retrieval {mode} k={k}.",
            file=sys.stderr,
        )
        results = run_answer_eval(
            items,
            backend.answer,
            judge,
            retrieve=retrieve,
            dataset=summary,
            config=config,
            usage=backend.usage,
        )
    if reranked is not None:
        # Answer costs include their reranking; this counts the calls and failures of the whole
        # run (the retrieval metrics rerank again) and the tokens of the retrieval metrics.
        results["rerank"] = _rerank_summary(reranked, settings)
    return _write(results, args.out, publish=args.publish)


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


def _add_split(parser: argparse.ArgumentParser, default: str = "all") -> None:
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=default,
        help="dev = questions with refs and an odd id, test = even id (default: %(default)s)",
    )


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
        "retrieval",
        help="retrieval metrics per mode, no LLM (latest_retrieval.json only with --publish)",
    )
    _add_selection(p)
    p.add_argument("--mode", action="append", choices=MODES, help="repeatable; default: all")
    p.add_argument(
        "--rerank",
        action="store_true",
        help=f"also evaluate the reranked retrieval of answers as mode {RERANK!r} (decision 9)",
    )
    p.add_argument("--k", type=_k_list, default=DEFAULT_K_VALUES, help="cutoffs, e.g. 1,3,5,10")
    _add_split(p)
    p.add_argument(
        "--publish",
        action="store_true",
        help="also write latest_retrieval.json, and latest.json (the web page) unless it holds an"
        " answer run; refused when semantic search lacks embeddings",
    )
    p.set_defaults(handler=_cmd_retrieval)

    p = sub.add_parser(
        "sweep",
        help="compare the retrieval variants on a split (dev: also apply the selection rule)",
    )
    _add_selection(p)
    p.set_defaults(out=DEFAULT_SWEEP_DIR)
    p.add_argument("--k", type=_k_list, default=DEFAULT_K_VALUES, help="cutoffs, e.g. 1,3,5,10")
    _add_split(p, default="dev")
    p.add_argument(
        "--only", action="append", metavar="VARIANT", help="repeatable; default: the whole grid"
    )
    p.add_argument(
        "--chosen-from",
        type=Path,
        metavar="SWEEP.json",
        help="a dev sweep whose chosen variant is compared with the baseline",
    )
    p.set_defaults(handler=_cmd_sweep)

    p = sub.add_parser(
        "answers",
        aliases=["run"],
        help="answer and score the questions (latest.json only with --publish)",
    )
    _add_selection(p)
    p.add_argument("--provider", choices=PROVIDERS, help="default: LLM_PROVIDER")
    p.add_argument("--model", help="default: the provider's configured model")
    p.add_argument("--mode", choices=MODES, help="retrieval mode (default: RETRIEVAL_MODE)")
    p.add_argument("--k", type=int, help="sources given to the model (default: DEFAULT_TOP_K)")
    p.add_argument(
        "--rerank",
        action="store_true",
        help="rerank the sources of hybrid answers (default: RERANK_PROVIDER)",
    )
    p.add_argument("--judge-provider", choices=PROVIDERS, help="default: JUDGE_PROVIDER")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between questions")
    p.add_argument(
        "--publish",
        action="store_true",
        help="also write latest.json (the web page); refused when the provider or judge is fake"
        " or semantic search lacks embeddings",
    )
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
