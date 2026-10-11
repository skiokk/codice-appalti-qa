"""Command-line entry point: `appalti-qa <command>`.

Each handler imports its module lazily so a missing or broken module only affects its own command.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable, Sequence

from appalti_qa.config import get_settings

Handler = Callable[[argparse.Namespace], int]


def _resolve(module: str, function: str) -> Callable[..., int] | None:
    try:
        return getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as exc:
        print(f"{module}.{function} is not available: {exc}", file=sys.stderr)
        return None


def _delegate(module: str, function: str) -> Handler:
    def run(args: argparse.Namespace) -> int:
        target = _resolve(module, function)
        return 2 if target is None else target(args)

    return run


def _init_db(args: argparse.Namespace) -> int:
    from appalti_qa.db import init_schema

    init_schema()
    print("Schema applied.")
    return 0


def _eval(args: argparse.Namespace) -> int:
    target = _resolve("appalti_qa.evaluation", "cli_main")
    return 2 if target is None else target(args.eval_args)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="appalti-qa", description="Question answering over D.Lgs. 36/2023."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="apply sql/schema.sql to DATABASE_URL")
    p.set_defaults(handler=_init_db)

    p = sub.add_parser("download", help="download Normattiva Akoma Ntoso snapshots")
    p.add_argument("--raw-dir", default=str(settings.raw_dir))
    p.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    p.add_argument("--force", action="store_true", help="re-download cached files")
    p.set_defaults(handler=_delegate("appalti_qa.ingest", "cli_download"))

    p = sub.add_parser("ingest", help="parse snapshots and load versioned units into the DB")
    p.add_argument("--raw-dir", default=str(settings.raw_dir))
    p.set_defaults(handler=_delegate("appalti_qa.ingest", "cli_ingest"))

    p = sub.add_parser("embed", help="embed unit versions that have no embedding yet")
    p.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    p.add_argument("--limit", type=int, default=None, help="embed at most N versions")
    p.set_defaults(handler=_delegate("appalti_qa.retrieval", "cli_embed"))

    p = sub.add_parser("serve", help="run the web application")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(handler=_delegate("appalti_qa.web", "cli_serve"))

    p = sub.add_parser(
        "eval",
        help="evaluation (validate, run, report, export, import); see `eval -h`",
        add_help=False,
    )
    p.add_argument("eval_args", nargs=argparse.REMAINDER, help="arguments for the eval tool")
    p.set_defaults(handler=_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
