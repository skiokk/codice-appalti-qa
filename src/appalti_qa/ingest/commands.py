"""Command-line handlers: ``appalti-qa download [--refine]`` and ``appalti-qa ingest``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from appalti_qa.db import connect, init_schema
from appalti_qa.ingest.akn import parse_snapshot
from appalti_qa.ingest.download import (
    DownloadError,
    NormattivaFetcher,
    download_snapshots,
    snapshot_files,
)
from appalti_qa.ingest.loader import load_corpus
from appalti_qa.ingest.refine import DEFAULT_MAX_PROBES, refine_snapshots
from appalti_qa.ingest.versioning import build_versions

__all__ = ["cli_download", "cli_ingest", "snapshot_files"]

logger = logging.getLogger(__name__)


def cli_download(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    try:
        paths = download_snapshots(raw_dir, delay_s=args.delay, force=args.force)
        if getattr(args, "refine", False):
            max_probes = getattr(args, "max_probes", None) or DEFAULT_MAX_PROBES
            with NormattivaFetcher(delay_s=args.delay) as fetch:
                report = refine_snapshots(raw_dir, fetch, max_probes=max_probes)
            paths = snapshot_files(raw_dir)
            print(
                f"Refined: {report.probes} probes, {len(report.deleted)} redundant files "
                f"deleted, boundaries {', '.join(f'{day:%Y-%m-%d}' for day in report.boundaries)}"
            )
            for lo, hi in report.unresolved:
                print(f"Not refined (probe limit): change between {lo} and {hi}", file=sys.stderr)
    except (DownloadError, httpx.HTTPError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Cannot parse a downloaded snapshot: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Cannot write snapshots to {raw_dir}: {exc}", file=sys.stderr)
        return 1
    print(f"{len(paths)} snapshots available in {raw_dir}")
    return 0


def cli_ingest(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    files = snapshot_files(raw_dir)
    if not files:
        print(
            f"No snapshots (CODE_YYYYMMDD.xml) in {raw_dir}: run `appalti-qa download` first.",
            file=sys.stderr,
        )
        return 1
    try:
        snapshots = []
        for path in files:
            logger.info("parsing %s", path.name)
            snapshots.append(parse_snapshot(path))
    except ValueError as exc:
        print(f"Cannot parse snapshot: {exc}", file=sys.stderr)
        return 1
    corpus = build_versions(snapshots)
    init_schema()
    with connect() as conn:
        stats = load_corpus(corpus, conn)
    print(
        f"Ingested {corpus.act.short_title}: {len(files)} files, {stats.snapshots} distinct "
        f"snapshots, {stats.events} events, {stats.units} units, {stats.versions} versions, "
        f"{stats.embeddings_kept} embeddings kept."
    )
    return 0
