"""Command-line handlers: ``appalti-qa download`` and ``appalti-qa ingest``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from appalti_qa.db import connect, init_schema
from appalti_qa.ingest.akn import parse_snapshot, vigenza_from_filename
from appalti_qa.ingest.download import DownloadError, download_snapshots
from appalti_qa.ingest.loader import load_corpus
from appalti_qa.ingest.versioning import build_versions

logger = logging.getLogger(__name__)


def snapshot_files(raw_dir: Path) -> list[Path]:
    """Files named CODE_YYYYMMDD.xml in ``raw_dir``, by vigenza date."""
    files = []
    for path in raw_dir.glob("*.xml"):
        try:
            files.append((vigenza_from_filename(path), path))
        except ValueError:
            logger.debug("ignoring %s", path.name)
    return [path for _, path in sorted(files)]


def cli_download(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir)
    try:
        paths = download_snapshots(raw_dir, delay_s=args.delay, force=args.force)
    except (DownloadError, httpx.HTTPError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
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
