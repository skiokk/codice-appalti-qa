"""Ingestion of the Normattiva Akoma Ntoso snapshots of D.Lgs. 36/2023.

``parse_snapshot`` turns one export into units with canonical keys (commi, articles without
commi, allegato blocks), ``build_versions`` derives their point-in-time versions across all
snapshots and ``load_corpus`` writes them to PostgreSQL. ``download_snapshots`` fetches the
exports from Normattiva.
"""

from appalti_qa.ingest.akn import ActInfo, ParsedSnapshot, ParsedUnit, parse_snapshot
from appalti_qa.ingest.commands import cli_download, cli_ingest
from appalti_qa.ingest.download import download_snapshots
from appalti_qa.ingest.loader import LoadStats, load_corpus
from appalti_qa.ingest.versioning import VersionedCorpus, VersionRecord, build_versions

__all__ = [
    "ActInfo",
    "LoadStats",
    "ParsedSnapshot",
    "ParsedUnit",
    "VersionRecord",
    "VersionedCorpus",
    "build_versions",
    "cli_download",
    "cli_ingest",
    "download_snapshots",
    "load_corpus",
    "parse_snapshot",
]
