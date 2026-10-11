import argparse
import shutil
from pathlib import Path

import psycopg
import pytest

from appalti_qa.db import connect
from appalti_qa.ingest import cli_ingest
from appalti_qa.ingest import commands as ingest_commands
from appalti_qa.ingest.commands import snapshot_files


def test_snapshot_files(tmp_path: Path) -> None:
    for name in ("23G00044_20250401.xml", "23G00044_20230401.xml", "notes.xml", "x_2025.xml"):
        (tmp_path / name).write_text("<x/>")
    assert [p.name for p in snapshot_files(tmp_path)] == [
        "23G00044_20230401.xml",
        "23G00044_20250401.xml",
    ]


def test_cli_ingest_without_snapshots(tmp_path: Path, capsys) -> None:
    assert cli_ingest(argparse.Namespace(raw_dir=str(tmp_path))) == 1
    assert "appalti-qa download" in capsys.readouterr().err


def test_cli_ingest_reports_unparseable_files(tmp_path: Path, capsys) -> None:
    (tmp_path / "23G00044_20250401.xml").write_text("<html>")
    assert cli_ingest(argparse.Namespace(raw_dir=str(tmp_path))) == 1
    assert "Cannot parse snapshot" in capsys.readouterr().err


@pytest.mark.db
def test_cli_ingest_loads_the_corpus(
    monkeypatch,
    tmp_path: Path,
    capsys,
    db_url: str,
    db_conn: psycopg.Connection,
    fixture_paths: list[Path],
) -> None:
    for path in fixture_paths:
        shutil.copy(path, tmp_path / path.name)
    shutil.copy(fixture_paths[2], tmp_path / "23G00044_20250402.xml")  # identical file
    schema_calls = []
    monkeypatch.setattr(ingest_commands, "init_schema", lambda: schema_calls.append(True))
    monkeypatch.setattr(ingest_commands, "connect", lambda: connect(db_url))

    assert cli_ingest(argparse.Namespace(raw_dir=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "D.Lgs. 36/2023: 4 files, 2 distinct snapshots, 12 events" in out
    assert schema_calls == [True]
    units = db_conn.execute("SELECT count(*) FROM units").fetchone()[0]
    assert units == 36
