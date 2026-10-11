from datetime import date

import psycopg
import pytest

from appalti_qa.cli import build_parser
from appalti_qa.config import Settings
from appalti_qa.db import schema_sql
from appalti_qa.keys import (
    annex_from_doc_name,
    article_from_num,
    comma_from_num,
    key_label,
    lettera_from_num,
    make_key,
    parse_key,
    ref_matches,
    sort_key,
)
from appalti_qa.models import UnitVersion


def test_settings_default_offline(settings: Settings) -> None:
    assert settings.llm_provider == "fake"
    assert settings.embedding_dim == 768
    assert settings.effective_provider("anthropic") == "fake"


def test_effective_provider_uses_key_when_present() -> None:
    s = Settings(_env_file=None, llm_provider="groq", groq_api_key="test-placeholder")
    assert s.effective_provider() == "groq"


def test_schema_template_has_dimension() -> None:
    ddl = schema_sql(384)
    assert "vector(384)" in ddl
    assert "{{" not in ddl


def test_cli_has_all_commands() -> None:
    parser = build_parser()
    for command in ("download", "ingest", "embed", "serve", "eval", "init-db"):
        assert parser.parse_args([command]).command == command


@pytest.mark.parametrize(
    ("func", "num", "expected"),
    [
        (article_from_num, "Articolo 50.", "50"),
        (article_from_num, "Articolo 225-bis.", "225-bis"),
        (comma_from_num, "1.", "1"),
        (comma_from_num, "1-bis.", "1-bis"),
        (lettera_from_num, "b)", "b"),
        (lettera_from_num, "b-bis)", "b-bis"),
    ],
)
def test_num_normalisation(func, num: str, expected: str) -> None:
    assert func(num) == expected


@pytest.mark.parametrize("key", ["50", "50/1", "50/1/b", "all:I.1/2/1", "all:I.3/p3", "all:I.1"])
def test_keys_round_trip(key: str) -> None:
    assert str(parse_key(key)) == key


@pytest.mark.parametrize("key", ["", "art50", "50/", "all:I.1/", "50/1/b/c", "ALL:I.1"])
def test_invalid_keys(key: str) -> None:
    with pytest.raises(ValueError):
        parse_key(key)


def test_key_helpers() -> None:
    assert make_key("50", "1", "b") == "50/1/b"
    assert make_key(annex="I.3", block="p3") == "all:I.3/p3"
    assert ref_matches("50/1/b", "50/1")
    assert ref_matches("50", "50/1")
    assert not ref_matches("50/1", "50/10")
    assert key_label("50/1/b") == "Art. 50, comma 1, lett. b)"
    assert sorted(["4-ter", "10", "4", "4-bis"], key=sort_key) == ["4", "4-bis", "4-ter", "10"]


def test_unit_version_in_force() -> None:
    version = UnitVersion(
        unit_key="50/1",
        kind="comma",
        text="1. ...",
        text_hash="0" * 64,
        valid_from=date(2023, 4, 1),
        valid_to=date(2024, 12, 31),
    )
    assert version.in_force_at(date(2023, 4, 1))
    assert not version.in_force_at(date(2024, 12, 31))


@pytest.mark.db
def test_schema_applied(db_conn: psycopg.Connection) -> None:
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    }
    assert {"acts", "snapshots", "events", "units", "unit_versions", "qa_log"} <= tables


@pytest.mark.db
def test_versions_cannot_overlap(db_conn: psycopg.Connection) -> None:
    act_id = db_conn.execute(
        "INSERT INTO acts (code, urn, title, issued_on) VALUES (%s, %s, %s, %s) RETURNING id",
        ("23G00044", "urn:test", "Test", date(2023, 3, 31)),
    ).fetchone()[0]
    unit_id = db_conn.execute(
        "INSERT INTO units (act_id, key, kind, article, comma, ordinal)"
        " VALUES (%s, '50/1', 'comma', '50', '1', 1) RETURNING id",
        (act_id,),
    ).fetchone()[0]
    insert = (
        "INSERT INTO unit_versions (unit_id, text, text_hash, valid_from, valid_to)"
        " VALUES (%s, %s, %s, %s, %s)"
    )
    db_conn.execute(insert, (unit_id, "a", "a" * 64, date(2023, 4, 1), date(2024, 1, 1)))
    db_conn.execute(insert, (unit_id, "b", "b" * 64, date(2024, 1, 1), None))
    with pytest.raises(psycopg.errors.ExclusionViolation):
        db_conn.execute(insert, (unit_id, "c", "c" * 64, date(2023, 6, 1), None))


@pytest.mark.db
def test_seed_corpus_point_in_time(seeded_conn: psycopg.Connection) -> None:
    in_force = """
        SELECT u.key FROM unit_versions v JOIN units u ON u.id = v.unit_id
        WHERE v.valid_from <= %(d)s AND (v.valid_to IS NULL OR v.valid_to > %(d)s)
        ORDER BY u.ordinal
    """
    before = [r[0] for r in seeded_conn.execute(in_force, {"d": date(2024, 6, 1)})]
    after = [r[0] for r in seeded_conn.execute(in_force, {"d": date(2025, 1, 1)})]
    assert before == ["1/1", "50/1", "all:I.3/p1"]
    assert after == ["1/1", "50/1", "225-bis", "all:I.3/p1"]
    hits = seeded_conn.execute(
        "SELECT count(*) FROM unit_versions WHERE tsv @@ websearch_to_tsquery('italian', %s)",
        ("affidamento diretto servizi",),
    ).fetchone()[0]
    assert hits == 2


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Allegato I.1-art. 2", "I.1"),
        ("Allegati - Allegato I.01 art. 1", "I.01"),
        ("Allegato I.3", "I.3"),
        ("Allegato II.2-Allegato II.2 bis art. 1", "II.2-bis"),
        ("Allegati - Indice allegati", None),
    ],
)
def test_annex_from_doc_name(name: str, expected: str | None) -> None:
    assert annex_from_doc_name(name) == expected


def test_cli_passes_leading_options_to_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    import appalti_qa.cli as cli

    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "_resolve", lambda *_: lambda argv: seen.append(argv) or 0)
    assert cli.main(["eval", "-h"]) == 0
    assert cli.main(["eval", "retrieval", "--mode", "lexical"]) == 0
    assert seen == [["-h"], ["retrieval", "--mode", "lexical"]]


def test_cli_rejects_unknown_options() -> None:
    from appalti_qa.cli import main

    with pytest.raises(SystemExit):
        main(["init-db", "--bogus"])
