"""Shared fixtures.

Every test session runs offline: API keys are blanked and providers forced to "fake". DB tests get
a throwaway database on the compose service (port 5433), created once per session and dropped at
the end; they are skipped when the server is unreachable.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from datetime import date

import psycopg
import pytest
from psycopg import sql

os.environ.update(
    {
        "LLM_PROVIDER": "fake",
        "JUDGE_PROVIDER": "fake",
        "ANTHROPIC_API_KEY": "",
        "GROQ_API_KEY": "",
    }
)

from appalti_qa.config import Settings, get_settings
from appalti_qa.db import connect, init_schema

ADMIN_URL = os.environ.get(
    "TEST_DATABASE_ADMIN_URL", "postgresql://appalti:appalti@localhost:5433/appalti"
)
TEST_EMBEDDING_DIM = 768


def _url_for(dbname: str) -> str:
    return psycopg.conninfo.make_conninfo(ADMIN_URL, dbname=dbname)


@pytest.fixture(autouse=True)
def _fresh_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Offline settings that ignore any local .env file."""
    return Settings(_env_file=None, llm_provider="fake", judge_provider="fake")


@pytest.fixture(scope="session")
def db_url() -> Iterator[str]:
    """URL of a session-scoped throwaway database with the schema applied."""
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database unreachable: {exc}")
    name = f"appalti_test_{uuid.uuid4().hex[:12]}"
    with admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = _url_for(name)
    try:
        init_schema(url, TEST_EMBEDDING_DIM)
        yield url
    finally:
        with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def db_settings(db_url: str) -> Settings:
    """Offline settings pointing at the throwaway database."""
    return Settings(
        _env_file=None,
        database_url=db_url,
        embedding_dim=TEST_EMBEDDING_DIM,
        llm_provider="fake",
        judge_provider="fake",
    )


@pytest.fixture
def db_conn(db_url: str) -> Iterator[psycopg.Connection]:
    """A pgvector-enabled connection; every table is truncated after the test."""
    conn = connect(db_url)
    try:
        yield conn
    finally:
        conn.rollback()
        tables = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        if tables:
            conn.execute(
                sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(t) for (t,) in tables)
                )
            )
            conn.commit()
        conn.close()


# A tiny, hand-written corpus for tests that need data in the DB. Texts are abridged fixtures,
# not authoritative legal text. Keys, dates and events mirror the real structure.
SEED_ORIGINAL = date(2023, 4, 1)
SEED_AMENDMENT = date(2024, 12, 31)
SEED_UNITS: list[dict] = [
    {
        "key": "1/1",
        "kind": "comma",
        "article": "1",
        "comma": "1",
        "heading": "Principio del risultato",
        "versions": [
            (
                SEED_ORIGINAL,
                None,
                "1. Le stazioni appaltanti e gli enti concedenti perseguono il risultato "
                "dell'affidamento del contratto e della sua esecuzione con la massima "
                "tempestivita' e il migliore rapporto possibile tra qualita' e prezzo.",
                False,
            )
        ],
    },
    {
        "key": "50/1",
        "kind": "comma",
        "article": "50",
        "comma": "1",
        "heading": "Procedure per l'affidamento",
        "versions": [
            (
                SEED_ORIGINAL,
                SEED_AMENDMENT,
                "1. Le stazioni appaltanti procedono all'affidamento dei contratti sotto soglia:\n"
                "a) affidamento diretto per lavori di importo inferiore a 150.000 euro;\n"
                "b) affidamento diretto dei servizi e forniture di importo inferiore a 140.000 "
                "euro.",
                False,
            ),
            (
                SEED_AMENDMENT,
                None,
                "1. Le stazioni appaltanti procedono all'affidamento dei contratti sotto soglia:\n"
                "a) affidamento diretto per lavori di importo inferiore a 150.000 euro;\n"
                "b) affidamento diretto dei servizi e forniture, ivi compresi i servizi di "
                "ingegneria e architettura, di importo inferiore a 140.000 euro.",
                True,
            ),
        ],
    },
    {
        "key": "225-bis",
        "kind": "article",
        "article": "225-bis",
        "comma": None,
        "heading": "Ulteriori disposizioni transitorie",
        "versions": [
            (
                SEED_AMENDMENT,
                None,
                "Le disposizioni del presente articolo si applicano alle procedure avviate "
                "dopo la data di entrata in vigore del decreto correttivo.",
                True,
            )
        ],
    },
    {
        "key": "all:I.3/p1",
        "kind": "allegato_block",
        "article": None,
        "comma": None,
        "allegato": "I.3",
        "heading": "Termini delle procedure di appalto e di concessione",
        "versions": [
            (
                SEED_ORIGINAL,
                None,
                "1. Le gare di appalto si concludono nei seguenti termini massimi:\n"
                "a) procedura aperta: nove mesi;\nb) procedura ristretta: dieci mesi.",
                False,
            )
        ],
    },
]


def seed_corpus(conn: psycopg.Connection) -> None:
    """Insert SEED_UNITS (act, events, snapshots, units, versions; no embeddings) and commit."""
    act_id = conn.execute(
        "INSERT INTO acts (code, urn, title, short_title, issued_on)"
        " VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (
            "23G00044",
            "urn:nir:stato:decreto.legislativo:2023-03-31;36",
            "Codice dei contratti pubblici",
            "D.Lgs. 36/2023",
            date(2023, 3, 31),
        ),
    ).fetchone()[0]
    event_ids = {}
    for event_date, kind, urn, label in [
        (
            date(2023, 3, 31),
            "original",
            "/akn/it/act/DECRETO_LEGISLATIVO/stato/2023-03-31/36/!main",
            "D.Lgs. 31 marzo 2023, n. 36",
        ),
        (
            SEED_AMENDMENT,
            "amendment",
            "/akn/it/act/DECRETO_LEGISLATIVO/stato/2024-12-31/209/!main",
            "D.Lgs. 31 dicembre 2024, n. 209",
        ),
    ]:
        event_ids[kind] = conn.execute(
            "INSERT INTO events (act_id, event_date, kind, amending_urn, amending_label)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (act_id, event_date, kind, urn, label),
        ).fetchone()[0]
    for vigenza in (SEED_ORIGINAL, SEED_AMENDMENT):
        conn.execute(
            "INSERT INTO snapshots (act_id, vigenza_date, effective_date, source_file, sha256)"
            " VALUES (%s, %s, %s, %s, %s)",
            (act_id, vigenza, vigenza, f"23G00044_{vigenza:%Y%m%d}.xml", "0" * 64),
        )
    for ordinal, unit in enumerate(SEED_UNITS, start=1):
        unit_id = conn.execute(
            "INSERT INTO units (act_id, key, kind, article, comma, allegato, heading, ordinal)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                act_id,
                unit["key"],
                unit["kind"],
                unit["article"],
                unit["comma"],
                unit.get("allegato"),
                unit["heading"],
                ordinal,
            ),
        ).fetchone()[0]
        for valid_from, valid_to, text, amended in unit["versions"]:
            conn.execute(
                "INSERT INTO unit_versions (unit_id, heading, text, text_hash, valid_from,"
                " valid_to, changed_by_event) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    unit_id,
                    unit["heading"],
                    text,
                    hashlib.sha256(text.encode()).hexdigest(),
                    valid_from,
                    valid_to,
                    event_ids["amendment"] if amended else None,
                ),
            )
    conn.commit()


@pytest.fixture
def seeded_conn(db_conn: psycopg.Connection) -> psycopg.Connection:
    """db_conn with the SEED_UNITS corpus loaded (truncated after the test)."""
    seed_corpus(db_conn)
    return db_conn
