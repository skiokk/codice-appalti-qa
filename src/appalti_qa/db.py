"""PostgreSQL access: schema bootstrap, connections and a shared connection pool."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from appalti_qa.config import Settings, get_settings

DIM_PLACEHOLDER = "{{EMBEDDING_DIM}}"
# sql/schema.sql lives at the repository root (editable install) or in the working directory
# (container image, WORKDIR /app).
_SCHEMA_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "sql" / "schema.sql",
    Path.cwd() / "sql" / "schema.sql",
)


def schema_sql(embedding_dim: int) -> str:
    """The schema DDL with the embedding dimension filled in."""
    path = next((p for p in _SCHEMA_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("sql/schema.sql not found")
    template = path.read_text(encoding="utf-8")
    return template.replace(DIM_PLACEHOLDER, str(int(embedding_dim)))


def init_schema(database_url: str | None = None, embedding_dim: int | None = None) -> None:
    """Create extensions, tables and indexes (idempotent)."""
    settings = get_settings()
    url = database_url or settings.database_url
    dim = embedding_dim or settings.embedding_dim
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(schema_sql(dim))


def configure_connection(conn: psycopg.Connection) -> None:
    """Per-connection setup: pgvector adapters (the extension must already exist)."""
    register_vector(conn)


def connect(database_url: str | None = None, **kwargs: object) -> psycopg.Connection:
    """A standalone connection with pgvector registered. Caller closes it."""
    conn = psycopg.connect(database_url or get_settings().database_url, **kwargs)
    configure_connection(conn)
    return conn


def _open_pool(settings: Settings) -> ConnectionPool:
    return ConnectionPool(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        configure=configure_connection,
        open=True,
    )


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """The process-wide pool built from get_settings(). Use `with get_pool().connection() as c`."""
    return _open_pool(get_settings())


def close_pool() -> None:
    if get_pool.cache_info().currsize:
        get_pool().close()
        get_pool.cache_clear()
