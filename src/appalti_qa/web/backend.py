"""What the web app reads and writes besides answering: corpus lookups and the usage log.

`DatabaseBackend` delegates corpus queries to `appalti_qa.retrieval` (imported lazily) on
connections from the shared pool, with a short acquisition timeout so that pages degrade quickly
when the database is down. Searches embed the query through a small LRU cache. Tests pass their
own `Backend` or a connection factory.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Protocol

import psycopg

from appalti_qa.config import RetrievalMode
from appalti_qa.db import get_pool
from appalti_qa.models import Answer, Hit, UnitVersion
from appalti_qa.web.querycache import QUERY_CACHE_SIZE, CachedQueryEmbedder

if TYPE_CHECKING:
    from appalti_qa.retrieval import Embedder

logger = logging.getLogger(__name__)

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]

DB_TIMEOUT_S = 3.0
RANGE_TTL_S = 300.0  # the range only changes when a new snapshot is ingested
RANGE_RETRY_S = 30.0  # retry sooner after a failure


@dataclass(frozen=True, slots=True)
class QALogEntry:
    """One row of qa_log. `client_hash` is a salted digest, never an address."""

    client_hash: str
    question: str
    at_date: date
    provider: str
    model: str
    abstained: bool = False
    cited_keys: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None

    @classmethod
    def from_answer(cls, client_hash: str, answer: Answer) -> QALogEntry:
        return cls(
            client_hash=client_hash,
            question=answer.question,
            at_date=answer.at_date,
            provider=answer.provider,
            model=answer.model,
            abstained=answer.abstained,
            cited_keys=answer.cited_keys,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            latency_ms=answer.latency_ms,
        )


class Backend(Protocol):
    def available_range(self) -> tuple[date, date] | None:
        """(earliest date with text, latest snapshot date), or None when unknown."""
        ...

    def lookup(self, ref: str, at_date: date) -> list[Hit]: ...

    def history(self, ref: str) -> list[UnitVersion]: ...

    def search(self, query: str, at_date: date, k: int, mode: RetrievalMode) -> list[Hit]: ...

    def ping(self) -> bool: ...

    def log(self, entry: QALogEntry) -> None:
        """Best effort: never raises."""
        ...


def _pool_connection() -> AbstractContextManager[psycopg.Connection]:
    return get_pool().connection(timeout=DB_TIMEOUT_S)


_INSERT_LOG = """
    INSERT INTO qa_log (client_hash, question, at_date, provider, model, abstained, cited_keys,
                        input_tokens, output_tokens, latency_ms, error)
    VALUES (%(client_hash)s, %(question)s, %(at_date)s, %(provider)s, %(model)s, %(abstained)s,
            %(cited_keys)s::text[], %(input_tokens)s, %(output_tokens)s, %(latency_ms)s, %(error)s)
"""


class DatabaseBackend:
    """Backend on PostgreSQL. Corpus queries raise `psycopg.Error` when the database fails."""

    def __init__(
        self,
        connect: ConnectionFactory = _pool_connection,
        *,
        clock: Callable[[], float] = time.monotonic,
        embedder: Embedder | None = None,
        query_cache_size: int = QUERY_CACHE_SIZE,
    ) -> None:
        """`embedder` defaults to `retrieval.get_embedder()`, wrapped on the first search."""
        self._connect = connect
        self._clock = clock
        self._range: tuple[date, date] | None = None
        self._range_expires = float("-inf")
        self._embedder = embedder
        self._query_cache_size = query_cache_size
        self._cached_embedder: CachedQueryEmbedder | None = None
        self._embedder_lock = threading.Lock()

    def query_embedder(self) -> CachedQueryEmbedder:
        """The embedder used by `search`, with its cache of recent query vectors."""
        with self._embedder_lock:
            if self._cached_embedder is None:
                from appalti_qa import retrieval

                inner = self._embedder or retrieval.get_embedder()
                self._cached_embedder = CachedQueryEmbedder(inner, self._query_cache_size)
            return self._cached_embedder

    def available_range(self) -> tuple[date, date] | None:
        now = self._clock()
        if now < self._range_expires:
            return self._range
        from appalti_qa import retrieval

        try:
            with self._connect() as conn:
                self._range = retrieval.available_range(conn=conn)
            self._range_expires = now + RANGE_TTL_S
        except psycopg.Error as exc:
            logger.warning("Text date range unavailable: %s", exc)
            self._range, self._range_expires = None, now + RANGE_RETRY_S
        return self._range

    def lookup(self, ref: str, at_date: date) -> list[Hit]:
        from appalti_qa import retrieval

        with self._connect() as conn:
            return retrieval.lookup([ref], at_date, conn=conn)

    def history(self, ref: str) -> list[UnitVersion]:
        from appalti_qa import retrieval

        with self._connect() as conn:
            return retrieval.history(ref, conn=conn)

    def search(self, query: str, at_date: date, k: int, mode: RetrievalMode) -> list[Hit]:
        from appalti_qa import retrieval

        embedder = self.query_embedder()
        with self._connect() as conn:
            return retrieval.search(query, at_date, k, mode, conn=conn, embedder=embedder)

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1")
        except psycopg.Error:
            return False
        return True

    def log(self, entry: QALogEntry) -> None:
        try:
            with self._connect() as conn:
                conn.execute(_INSERT_LOG, asdict(entry))
                conn.commit()
        except psycopg.Error as exc:
            logger.warning("Could not write qa_log: %s", exc)
