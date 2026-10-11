"""Embeddings of unit versions and queries, and the batch job that fills them in.

The production model is multilingual-e5-base through sentence-transformers, loaded lazily so that
importing this module (or running lexical search) never loads torch. `FakeEmbedder` is a
deterministic stand-in for tests and offline demos, selected with `EMBEDDING_MODEL=fake`.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import threading
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np
import psycopg

from appalti_qa.config import Settings, get_settings
from appalti_qa.db import connect

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

FAKE_MODEL_NAME = "fake"
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    """Maps texts to L2-normalised float32 vectors, shape (len(texts), dim)."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """A sentence-transformers model, loaded on first use (thread-safe).

    E5 models expect "query: " / "passage: " prefixes; other models can pass empty prefixes.
    Inputs longer than the model's window (512 tokens for e5-base) are truncated by the model.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = E5_QUERY_PREFIX,
        passage_prefix: str = E5_PASSAGE_PREFIX,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return int(self._load().get_embedding_dimension())

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.query_prefix + text for text in texts])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + text for text in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _load(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            return self._model


_WORD_RE = re.compile(r"\w+")


class FakeEmbedder:
    """Deterministic hashed bag of words: instant, offline, and texts sharing words are close.

    Words are lowercased, accent-folded and cut to 6 characters (a crude Italian stemmer), then
    hashed into `dim` signed buckets. Queries and passages are embedded the same way.
    """

    model_name = FAKE_MODEL_NAME

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _fake_tokens(text):
                digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
                vectors[row, digest % self.dim] += 1.0 if digest >> 63 else -1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
            else:  # no usable words: a fixed unit vector keeps cosine similarity defined
                vectors[row, 0] = 1.0
        return vectors


def _fake_tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [word[:6] for word in _WORD_RE.findall(folded) if len(word) > 2 or word.isdigit()]


def get_embedder(settings: Settings | None = None) -> Embedder:
    """The embedder configured in settings (cached per configuration); "fake" -> FakeEmbedder."""
    settings = settings or get_settings()
    return _embedder_for(
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_device,
        settings.embedding_batch_size,
    )


@lru_cache(maxsize=4)
def _embedder_for(model_name: str, dim: int, device: str, batch_size: int) -> Embedder:
    if model_name == FAKE_MODEL_NAME:
        return FakeEmbedder(dim)
    if "e5" in model_name.lower():
        return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
    return SentenceTransformerEmbedder(
        model_name, device=device, batch_size=batch_size, query_prefix="", passage_prefix=""
    )


def passage_text(heading: str | None, text: str) -> str:
    """The exact string embedded for a unit version (before any model prefix)."""
    heading = (heading or "").strip()
    return f"{heading}\n{text}" if heading else text


Progress = Callable[[int, int], None]

_PENDING_SQL = """
    SELECT text_hash, heading, min(text) AS text
    FROM unit_versions
    WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s
    GROUP BY text_hash, heading
    ORDER BY min(id)
    LIMIT %(limit)s
"""
_UPDATE_SQL = """
    UPDATE unit_versions SET embedding = %(embedding)s, embedding_model = %(model)s
    WHERE text_hash = %(text_hash)s AND heading IS NOT DISTINCT FROM %(heading)s
      AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %(model)s)
"""


def embed_missing(
    conn: psycopg.Connection,
    embedder: Embedder,
    *,
    batch_size: int = 32,
    limit: int | None = None,
    progress: Progress | None = None,
) -> int:
    """Embed versions with no embedding, or one from another model. Commits after each batch.

    One embedding is computed per distinct (text_hash, heading) and written to every matching row.
    `limit` caps the number of distinct passages embedded; `progress(done, total)` is called after
    each batch. Returns the number of rows updated.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = embedder.model_name
    pending = conn.execute(_PENDING_SQL, {"model": model, "limit": limit}).fetchall()
    conn.commit()
    updated = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_passages([passage_text(head, text) for _, head, text in batch])
        params = [
            {"embedding": vector, "model": model, "text_hash": text_hash, "heading": head}
            for (text_hash, head, _), vector in zip(batch, vectors, strict=True)
        ]
        with conn.cursor() as cur:
            cur.executemany(_UPDATE_SQL, params)
            updated += cur.rowcount
        conn.commit()
        if progress:
            progress(start + len(batch), len(pending))
    logger.info("Embedded %d passages (%d versions) with %s", len(pending), updated, model)
    return updated


def cli_embed(args: argparse.Namespace) -> int:
    """`appalti-qa embed`: embed the versions that lack an embedding from the configured model."""
    settings = get_settings()
    embedder = get_embedder(settings)

    def report(done: int, total: int) -> None:
        print(f"  {done}/{total} passages", flush=True)

    print(f"Embedding missing versions with {embedder.model_name}...")
    try:
        with connect(settings.database_url) as conn:
            updated = embed_missing(
                conn, embedder, batch_size=args.batch_size, limit=args.limit, progress=report
            )
    except psycopg.OperationalError as exc:
        print(f"Database not reachable: {exc}")
        return 1
    print(f"Done: {updated} versions updated." if updated else "Nothing to embed.")
    return 0
