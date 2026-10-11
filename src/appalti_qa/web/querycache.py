"""A small LRU cache of query embeddings for the search API.

Embedding a query runs the model on CPU and costs far more than the SQL that follows; repeated
searches (the same question, paging through `k`, comparing modes) reuse the vector.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from appalti_qa.retrieval import Embedder

QUERY_CACHE_SIZE = 256  # about 3 KB per vector at 768 dimensions


class CachedQueryEmbedder:
    """An `Embedder` that remembers the vectors of recent queries; passages are not cached."""

    def __init__(self, inner: Embedder, maxsize: int = QUERY_CACHE_SIZE) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._vectors: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts or self._maxsize <= 0:
            return self._inner.embed_queries(texts)
        rows: list[np.ndarray | None] = []
        with self._lock:
            for text in texts:
                row = self._vectors.get(text)
                if row is not None:
                    self._vectors.move_to_end(text)
                rows.append(row)
            missing = [index for index, row in enumerate(rows) if row is None]
            self.hits += len(texts) - len(missing)
            self.misses += len(missing)
        if missing:
            computed = self._inner.embed_queries([texts[index] for index in missing])
            with self._lock:
                for index, vector in zip(missing, computed, strict=True):
                    kept = vector.copy()
                    kept.setflags(write=False)
                    rows[index] = kept
                    self._vectors[texts[index]] = kept
                    self._vectors.move_to_end(texts[index])
                while len(self._vectors) > self._maxsize:
                    self._vectors.popitem(last=False)
        return np.stack([row for row in rows if row is not None])

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._inner.embed_passages(texts)
