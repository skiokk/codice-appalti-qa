from collections.abc import Sequence

import numpy as np
import pytest

from appalti_qa.retrieval import FakeEmbedder
from appalti_qa.web.querycache import CachedQueryEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=16)
        self.queries: list[list[str]] = []

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        self.queries.append(list(texts))
        return super().embed_queries(texts)


def test_repeated_queries_are_embedded_once() -> None:
    inner = CountingEmbedder()
    cached = CachedQueryEmbedder(inner, maxsize=4)
    first = cached.embed_queries(["affidamento diretto"])
    again = cached.embed_queries(["affidamento diretto"])
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, FakeEmbedder(16).embed_queries(["affidamento diretto"]))
    assert inner.queries == [["affidamento diretto"]]
    assert first.shape == (1, 16) and first.dtype == np.float32
    assert (cached.hits, cached.misses) == (1, 1)


def test_mixed_batch_embeds_only_the_missing_queries() -> None:
    inner = CountingEmbedder()
    cached = CachedQueryEmbedder(inner)
    cached.embed_queries(["b"])
    vectors = cached.embed_queries(["a", "b", "c"])
    assert inner.queries == [["b"], ["a", "c"]]
    np.testing.assert_array_equal(vectors, FakeEmbedder(16).embed_queries(["a", "b", "c"]))


def test_least_recently_used_query_is_evicted() -> None:
    inner = CountingEmbedder()
    cached = CachedQueryEmbedder(inner, maxsize=2)
    for text in ("a", "b", "a", "c", "a", "b"):
        cached.embed_queries([text])
    # "b" was evicted when "c" arrived ("a" had been used more recently)
    assert inner.queries == [["a"], ["b"], ["c"], ["b"]]


def test_returned_vectors_do_not_alias_the_cache() -> None:
    cached = CachedQueryEmbedder(CountingEmbedder())
    vectors = cached.embed_queries(["a"])
    vectors[0, 0] = 42.0
    assert cached.embed_queries(["a"])[0, 0] != 42.0


@pytest.mark.parametrize("maxsize", [0, -1])
def test_disabled_cache_passes_through(maxsize: int) -> None:
    inner = CountingEmbedder()
    cached = CachedQueryEmbedder(inner, maxsize=maxsize)
    cached.embed_queries(["a"])
    cached.embed_queries(["a"])
    assert inner.queries == [["a"], ["a"]]


def test_passages_are_not_cached() -> None:
    inner = CountingEmbedder()
    cached = CachedQueryEmbedder(inner)
    np.testing.assert_array_equal(cached.embed_passages(["testo"]), inner.embed_passages(["testo"]))
    assert cached.embed_queries([]).shape[0] == 0
