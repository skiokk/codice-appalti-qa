import sys
import types
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pytest

from appalti_qa.config import Settings
from appalti_qa.retrieval import (
    FakeEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
    passage_text,
)

E5 = "intfloat/multilingual-e5-base"


def test_fake_embedder_is_normalised_and_deterministic() -> None:
    texts = ["Affidamento diretto di servizi", "Termini delle procedure", ""]
    vectors = FakeEmbedder(dim=64).embed_passages(texts)
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert np.array_equal(vectors, FakeEmbedder(dim=64).embed_queries(texts))


def test_fake_embedder_brings_texts_with_shared_words_closer() -> None:
    query, related, unrelated = FakeEmbedder().embed_queries(
        [
            "affidamento diretto dei servizi",
            "Affidamenti diretti di servizi e forniture",
            "termini massimi della procedura ristretta",
        ]
    )
    assert float(query @ related) > float(query @ unrelated) + 0.3


def test_passage_text_prepends_the_heading() -> None:
    assert (
        passage_text("Principio del risultato", "1. Testo.") == "Principio del risultato\n1. Testo."
    )
    assert passage_text(None, "1. Testo.") == "1. Testo."
    assert passage_text("  ", "1. Testo.") == "1. Testo."


def test_get_embedder_selects_and_caches_by_configuration(settings: Settings) -> None:
    fake = get_embedder(
        settings.model_copy(update={"embedding_model": "fake", "embedding_dim": 32})
    )
    assert isinstance(fake, FakeEmbedder)
    assert fake.dim == 32
    again = get_embedder(
        settings.model_copy(update={"embedding_model": "fake", "embedding_dim": 32})
    )
    assert again is fake

    e5 = get_embedder(settings.model_copy(update={"embedding_model": E5}))
    assert isinstance(e5, SentenceTransformerEmbedder)
    assert (e5.query_prefix, e5.passage_prefix) == ("query: ", "passage: ")
    bge = get_embedder(settings.model_copy(update={"embedding_model": "BAAI/bge-m3"}))
    assert isinstance(bge, SentenceTransformerEmbedder)
    assert (bge.query_prefix, bge.passage_prefix) == ("", "")


class _StubModel:
    """Stands in for sentence_transformers.SentenceTransformer: no download, records calls."""

    instances: ClassVar[list["_StubModel"]] = []

    def __init__(self, name: str, *, device: str) -> None:
        self.name, self.device = name, device
        self.calls: list[tuple[list[str], dict]] = []
        _StubModel.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), 4), dtype=np.float64) / 2.0


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[_StubModel]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _StubModel.instances = []
    return _StubModel


def test_sentence_transformer_embedder_uses_e5_prefixes(
    stub_sentence_transformers: type[_StubModel],
) -> None:
    embedder = SentenceTransformerEmbedder(E5, device="cpu", batch_size=7)
    assert not stub_sentence_transformers.instances  # lazy: nothing loaded yet

    queries = embedder.embed_queries(["soglie"])
    passages = embedder.embed_passages(["Art. 14", "Art. 50"])

    (model,) = stub_sentence_transformers.instances
    assert (model.name, model.device) == (E5, "cpu")
    assert [texts for texts, _ in model.calls] == [
        ["query: soglie"],
        ["passage: Art. 14", "passage: Art. 50"],
    ]
    _, options = model.calls[0]
    assert options["normalize_embeddings"] is True
    assert options["batch_size"] == 7
    assert queries.shape == (1, 4) and passages.shape == (2, 4)
    assert passages.dtype == np.float32
    assert embedder.dim == 4
    assert embedder.embed_passages([]).shape == (0, 4)


def _cached_model_available(name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(name, "config.json"), str)


@pytest.mark.slow
def test_real_e5_model_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _cached_model_available(E5):
        pytest.skip(f"{E5} is not in the local Hugging Face cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    embedder = SentenceTransformerEmbedder(E5)
    query = embedder.embed_queries(["Qual è la soglia per l'affidamento diretto di servizi?"])
    passages = embedder.embed_passages(
        [
            "1. Affidamento diretto dei servizi e forniture di importo inferiore a 140.000 euro.",
            "1. Le gare di appalto si concludono nei seguenti termini massimi.",
        ]
    )
    assert embedder.dim == 768
    assert np.linalg.norm(passages, axis=1) == pytest.approx([1.0, 1.0], abs=1e-4)
    scores = passages @ query[0]
    assert scores[0] > scores[1]
