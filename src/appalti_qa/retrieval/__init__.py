"""Point-in-time retrieval over the versioned units of D.Lgs. 36/2023."""

from appalti_qa.retrieval.embedding import (
    Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
    cli_embed,
    embed_missing,
    get_embedder,
    passage_text,
)
from appalti_qa.retrieval.refs import extract_refs
from appalti_qa.retrieval.search import (
    PINNED_SCORE,
    available_range,
    candidate_pool,
    history,
    lookup,
    search,
)
from appalti_qa.retrieval.urls import normattiva_url

__all__ = [
    "PINNED_SCORE",
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "available_range",
    "candidate_pool",
    "cli_embed",
    "embed_missing",
    "extract_refs",
    "get_embedder",
    "history",
    "lookup",
    "normattiva_url",
    "passage_text",
    "search",
]
