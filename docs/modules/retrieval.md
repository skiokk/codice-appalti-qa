# Retrieval

Package `appalti_qa.retrieval` finds the units of D.Lgs. 36/2023 in force at a date that best
answer a question, and looks units up by reference.

| Module | Content |
| --- | --- |
| `embedding.py` | `Embedder`, `SentenceTransformerEmbedder` (lazy), `FakeEmbedder`, `get_embedder`, `passage_text`, `embed_missing`, `cli_embed` |
| `search.py` | `search`, `lookup`, `history`, `available_range` |
| `fusion.py` | reciprocal rank fusion |
| `refs.py` | `extract_refs`: Italian references to canonical keys |
| `urls.py` | `normattiva_url` |

## Point in time

Every query filters `valid_from <= D AND (valid_to IS NULL OR valid_to > D)`. Searches skip
repealed versions; `lookup` and `history` return them, since "abrogato" is itself an answer.
`history` fills `change_kind` on each version; `Hit.modified_by` lists only amending acts, so
editorial versions (no act) never appear there.

## Modes

- **lexical**: PostgreSQL `italian` full-text search. `websearch_to_tsquery` AND-s every word,
  which rarely matches a natural-language question, so `&` is relaxed to `|` (quoted phrases kept)
  and `ts_rank_cd` is weighted by the share of query lexemes the unit contains. Queries with a
  negation are kept as written.
- **semantic**: `intfloat/multilingual-e5-base` on CPU (`query: ` / `passage: ` prefixes,
  normalised), cosine distance on a pgvector HNSW index. `hnsw.ef_search` and
  `hnsw.iterative_scan = relaxed_order` are set per transaction so the date filter does not starve
  results. Only embeddings of the configured model are used; if there are none, the search logs a
  warning and returns nothing without loading the model.
- **hybrid** (default): reciprocal rank fusion (`RRF_K`, default 60) of a lexical and a semantic
  list of `4*k` each. Units explicitly cited in the question ("art. 50, comma 1, lett. b)") are
  pinned first.

## Embedding

`appalti-qa embed` embeds every version without an embedding (or with one from another model),
once per distinct `(text_hash, heading)`, committing per batch. The passage is the heading line
followed by the text. On a laptop CPU the full corpus (about 3,200 distinct passages) takes
roughly 25 minutes; ingesting again keeps existing embeddings for unchanged texts.

`EMBEDDING_MODEL=fake` selects a deterministic hashed bag-of-words embedder for tests and offline
demos. `BAAI/bge-m3` (1024 dimensions) is the documented upgrade path: set `EMBEDDING_DIM`,
recreate the schema and re-embed.

## Links

`normattiva_url("225-bis", date)` gives
`https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36~art225bis!vig=YYYY-MM-DD`;
allegati link to the act.
