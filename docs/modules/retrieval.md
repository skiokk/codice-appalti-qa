# Retrieval

Package `appalti_qa.retrieval` finds the units of D.Lgs. 36/2023 in force at a date that best
answer a question, and looks units up by reference.

| Module | Content |
| --- | --- |
| `embedding.py` | `Embedder`, `SentenceTransformerEmbedder` (lazy), `FakeEmbedder`, `get_embedder`, `passage_text`, `embed_missing`, `cli_embed` |
| `passages.py` | `split_passages`, `windows`, `build_passages`, `embed_passages`, `max_sim`, `hub_scores`, `update_hub_scores` |
| `search.py` | `search`, `lookup`, `history`, `available_range` |
| `fusion.py` | reciprocal rank fusion, optionally weighted |
| `refs.py` | `extract_refs`: Italian references to canonical keys |
| `urls.py` | `normattiva_url` |

## Point in time

Every query filters `valid_from <= D AND (valid_to IS NULL OR valid_to > D)`. Searches skip
repealed versions; `lookup` and `history` return them, since "abrogato" is itself an answer.
`history` fills `change_kind` on each version; `Hit.modified_by` lists only credited acts, so
editorial versions (no act) never appear there.

Headings are point-in-time too. A row of `unit_versions` is one text under one heading: when only
the article heading changes, ingestion opens a `heading` row with the same text. The heading of a
hit (shown in the interface and given to the model as the source's *Rubrica*) and the passage
matched by full-text and semantic search are therefore those in force at the query date. A hit's
validity is that of the text, across such rows; `history` folds them into the text version as
`heading_changes`, and the history table of the unit page lists them. On the corpus there are
25 such rows (arts. 57, 140, 216, 217 and all. I.5, II.3, V.2).

## Modes

- **lexical**: PostgreSQL `italian` full-text search. `websearch_to_tsquery` AND-s every word,
  which rarely matches a natural-language question, so `&` is relaxed to `|` (quoted phrases kept)
  and `ts_rank_cd` is weighted by the share of query lexemes the unit contains. Queries with a
  negation are kept as written.
- **semantic**: `intfloat/multilingual-e5-base` on CPU (`query: ` / `passage: ` prefixes,
  normalised), cosine distance on a pgvector HNSW index. `hnsw.ef_search` and
  `hnsw.iterative_scan = relaxed_order` are set per transaction so the date filter does not starve
  results. Only embeddings of the configured model are used; if there are none, the search logs a
  warning and returns nothing without loading the model. With `SEMANTIC_GRANULARITY=passage` a
  version scores its best sub-passage instead (see Passages).
- **hybrid** (default): reciprocal rank fusion (`RRF_K`, default 60) of a lexical and a semantic
  list of `FUSION_DEPTH * k` each (default 4), the lexical list weighted by `LEXICAL_WEIGHT`
  (default 1, plain RRF). Units explicitly cited in the question ("art. 50, comma 1, lett. b)")
  are pinned first.

`candidate_pool(query, date, size)` serves the reranked answers (decision 9): three lists of
`size` units (lexical, semantic on whole units, semantic on passages without hubness correction)
fused by plain RRF, with cited units pinned first (score at least `PINNED_SCORE`). It embeds the
query once. It is not a search mode: `search` and `GET /api/v1/search` never call a model.

## Embedding

`appalti-qa embed` embeds every version without an embedding (or with one from another model),
once per distinct `(text_hash, heading)`, committing per batch. The passage is the heading line
followed by the text. On a laptop CPU the full corpus (about 3,200 distinct passages) takes
roughly 25 minutes; ingesting again keeps existing embeddings for unchanged texts.

The same command then builds and embeds the passages (next section): about 10 more minutes the
first time (3,309 distinct passage texts).

`EMBEDDING_MODEL=fake` selects a deterministic hashed bag-of-words embedder for tests and offline
demos. `BAAI/bge-m3` (1024 dimensions) is the documented upgrade path: set `EMBEDDING_DIM`,
recreate the schema and re-embed.

## Passages

One vector for a long comma averages its parts: a comma with five lettere is far from a question
about one of them, and a 31,000-character allegato table is cut by e5 at 512 tokens. Each
version is therefore also split into passages (`split_passages`, stored in `unit_passages`):

- ordinal 0 is the whole unit, the same string as the unit embedding (its vector is reused);
- a paragraph with two or more lettere gives its intro alone and, for each lettera, heading +
  intro (clipped to 500 characters) + the lettera with its nested lines;
- other text longer than 1,200 characters gives windows of at most 1,000 characters cut at line
  or sentence boundaries, consecutive windows sharing at most 200 characters, each prefixed with
  the heading.

On the corpus the 3,238 versions give 7,021 passages (3,309 distinct texts besides the whole
units). Passages are embedded once per distinct text, kept across re-ingestion like the version
embeddings, and rebuilt when `PASSAGE_SCHEME` changes. With `SEMANTIC_GRANULARITY=passage` the
search reads the nearest `10 * limit` passages of the versions in force (HNSW on
`unit_passages`; on the development questions the top 10 equals an exact scan) and scores each
version by its best passage (max-sim). `HUBNESS_WEIGHT` subtracts that many times the passage's
hub score, the mean similarity to its 10 nearest probe queries (the 515 corpus headings
embedded as queries), the r(p) term of CSLS: 0.5 ranks exactly like CSLS.

**Status: available, not the default.** The variants were compared on a fixed development
split and the best one did not beat the baseline on the held-out test split, so the defaults
stay `SEMANTIC_GRANULARITY=unit`, `LEXICAL_WEIGHT=1`, `FUSION_DEPTH=4` (decision 8 in
`docs/decisions.md`, with the numbers). Passage-level semantic search alone is better than
unit-level on both splits; the loss is in the fusion.

## Links

`normattiva_url("225-bis", date)` gives
`https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36~art225bis!vig=YYYY-MM-DD`;
allegati link to the act.
