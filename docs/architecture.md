# Architecture and module contract

This document describes the packages of `appalti_qa` and the interfaces between them. Packages
use each other only through the public names listed below, and the tests of a package replace the
packages it depends on with fakes or monkeypatching rather than relying on their internals.

```
 Normattiva AKN snapshots ──► ingest ──► PostgreSQL (units, unit_versions, events)
                                              │  tsvector (italian)  +  pgvector (e5-base)
                                              ▼
 question + date ──► answer ──► retrieval.search ──► Hit[] ──► llm provider ──► Answer
                       ▲                                                          │
                 web (FastAPI, htmx, Bootstrap Italia)      evaluation (dataset, metrics, judge, report)
```

## 1. Package boundaries

| Package | Public API (`__init__.py`) | Depends on | Notes |
| --- | --- | --- | --- |
| shared modules | `config`, `db`, `keys`, `models`, `cli` (section 3) | — | settings, connections, canonical keys, shared models, the `appalti-qa` entry point |
| `ingest` | `parse_snapshot`, `build_versions`, `load_corpus`, `download_snapshots`, `refine_snapshots`, `cli_download`, `cli_ingest`, dataclasses (section 4) | `db`, `keys`, `models` | the only writer of the corpus tables |
| `retrieval` | `search`, `lookup`, `history`, `available_range`, `extract_refs`, `normattiva_url`, embedders, `embed_missing`, `cli_embed` (section 5) | `config`, `db`, `keys`, `models` | read-only on the corpus, except embeddings |
| `llm` | `get_provider`, `LLMProvider`, `Generation`, `LLMError`, the providers, `ABSTAIN_TOKEN` (section 6) | `config`, `models` | no database access |
| `answer` | `answer_question`, `Retriever`, `ABSTAIN_MESSAGE` (section 6) | `llm`, `retrieval` (lazily, default retriever only), `config`, `models` | |
| `web` | `create_app`, `cli_serve`, `AnswerFn` (section 7) | `answer`, `retrieval`, `llm` (`LLMError`), shared modules | writes only `qa_log`; reads `eval/results/latest.json` |
| `evaluation` | dataset, metrics, judges, xlsx review, reports, `cli_main` (section 8) | `answer`, `retrieval`, `llm`, shared modules | writes `eval/results/**`; edits `eval/dataset.yaml` only through `import-xlsx` |

Imports go one way: `ingest` and `retrieval` never import `llm`, `answer`, `web` or `evaluation`,
and only the `cli` entry point imports `web` and `evaluation`. Each package's `__init__.py` re-exports exactly the
public names listed for it. `eval/dataset.yaml` holds questions checked by a human reviewer; the
evaluation package reads it and writes it back only through `import-xlsx` (section 8).

Contributor guidelines:

- A change that crosses packages (a dependency, a setting, a column, a fixture) goes in the shared
  modules first and is recorded in this document.
- Everything runs offline by default: `LLM_PROVIDER=fake`, no API keys. Never require a key in
  tests; tests must not hit the network (use `httpx.MockTransport` or fakes).
- DB tests use the fixtures in `tests/conftest.py` (`db_url`, `db_conn`, `seeded_conn`,
  `db_settings`) and are marked `@pytest.mark.db`. They run on a throwaway database, never on the
  main one.
- Tests that load real models or parse the full corpus are marked `@pytest.mark.slow` and skip
  cleanly when the resource is missing (CI runs `pytest -m "not slow"`).
- English for code, comments and docs; Italian for user-facing UI text, prompts shown to users and
  eval questions. Use `logging.getLogger(__name__)`, not `print`, except in `cli_*` functions.
- Lint: `ruff check . && ruff format --check .` (line length 100). Typed, small modules.

## 2. Canonical unit key

Implemented in `appalti_qa.keys`, the single implementation every package uses.

- Article number from `<num>`: `"Articolo 50."` → `"50"`; `"Articolo 225-bis."` → `"225-bis"`.
- Comma from paragraph `<num>`: `"1."` → `"1"`; `"1-bis."` → `"1-bis"`.
- Lettera from point `<num>`: `"b)"` → `"b"`; `"b-bis)"` → `"b-bis"`.
- Key: `"50"` | `"50/1"` | `"50/1/b"`.
- Allegati: `"all:I.1/2/1"` (allegato code from the doc name, then article/comma/lettera inside it
  when the allegato is structured in articles); an allegato without articles uses numbered blocks,
  `"all:I.3/p3"` for its 3rd numbered block. Other unnumbered parts (tables, schede) use a
  lowercase slug starting with a letter, e.g. `"all:I.4/tabella-a"`.
- Retrieval units are commi. An article without commi is one unit (key `"225-bis"`). Long commi stay
  one unit. A reference to a lettera matches its comma unit by prefix.

API (`appalti_qa.keys`):

```python
LATIN_SUFFIXES: tuple[str, ...]                       # "bis", "ter", "quater", ...
def article_from_num(num: str) -> str                 # ValueError if not an article number
def comma_from_num(num: str) -> str
def lettera_from_num(num: str) -> str
def annex_from_doc_name(name: str) -> str | None      # "Allegato I.1-art. 2" -> "I.1"
@dataclass(frozen=True) class UnitKey:                 # annex, article, comma, lettera, block
    unit: UnitKey                                      # property: same key without the lettera
def make_key(article=None, comma=None, lettera=None, *, annex=None, block=None) -> str
def parse_key(key: str) -> UnitKey                    # ValueError on invalid keys
def is_valid_key(key: str) -> bool
def ref_matches(ref: str, unit_key: str) -> bool      # unit == ref, unit under ref, or ref under unit
def key_label(key: str) -> str                        # "50/1/b" -> "Art. 50, comma 1, lett. b)"
def sort_key(key: str) -> tuple                       # document order, "4" < "4-bis" < "4-ter" < "5"
```

Observed in the corpus:

- Snapshot 2026-08-26 (the text in force on 2026-10-11): 233 articles, 1319 paragraphs, 690
  points, 323 allegato docs. The 24 paragraphs without `<num>` are not single-comma articles:
  they are headings exported as paragraphs, Normattiva "AGGIORNAMENTO" notes, and the repealed
  art. 109, which is the only article-level unit. In force on that date: 2,811 units (1,319 commi,
  1 article unit, 1,491 allegato units across 41 allegati); 2,814 units over all snapshots.
- Exports up to 2024-12-31 are flat: article `<num>` is `"Art. 50."` and each article is one
  paragraph of running text with commi and lettere inline, so commi and lettere are found in the
  text (a comma number counts only after punctuation, continuing the numbering, outside «...»
  quotes). Exports from 2025-03-15 (the first refined boundary of 2025) have one `<paragraph>`
  per comma and one `<point>` per lettera.
- Allegato docs are named `"Allegato I.1-art. 2"`, `"Allegati - Allegato I.01 art. 1"`,
  `"Allegato I.4-art 1"`, `"Allegato I.7-art. 4 bis"`, `"Allegato I.3"` (no articles),
  `"Allegato I.4-Tabella A"`, `"Allegato II.2-Allegato II.2 bis Tabella A"` (the last allegato named
  wins: `II.2-bis`); `"Allegati - Indice allegati"` is not content. Allegato bodies are flat text in
  every snapshot and are split into commi by line-leading numbers. An allegato with no numbered
  blocks at all is one unit with the bare key `"all:X"`.

## 3. Shared code

### 3.1 `appalti_qa.config`

```python
ProviderName = Literal["fake", "anthropic", "groq"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]

class Settings(BaseSettings):            # env vars / .env, case-insensitive, extra ignored
    database_url: str = "postgresql://appalti:appalti@localhost:5433/appalti"
    db_pool_min_size: int = 1; db_pool_max_size: int = 10
    raw_dir: Path = Path("data/raw/akn")
    embedding_model: str = "intfloat/multilingual-e5-base"   # "fake" selects FakeEmbedder
    embedding_dim: int = 768; embedding_device: str = "cpu"; embedding_batch_size: int = 32
    llm_provider: ProviderName = "fake"
    llm_model: str = "claude-opus-5-5"; llm_effort: Effort = "medium"
    llm_max_tokens: int = 16000; llm_timeout_s: float = 60.0   # ceiling; reasoning tokens count
    groq_model: str = "openai/gpt-oss-120b"
    anthropic_api_key: SecretStr | None = None; groq_api_key: SecretStr | None = None
    judge_provider: ProviderName = "fake"; judge_model: str = "claude-opus-5-5"
    default_top_k: int = 8; retrieval_mode: RetrievalMode = "hybrid"; rrf_k: int = 60
    min_relevance_score: float = 0.0
    max_question_chars: int = 500
    rate_limit_per_ip_per_hour: int = 20; rate_limit_global_per_day: int = 500
    rate_limit_search_per_ip_per_hour: int = 120     # GET /api/v1/search, separate budget
    client_hash_salt: SecretStr | None = None
    forwarded_allow_ips: str = ""          # trusted reverse proxies (IPs/CIDRs) for X-Forwarded-For;
                                           # "*": only the rightmost hop is used
    log_level: str = "INFO"

    def api_key(self, provider: ProviderName) -> str | None
    def effective_provider(self, provider: ProviderName | None = None) -> ProviderName
        # requested provider (default llm_provider), or "fake" when its key is missing

def get_settings() -> Settings          # cached; tests call get_settings.cache_clear()
```

Functions that take `settings: Settings | None = None` use `get_settings()` when it is `None`.

### 3.2 `appalti_qa.db`

```python
def schema_sql(embedding_dim: int) -> str                      # sql/schema.sql, {{EMBEDDING_DIM}} filled
def init_schema(database_url: str | None = None, embedding_dim: int | None = None) -> None  # idempotent
def configure_connection(conn: psycopg.Connection) -> None     # register_vector
def connect(database_url: str | None = None, **kwargs) -> psycopg.Connection   # caller closes
def get_pool() -> psycopg_pool.ConnectionPool                  # cached, from get_settings()
def close_pool() -> None
```

Connections return tuples; use `conn.cursor(row_factory=dict_row)` where convenient. Embeddings
are `numpy.ndarray` (float32) or `pgvector.Vector` on the way in and come back as `pgvector.Vector`
(use `.to_numpy()` for an array; a `Vector` can be written back unchanged).
Functions that take `conn: psycopg.Connection | None = None` use `with get_pool().connection()`
when it is `None`. Functions given a connection do not close it and do not commit unless their
docstring says so (writers like `load_corpus` and `embed_missing` commit).

### 3.3 Database schema (`sql/schema.sql`)

- `acts(id, code UNIQUE "23G00044", urn UNIQUE "urn:nir:stato:decreto.legislativo:2023-03-31;36", title, short_title, issued_on)`
- `snapshots(id, act_id, vigenza_date, effective_date, source_file, sha256, ingested_at)`, unique `(act_id, vigenza_date)`;
  one row per *distinct* snapshot actually used; `effective_date` equals `vigenza_date` (section 4).
- `events(id, act_id, event_date, kind 'original'|'amendment', amending_urn, amending_label)`,
  unique `(act_id, amending_urn, event_date)`.
- `units(id, act_id, key, kind 'article'|'comma'|'allegato_block', article, comma, allegato, heading, ordinal)`,
  unique `(act_id, key)`. Every unit inside an allegato has kind `allegato_block` (with `allegato`
  set and `article`/`comma` filled when the allegato is structured). `heading` = latest heading;
  `ordinal` = document order over the whole act (code body, then allegati in order).
- `unit_versions(id, unit_id, heading, text, text_hash, valid_from, valid_to, changed_by_event, change_kind, repealed, tsv, embedding vector(DIM), embedding_model)`.
  - `change_kind` is `'original'`, `'amendment'` (default) or `'editorial'`; `changed_by_event`
    is set only for amendments. The schema adds the column to databases created without it
    (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`), so `init-db` or `ingest` upgrades them.
  - In force at date D: `valid_from <= D AND (valid_to IS NULL OR valid_to > D)`; `valid_to` is
    exclusive. An exclusion constraint forbids overlapping intervals for the same unit.
  - `text_hash` = sha256 hex of `text` (UTF-8).
  - `tsv` is generated: heading (weight A) + text (weight B), `italian` config, GIN index.
  - `embedding` has an HNSW `vector_cosine_ops` index; `embedding_model` records the model name.
- `qa_log(id, created_at, client_hash, question, at_date, provider, model, abstained, cited_keys text[], input_tokens, output_tokens, latency_ms, error)`
  for the web app (usage, rate limiting across restarts, cost tracking). Never store raw IPs.

### 3.4 `appalti_qa.models`

```python
UnitKind = Literal["article", "comma", "allegato_block"]
ChangeKind = Literal["original", "amendment", "editorial"]

class Event(BaseModel):
    id: int | None = None; date: date; kind: Literal["original", "amendment"] = "amendment"
    amending_urn: str; label: str                       # label e.g. "D.L. 10 maggio 2023, n. 51"

class UnitVersion(BaseModel):
    id: int | None = None; unit_key: str; kind: UnitKind
    article: str | None; comma: str | None; allegato: str | None; heading: str | None
    text: str; text_hash: str; valid_from: date; valid_to: date | None
    changed_by: Event | None; change_kind: ChangeKind = "amendment"; repealed: bool = False
    def in_force_at(self, day: date) -> bool

class Hit(BaseModel):
    unit_key: str; version_id: int; heading: str | None; text: str; score: float
    valid_from: date; valid_to: date | None; normattiva_url: str
    modified_by: list[Event] = []                       # amendments to this unit up to the query date
                                                        # (editorial versions have no event)
    label: str                                          # property, key_label(unit_key)

class Citation(BaseModel):
    unit_key: str; cited_text: str; source_index: int   # 0-based index into Answer.sources

class Answer(BaseModel):
    question: str; at_date: date; text: str
    citations: list[Citation] = []; sources: list[Hit] = []
    abstained: bool = False; model: str; provider: str
    input_tokens: int = 0; output_tokens: int = 0; latency_ms: int = 0
    cited_keys: list[str]                               # property, distinct, in citation order
```

### 3.5 `appalti_qa.cli`

`appalti-qa <command>`; each handler imports its package lazily, so a missing optional dependency
only affects its own command, which then prints a clear message (exit 2).

| Command | Delegates to | Arguments (argparse `Namespace` attributes) |
| --- | --- | --- |
| `init-db` | `appalti_qa.db.init_schema()` | — |
| `download` | `appalti_qa.ingest.cli_download(args) -> int` | `raw_dir: str`, `delay: float = 2.0`, `force: bool`, `refine: bool`, `max_probes: int = 150` |
| `ingest` | `appalti_qa.ingest.cli_ingest(args) -> int` | `raw_dir: str` |
| `embed` | `appalti_qa.retrieval.cli_embed(args) -> int` | `batch_size: int`, `limit: int \| None` |
| `serve` | `appalti_qa.web.cli_serve(args) -> int` | `host: str = "127.0.0.1"`, `port: int = 8000`, `reload: bool` |
| `eval ...` | `appalti_qa.evaluation.cli_main(argv: list[str]) -> int` | raw remaining argv, leading options included (`eval -h`); the evaluation package defines its sub-parser |

### 3.6 Test fixtures (`tests/conftest.py`)

The conftest forces `LLM_PROVIDER=fake`, `JUDGE_PROVIDER=fake` and blank API keys for the whole
session and clears the `get_settings()` cache around each test.

- `settings -> Settings`: offline settings ignoring `.env`.
- `db_url -> str` (session): a throwaway database `appalti_test_<random>` on the compose server
  (`TEST_DATABASE_ADMIN_URL`, default `postgresql://appalti:appalti@localhost:5433/appalti`), schema
  applied with dim 768, dropped at session end; skips if the server is unreachable.
- `db_settings -> Settings`: offline settings with `database_url=db_url`.
- `db_conn -> psycopg.Connection`: pgvector-enabled connection; all tables truncated afterwards.
- `seeded_conn -> psycopg.Connection`: `db_conn` plus `seed_corpus()` (committed): units `1/1`,
  `50/1` (two versions, changed on 2024-12-31 by "D.Lgs. 31 dicembre 2024, n. 209"; versions
  from the first snapshot have `change_kind` `original`), `225-bis`
  (article without commi, from 2024-12-31), `all:I.3/p1`; constants `SEED_ORIGINAL`
  (2023-04-01), `SEED_AMENDMENT` (2024-12-31), `SEED_UNITS`. No embeddings.

Code that uses the pool (`get_pool()`) in tests must point it at the test DB: set
`DATABASE_URL` via `monkeypatch.setenv(...)` and clear `get_settings`/`get_pool` caches, or (better)
pass `conn=` explicitly.

## 4. ingest

Package `appalti_qa.ingest`; exports `parse_snapshot`, `build_versions`, `load_corpus`,
`download_snapshots`, `refine_snapshots`, `cli_download`, `cli_ingest`, and the dataclasses below
(`ActInfo`, `ParsedSnapshot`, `ParsedUnit`, `VersionRecord`, `VersionedCorpus`, `LoadStats`,
`RefineReport`).

```python
# appalti_qa/ingest/akn.py
AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

@dataclass(frozen=True, slots=True)
class ActInfo:
    code: str; urn: str; title: str; short_title: str; issued_on: date

@dataclass(frozen=True, slots=True)
class ParsedUnit:
    key: str; kind: UnitKind; article: str | None; comma: str | None; allegato: str | None
    heading: str | None; text: str; ordinal: int; repealed: bool = False
    amended: bool = False              # the export marks (part of) the text as inserted/replaced

@dataclass(frozen=True, slots=True)
class ParsedSnapshot:
    vigenza_date: date                 # from the file name 23G00044_YYYYMMDD.xml
    expression_date: date | None       # FRBRExpression/FRBRdate
    source_file: str; sha256: str      # sha256 of the file bytes
    act: ActInfo
    events: list[Event]                # from meta/lifecycle/eventRef + references (original included)
    units: list[ParsedUnit]

def normalize_text(text: str) -> str   # drop "((" "))" markers, normalise whitespace per line
def text_hash(text: str) -> str        # sha256 hex of the UTF-8 text
def parse_snapshot(path: Path) -> ParsedSnapshot
```

Text rules: a comma's `text` starts with its number (`"1. ..."`); the list intro and each point go
on their own line (`"a) ..."`), nested points likewise. Article-level units (no commi) contain the
article body only. Headings come from `<heading>` (allegati: the article heading line, or the
allegato title for blocks). `<ins>`/`((...))` markers are removed from the stored text. A unit whose
text says it was repealed (e.g. `((ARTICOLO ABROGATO ...))`, `((COMMA ABROGATO ...))`) is kept
with `repealed=True`. Event labels are Italian and human-readable (`"D.L. 10 maggio 2023, n. 51"`,
`"L. 3 luglio 2023, n. 87"`), built from the `passiveRef` href.

```python
# appalti_qa/ingest/versioning.py
@dataclass(frozen=True, slots=True)
class VersionRecord:
    unit_key: str; heading: str | None; text: str; text_hash: str
    valid_from: date; valid_to: date | None; changed_by: Event | None; repealed: bool
    change_kind: ChangeKind = "amendment"

@dataclass(slots=True)
class VersionedCorpus:
    act: ActInfo
    snapshots: list[tuple[ParsedSnapshot, date]]   # distinct snapshots used, with effective date
    events: list[Event]                            # union over all snapshots, by date
    units: list[ParsedUnit]                        # latest metadata per key, final ordinals
    versions: list[VersionRecord]

def content_digest(snapshot: ParsedSnapshot) -> str   # keys, headings and wording; layout ignored
def effective_date(snapshot: ParsedSnapshot, previous_vigenza: date | None) -> date
    # the vigenza_date; the expression date is only compared with it and disagreements logged
def build_versions(snapshots: Sequence[ParsedSnapshot]) -> VersionedCorpus
```

Versioning rules (decision 7 in `docs/decisions.md`):

1. Sort by `vigenza_date`. A snapshot whose `content_digest` equals the previous one adds no
   version; it only refreshes the stored rendering of the open versions.
2. A version takes effect on the `vigenza_date` of the snapshot that introduced it. After
   `appalti-qa download --refine` every file is a boundary named by the first day Normattiva
   serves its content, so this is the exact date of the change. The FRBRExpression date is stale
   in several exports and is only compared: `INFO` when stale, `WARNING` when it falls between
   the previous file and this one (the directory was not refined) or after the file date.
3. For each key, a new version starts when the wording differs from the open version
   (`content_fingerprint`: whitespace, dots, hyphens, quotes, apostrophe-style accents and
   `((...))` markers ignored), so the 2025 switch from flat to structured exports creates no
   versions. A change of the heading alone updates the open version's heading without a new
   version, and an unchanged version keeps the latest rendering of its text. The open version
   gets `valid_to` = the new `valid_from`. A key missing from a later snapshot is closed at that
   snapshot's date; if it reappears it gets a new version.
4. `change_kind`: `original` for versions of the first snapshot; `editorial` when only
   punctuation or capitalisation changed (same `word_fingerprint`) or the article's words are
   split into commi differently; `amendment` otherwise (including a unit that appears or
   reappears). `changed_by` of an amendment = the latest amendment dated on or before
   `valid_from` in the lifecycle of the snapshot that introduced it (all known events when that
   lifecycle has none); `None` for original and editorial versions. Doubtful attributions (no
   act since the previous change, several acts in the interval, an act published but missing
   from the export's lifecycle) are logged.
5. Versions of the last snapshot stay open (`valid_to=None`).

```python
# appalti_qa/ingest/loader.py
@dataclass(frozen=True, slots=True)
class LoadStats:
    snapshots: int; events: int; units: int; versions: int; embeddings_kept: int

def load_corpus(corpus: VersionedCorpus, conn: psycopg.Connection) -> LoadStats   # commits
```

`load_corpus` replaces the act's data in one transaction and is idempotent. It must preserve
existing embeddings: before replacing, keep `(text_hash, heading) -> (embedding, embedding_model)`
and restore them on matching new rows.

```python
# appalti_qa/ingest/download.py
def download_snapshots(raw_dir: Path, *, delay_s: float = 2.0, force: bool = False,
                       client: httpx.Client | None = None, today: date | None = None) -> list[Path]
class NormattivaFetcher:             # (vigenza, path) -> path; one session cookie, delay_s apart
    def __init__(self, *, delay_s: float = 2.0, client: httpx.Client | None = None) -> None

# appalti_qa/ingest/refine.py
@dataclass(slots=True)
class RefineReport:
    boundaries: list[date]; probes: int; unresolved: list[tuple[date, date]]; deleted: list[str]
def refine_snapshots(raw_dir: Path, fetch: Callable[[date, Path], Path], *,
                     max_probes: int = 150,
                     digest: Callable[[Path], str] = file_digest) -> RefineReport

# appalti_qa/ingest/commands.py
def cli_download(args: argparse.Namespace) -> int    # download, then refine with --refine
def cli_ingest(args: argparse.Namespace) -> int      # parse raw_dir, build, init_schema, load, print stats
```

The downloader is polite (one request at a time, `delay_s` between requests, a descriptive
User-Agent, a session cookie from the act's page), caches files in `raw_dir` (skips existing
unless `force`), discovers the vigenza dates from the lifecycle of the latest snapshot, and never
runs in tests against the real site. `refine_snapshots` bisects on `dataVigenza` between
neighbouring files whose `content_digest` differs, keeps only boundary files named by their first
day, and records `verified_through`, the boundaries and any gap left by `max_probes` in
`raw_dir/refined.json`; later downloads skip lifecycle dates up to `verified_through`, and later
refines resume the open gaps.
Tests: hand-written minimal AKN fixtures in `tests/ingest/fixtures/`; a `slow` test may parse the
real `data/raw/akn` when present.

## 5. retrieval

Package `appalti_qa.retrieval`; exports `Embedder`, `SentenceTransformerEmbedder`, `FakeEmbedder`,
`get_embedder`, `passage_text`, `embed_missing`, `cli_embed`, `search`, `lookup`, `history`,
`extract_refs`, `normattiva_url`, `available_range`.

```python
# appalti_qa/retrieval/embedding.py
class Embedder(Protocol):
    model_name: str                     # read-only properties: plain attributes conform
    dim: int
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray    # (n, dim) float32, L2-normalised
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray

class SentenceTransformerEmbedder:      # lazy model load; "query: " / "passage: " prefixes (e5)
    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32,
                 query_prefix: str = "query: ", passage_prefix: str = "passage: ") -> None
class FakeEmbedder:                     # deterministic hashed bag-of-words, model_name "fake"
    def __init__(self, dim: int = 768) -> None

def get_embedder(settings: Settings | None = None) -> Embedder   # cached; "fake" -> FakeEmbedder
def passage_text(heading: str | None, text: str) -> str         # exact string embedded (no prefix)
def embed_missing(conn: psycopg.Connection, embedder: Embedder, *, batch_size: int = 32,
                  limit: int | None = None,
                  progress: Callable[[int, int], None] | None = None) -> int
                  # rows updated; commits per batch; limit caps distinct passages
def cli_embed(args: argparse.Namespace) -> int
```

`embed_missing` embeds versions whose `embedding` is NULL or whose `embedding_model` differs from
`embedder.model_name`, computing one embedding per distinct `(text_hash, heading)` and writing it to
every matching row. The passage is `passage_text(heading, text)` = heading line + text.

```python
# appalti_qa/retrieval/search.py
def search(query: str, at_date: date, k: int = 8, mode: RetrievalMode = "hybrid", *,
           conn: psycopg.Connection | None = None, embedder: Embedder | None = None,
           settings: Settings | None = None) -> list[Hit]
def lookup(refs: Sequence[str], at_date: date, *, conn: psycopg.Connection | None = None) -> list[Hit]
def history(ref: str, *, conn: psycopg.Connection | None = None) -> list[UnitVersion]
def available_range(*, conn: psycopg.Connection | None = None) -> tuple[date, date] | None

# appalti_qa/retrieval/refs.py
def extract_refs(text: str) -> list[str]     # "art. 50, comma 1, lett. b)" -> ["50/1/b"]; allegati too

# appalti_qa/retrieval/urls.py
def normattiva_url(unit_key: str, at_date: date) -> str
```

- Only versions in force at `at_date` and not `repealed` are searched.
- `lexical`: `websearch_to_tsquery('italian', query)` on `tsv`, ranked by `ts_rank_cd`. Because
  AND-ing every word of a natural-language question rarely matches, `&` is relaxed to `|`
  (phrases kept) and the rank is weighted by the share of query lexemes the unit contains; queries
  containing a negation are kept as written.
- `semantic`: cosine similarity (`1 - (embedding <=> q)`) with the query embedding; use
  `SET LOCAL hnsw.ef_search` / `hnsw.iterative_scan` (pgvector 0.8) so the date filter does not
  starve results; rows without embeddings are ignored.
- `hybrid`: reciprocal rank fusion of the lexical and semantic lists (each `4*k` long) with
  `settings.rrf_k`; units explicitly referenced in the query (`extract_refs`) are placed first
  (score `1.0` + their fusion score). Without embeddings for the configured model, semantic search
  logs a warning and returns `[]` without loading the model, so hybrid degrades to lexical.
- `Hit.score` is the mode's score (higher is better); ties broken by `units.ordinal`.
  `Hit.modified_by` = events of the unit's versions with `valid_from <= at_date`, by date.
- `lookup` returns every unit matching any ref (`keys.ref_matches`) in force at `at_date`, in
  document order, score `1.0`. `history` returns all versions of matching units, ordered by
  ordinal then `valid_from`, with `changed_by` filled; both return `[]` for invalid keys.
  `available_range` = (earliest `valid_from`, latest snapshot `vigenza_date`) or `None` if empty.
- `normattiva_url`: `https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36~art{N}!vig={YYYY-MM-DD}`
  where `{N}` is the article without dash (`225-bis` → `225bis`); for allegati, the act URL without
  the `~art` part.
- Tests use `seeded_conn` + `FakeEmbedder` (call `embed_missing` first for semantic/hybrid).

## 6. llm and answer

Package `appalti_qa.llm`; exports `Generation`, `LLMProvider`, `LLMError`, `FakeProvider`,
`AnthropicProvider`, `GroqProvider`, `get_provider`, `ABSTAIN_TOKEN`.

```python
# appalti_qa/llm/base.py
class LLMError(RuntimeError): ...       # provider/network failure, message safe to log

@dataclass(frozen=True, slots=True)
class Generation:
    text: str                           # with "[n]" markers, n = source_index + 1
    citations: list[Citation]
    model: str; provider: str
    input_tokens: int = 0; output_tokens: int = 0
    stop_reason: str | None = None

class LLMProvider(Protocol):
    name: ProviderName
    model: str
    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation
    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation

def get_provider(settings: Settings | None = None, *, provider: ProviderName | None = None,
                 model: str | None = None) -> LLMProvider
    # uses settings.effective_provider(provider): "fake" whenever the key is missing;
    # default models: anthropic -> settings.llm_model, groq -> settings.groq_model, fake -> "fake"

# appalti_qa/llm/prompts.py
ABSTAIN_TOKEN = "NON_TROVATO"
SYSTEM_PROMPT: str                      # Italian; answer only from the sources, cite every claim,
                                        # state the reference date, reply ABSTAIN_TOKEN if unsupported
```

- `AnthropicProvider(settings, model=None)`: Messages API with native citations. One `document`
  block per source (`source: {"type": "text", "media_type": "text/plain", "data": hit.text}`,
  `title`: label + validity, `citations: {"enabled": True}`), then the question and the reference
  date. `output_config={"effort": settings.llm_effort}`; no `thinking` disabling and no
  structured output (incompatible with citations); stable system prompt with `cache_control`.
  Opt in to server-side refusal fallback (`betas=["server-side-fallback-2026-07-01"]`,
  `fallbacks="default"`) and treat `stop_reason == "refusal"` as abstention. Each `char_location`
  citation maps `document_index` → `source_index` and keeps the exact `cited_text`; the provider
  appends `[n]` after each cited text block. Tokens from `usage`.
- `GroqProvider(settings, model=None)`: chat completions with sources numbered `[1]..[n]` in the
  prompt; markers in the output are parsed into citations (`source_index = n - 1`, out-of-range
  markers dropped). `cited_text` is best effort: the sentence of the source with the highest word
  overlap with the answer sentence carrying the marker.
- `FakeProvider(model="fake")`: deterministic, no network. `generate` with no sources returns
  `ABSTAIN_TOKEN`; otherwise a short Italian answer quoting the first sentence of the first one or
  two sources, with `[n]` markers and exact `cited_text` substrings. `complete` returns
  `'{"fake": true}'`.
- Map SDK errors (timeouts, rate limits, 5xx, auth) to `LLMError`. Never log keys or full prompts
  at INFO level.

Package `appalti_qa.answer`; exports `answer_question`, `Retriever`, `ABSTAIN_MESSAGE`.

```python
# appalti_qa/answer/service.py
Retriever = Callable[[str, date, int], list[Hit]]    # (question, at_date, k)
ABSTAIN_MESSAGE = ("Le fonti disponibili non consentono di rispondere a questa domanda "
                   "con riferimento alla data indicata.")   # wording may be refined

def answer_question(question: str, at_date: date, settings: Settings | None = None, *,
                    provider: LLMProvider | None = None, retriever: Retriever | None = None,
                    k: int | None = None, mode: RetrievalMode | None = None) -> Answer
```

1. Strip the question; raise `ValueError` (Italian message) if empty or longer than
   `settings.max_question_chars`.
2. Retrieve `k or settings.default_top_k` hits (default retriever: `retrieval.search` with
   `mode or settings.retrieval_mode`, imported lazily).
3. No hits, or best score below `settings.min_relevance_score` → abstain without calling the LLM.
4. Otherwise `provider.generate(...)`; drop citations whose `source_index` is out of range; abstain
   when the text is `ABSTAIN_TOKEN` (or starts with it) or no valid citation remains.
5. Abstained answers have `text = ABSTAIN_MESSAGE`, `abstained=True`, empty citations, and keep
   `sources` for transparency. `latency_ms` covers retrieval + generation.

Tests use `FakeProvider` and a stub retriever; no DB needed.

## 7. web

Package `appalti_qa.web`; exports `create_app`, `cli_serve`, `AnswerFn`.

```python
# appalti_qa/web/app.py
AnswerFn = Callable[[str, date], Answer]

def create_app(settings: Settings | None = None, *, answer_fn: AnswerFn | None = None,
               backend: Backend | None = None,          # corpus lookups + qa_log (tests: in-memory)
               results_path: Path = Path("eval/results/latest.json"),
               trusted_proxies: Iterable[str] | None = None) -> FastAPI   # default: settings
def cli_serve(args: argparse.Namespace) -> int     # uvicorn "appalti_qa.web:create_app", factory=True
```

`create_app` with no arguments must work (the Docker image runs
`uvicorn appalti_qa.web:create_app --factory`). Default `answer_fn` calls
`answer.answer_question(q, d, settings)`.

Pages (Italian, Bootstrap Italia, htmx; templates in `web/templates/`, static in `web/static/`):

- `GET /` — question form (textarea with counter up to `max_question_chars`, "Testo vigente al"
  date input defaulting to today, bounded by `available_range()` when known).
- `POST /chiedi` — htmx partial (full page without htmx): answer with numbered citation markers,
  cited passages, sources with heading, validity interval, "modificato da" events and Normattiva
  link, model/provider/latency, abstention notice, and a legal disclaimer.
- `GET /norma/{key:path}?data=YYYY-MM-DD` — text of the unit(s) in force at the date (via
  `retrieval.lookup`) and their version history (`retrieval.history`).
- `GET /valutazione` — renders `eval/results/latest.json` if present (see section 8).
- `GET /informazioni` — sources, method, limitations, privacy, accessibility statement.

JSON API (`/api/v1`, OpenAPI docs at `/api/docs`): `POST /api/v1/answer` (alias `POST /api/ask`)
`{"question": str, "date": "YYYY-MM-DD"?}` → `Answer`;
`GET /api/v1/search?q=&date=&k=&mode=` → `list[Hit]`; `GET /api/v1/units/{key:path}?date=` →
`list[Hit]`; `GET /healthz` → `{"status": "ok", "db": bool}`.

Rate limiting on the answering endpoints: per client (hashed IP with `client_hash_salt`) per
rolling hour and global per day, from settings (0 disables a limit); 429 with `Retry-After` and an
Italian message. `GET /api/v1/search` has its own per-client hourly budget
(`rate_limit_search_per_ip_per_hour`), `k <= 20`, and caches query embeddings (LRU, 256 entries).
Invalid requests use no quota. Limits are kept in process (they reset on restart and are per
worker). Clients are bucketed before hashing (IPv6 by /64); with `forwarded_allow_ips="*"` only
the rightmost `X-Forwarded-For` hop is trusted. Each
answer is logged to `qa_log` (best effort; a logging failure never breaks the response). Validation errors → 422 with Italian messages; `LLMError` → 503 with a generic message.

Accessibility (WCAG 2.1 AA): `lang="it"`, skip link, labelled controls, visible focus, sufficient
contrast, `aria-live="polite"` on the answer region, `hx-indicator` with a text alternative, no
information conveyed by color alone. Third-party assets pinned to exact versions.
Tests use `TestClient(create_app(settings, answer_fn=stub))`; DB-backed routes may use
`seeded_conn` and monkeypatch the retrieval calls.

## 8. evaluation

Package `appalti_qa.evaluation`; exports `EvalItem`, `load_dataset`, `validate_dataset`,
`retrieval_metrics`, `answer_metrics`, `Judge`, `JudgeVerdict`, `get_judge`, `export_xlsx`,
`import_xlsx`, `write_results`, `render_report`, `cli_main`.

Dataset format (`eval/dataset.yaml`, a YAML list of mappings):

```yaml
- id: Q001
  question: "..."            # Italian
  kind: answerable | temporal | unanswerable | multi
  category: affidamento_diretto   # e.g. soglie, procedure, rup, garanzie, subappalto, esecuzione,
                                  # digitalizzazione, principi, concessioni, contenzioso
  vigenza: "2026-10-11"      # date whose text in force the answer refers to (ISO)
  expected_answer: "..."     # 1-3 sentences, Italian
  expected_refs: ["50/1/b"]  # canonical keys; [] for unanswerable
  source_excerpt: "..."      # verbatim excerpt (<=300 chars) of the referenced text, for the human verifier
  difficulty: easy | medium | hard
  status: bozza              # bozza | verificata | corretta | scartata (set by the human reviewer)
  notes: ""
```

```python
# appalti_qa/evaluation/dataset.py
class EvalItem(BaseModel):      # fields exactly as above; vigenza: date; id matches r"^Q\d{3,}$"
    ...
class DatasetError(ValueError): problems: list[str]
def load_dataset(path: Path = Path("eval/dataset.yaml")) -> list[EvalItem]   # raises DatasetError
def validate_dataset(items: Sequence[EvalItem], *, conn: psycopg.Connection | None = None) -> list[str]
```

Static checks: unique ids, valid keys (`keys.is_valid_key`), `unanswerable` ⇔ empty refs,
`multi` has 2+ refs, excerpt length. With `conn`: each ref matches a unit in force at `vigenza`,
and `source_excerpt` occurs (whitespace-normalised) in one of the referenced texts.

```python
# appalti_qa/evaluation/metrics.py
def retrieval_metrics(items, retrieve: Callable[[str, date, int], list[Hit]], *,
                      k_values=(1, 3, 5, 10)) -> dict
    # over items with refs: recall@k (share of expected refs matched in top k, via ref_matches),
    # hit@k, MRR; overall, by kind and by category
def answer_metrics(items, answer: Callable[[str, date], Answer], judge: Judge) -> dict
    # correctness (judge, 0 / 0.5 / 1), citation precision and recall vs expected_refs,
    # abstention accuracy (unanswerable should abstain, others should not), latency p50/p95,
    # tokens, cost (known prices summed, cost_unknown counts answers without a price);
    # overall, by kind and by category; plus per-item records

# appalti_qa/evaluation/judge.py
@dataclass(frozen=True) class JudgeVerdict:
    score: float; rationale: str; input_tokens: int = 0; output_tokens: int = 0
    cost_usd: float | None = 0.0
class Judge(Protocol):
    name: str                   # "overlap" or "<provider>:<model>"
    def grade(self, item: EvalItem, answer: Answer) -> JudgeVerdict
class LLMJudge: ...             # LLMProvider.complete with an Italian rubric, JSON verdict
class OverlapJudge: ...         # deterministic token overlap; used when the judge resolves to fake
def get_judge(settings: Settings | None = None) -> Judge

# appalti_qa/evaluation/xlsx.py
def export_xlsx(items: Sequence[EvalItem], path: Path, *, answers: Mapping[str, Answer] | None = None,
                unit_texts: Mapping[str, str] | None = None) -> None
def import_xlsx(path: Path, dataset_path: Path) -> list[EvalItem]   # merge reviewer edits, rewrite YAML

# appalti_qa/evaluation/report.py
def write_results(results: dict, out_dir: Path = Path("eval/results"), *, latest: bool = True) -> Path
def render_report(results: dict) -> str                             # markdown
def cli_main(argv: Sequence[str] | None = None) -> int
```

CLI (`appalti-qa eval <sub>`): `validate [--db]`, `retrieval [--mode] [--k] [--publish]`,
`answers` (alias `run`) `[--provider] [--model] [--mode] [--k] [--limit] [--ids Q001,...]
[--publish]`, `report RESULTS.json`, `export-xlsx OUT.xlsx [--with-answers RESULTS.json]`,
`import-xlsx IN.xlsx [--db] [--dry-run]`, `compare RESULTS.json...`. Runs use only `verificata` and
`corretta` questions unless `--all` is given. A run updates `latest.json` (the published results)
only with `--publish`; `answers --publish` is refused when the answering provider or the judge
resolves to fake, so placeholder numbers are never published. In an answer run each question is
answered through `answer_question` with the same `search(question, date, k, mode)` call as the
app; the retrieval metrics come from separate calls at the largest cutoff.
The xlsx is for a human reviewer: Italian column headers, wrapped text, frozen header row, an
`Esito` dropdown (`OK`, `Da correggere`, `Scartare`) and a `Correzione` column; on import, edited
rows become `corretta`, `OK` rows `verificata` and `Scartare` rows `scartata` (kept, never
evaluated). Every metric group carries an `n` count, and a metric is `null` when it has no data.

Results JSON (`eval/results/<UTC timestamp>_<provider>_<model>.json`, copied to
`eval/results/latest.json` when published; the web page reads it):

```json
{
  "run_id": "20261011T120000Z_fake_fake",
  "created_at": "2026-10-11T12:00:00Z",
  "dataset": {"path": "eval/dataset.yaml", "size": 120, "verified": 80},
  "config": {"provider": "fake", "model": "fake", "retrieval_mode": "hybrid", "k": 8,
             "embedding_model": "intfloat/multilingual-e5-base", "judge": "overlap"},
  "retrieval": {"overall": {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}, "by_kind": {}, "by_category": {}},
  "answer": {"overall": {"correctness": 0.0, "citation_precision": 0.0, "citation_recall": 0.0,
             "abstention_accuracy": 0.0, "latency_p50_ms": 0, "latency_p95_ms": 0,
             "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_unknown": 0},
             "by_kind": {}, "by_category": {}},
  "items": [{"id": "Q001", "retrieved": ["50/1"], "cited": ["50/1"], "abstained": false,
             "score": 1.0, "rationale": "...", "latency_ms": 0}]
}
```

Tests use small in-memory datasets, `FakeProvider`/stub answer functions and `OverlapJudge`.
