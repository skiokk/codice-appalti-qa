# codice-appalti-qa

Question answering over the Italian Public Procurement Code, D.Lgs. 31 marzo 2023, n. 36
(*Codice dei contratti pubblici*). Every answer cites the article and comma it relies on, refers to
the text **in force at a date the user chooses**, and the system's quality is measured on a
dataset of Italian questions.

## Why

RUPs, buyers in health authorities and municipalities, and the companies bidding for their
contracts need precise answers on a Code that keeps changing. Since 2023 it has been amended about
twenty times, from D.L. 51/2023 through the 2024 *correttivo* (D.Lgs. 209/2024) up to
D.Lgs. 149/2026. A procedure started in 2024 is governed by the 2024 text, not today's. A general
chatbot paraphrases from memory, cites nothing verifiable and does not know which version applied
when.

This project answers only from the official consolidated text, at the requested date, with the
cited passage shown next to the answer and a link to Normattiva. It is built to abstain when the
sources do not answer the question. It also publishes how often it gets things right.

## Features

- **Point-in-time text.** 20 Normattiva consolidated snapshots, one for each day the text
  changed from 2023-04-01 to 2026-08-26 (checked through 2026-10-11), are parsed into 2,814 units
  (commi, and the allegati) with 3,215 versions. Each version records its validity interval,
  dated to the first day Normattiva serves it, and the act that amended it.
- **Hybrid retrieval in PostgreSQL.** Italian full-text search (`tsvector`) and local
  multilingual embeddings (`intfloat/multilingual-e5-base` on CPU, pgvector HNSW), fused with
  reciprocal rank fusion and filtered by date. Explicit references such as "art. 50, comma 1,
  lett. b)" are recognised and pinned first.
- **Cited answers.** Claude through the Anthropic Messages API with native citations: each comma
  is a document, and the API returns the exact quoted span. Open-weight models on Groq can be
  compared, and a deterministic fake provider runs everything offline with no API key.
- **Abstention.** When retrieval returns nothing, when the model states that the sources do not
  answer (`NON_TROVATO`), or when no citation points at a source, the answer is "the sources do
  not allow an answer" instead of a guess. Hybrid and semantic search return k sources once
  the corpus is embedded, so in practice the decision rests with the model. A relevance threshold
  (`MIN_RELEVANCE_SCORE`) can make the system abstain before calling it, but it is not calibrated
  yet (default 0, off); it will be calibrated on the unanswerable questions once they are
  verified.
- **Measured.** A dataset of 81 Italian questions (answerable, temporal, unanswerable,
  multi-reference) with expected canonical references. It measures retrieval recall and MRR,
  citation precision and recall, abstention accuracy and answer correctness (LLM judge), with an
  xlsx round trip for human review.
- **Built for the public sector.** Bootstrap Italia, WCAG 2.1 AA, works without JavaScript, a
  strict Content Security Policy, rate limits, and a usage log that never stores raw IP
  addresses.

## Architecture

```mermaid
flowchart LR
    N[("Normattiva<br/>Akoma Ntoso XML<br/>one snapshot per amendment")] --> I["ingest<br/>parse · version · load"]
    I --> DB[("PostgreSQL 16<br/>units · unit_versions · events<br/>tsvector (italian) + pgvector")]
    E["embed<br/>multilingual-e5-base (CPU)"] --> DB
    U(["User:<br/>question + date"]) --> W["web<br/>FastAPI · htmx · Bootstrap Italia"]
    W --> A["answer<br/>validate · retrieve · generate · abstain"]
    A --> R["retrieval<br/>lexical + semantic → RRF<br/>in force at the date"]
    R --> DB
    A --> L{"LLM provider"}
    L --> C["Claude<br/>native citations"]
    L --> G["Groq<br/>numbered markers"]
    L --> F["Fake<br/>offline, deterministic"]
    V["evaluation<br/>dataset · metrics · judge · report"] --> A
    V --> R
    V -. "eval/results/latest.json" .-> W
```

| Package | Role |
| --- | --- |
| `appalti_qa.ingest` | download, parse and version the Normattiva snapshots; load PostgreSQL |
| `appalti_qa.retrieval` | embeddings, lexical / semantic / hybrid search, lookups, version history |
| `appalti_qa.llm` | Anthropic, Groq and fake providers behind one protocol |
| `appalti_qa.answer` | the answer flow: validation, retrieval, generation, abstention |
| `appalti_qa.web` | Italian web interface and JSON API |
| `appalti_qa.evaluation` | dataset checks, metrics, judges, reports, xlsx review |

The module contract is in [docs/architecture.md](docs/architecture.md), the design decisions in
[docs/decisions.md](docs/decisions.md), and each module has a page in [docs/modules](docs/modules).

## How point-in-time versions work

Normattiva publishes the consolidated text of the Code as Akoma Ntoso XML for any *dataVigenza*.
The downloader reads the amendment dates from the act's lifecycle and fetches the text in force on
each of them. Those are publication dates, and most amendments take effect later (a decree-law the
next day, a law after its *vacatio legis*), while the expression date inside an export is stale
in several of them. So `appalti-qa download --refine` compares consecutive exports on their
content and, where they differ, fetches intermediate days by bisection until it knows the first
day of each new text; only those boundary snapshots are kept, named by that day. Ingestion then:

1. **Parses each snapshot into units with canonical keys**: `50/1` (art. 50, comma 1), `225-bis`
   (an article without commi), `all:I.1/2/1` (allegato I.1, art. 2, comma 1), `all:I.3/p3` (the
   third numbered block of an allegato without articles). A reference to a lettera (`50/1/b`)
   matches its comma. Exports up to 2024 are flat text, so commi and lettere are found in running
   text with numbering rules. Exports from 2025 are structured. Both shapes produce the same keys
   and the same wording.
2. **Dates each version by its snapshot's file date**, the first day the text was served. For
   example, D.L. 124/2023, published on 19 September 2023, changes allegato V.3 from 20
   September, and L. 34/2026, published on 23 March 2026, applies from 7 April 2026.
3. **Opens a new version only when the wording changes.** The previous version is closed
   (`valid_to` is exclusive) and the new one is credited to the amending act that introduced it.
   A change of punctuation or capitalisation only, or a different split of the same words into
   commi, is an *editorial* version credited to no act; a new article heading alone updates the
   heading without a new version. The corpus has 2,575 original, 630 amended and 10 editorial
   versions.

A query at date *D* only sees versions with `valid_from <= D < valid_to`. The interface shows each
source's validity interval, the acts that modified it, and its full version history.

```text
60/2 (revisione prezzi)  2023-04-01 → 2024-12-31   original text
                         2024-12-31 → (open)       D.Lgs. 31 dicembre 2024, n. 209
```

## Evaluation

[`eval/dataset.yaml`](eval/dataset.yaml) holds 81 Italian questions written the way a RUP or a
bidder would ask them: 52 answerable, 12 temporal (pairs of the same question at two dates), 8
unanswerable (regional rules, tax or criminal law, figures set elsewhere) and 9 that need several
provisions. Each question has its date, an expected answer, the expected canonical references and
a verbatim excerpt. `appalti-qa eval validate --db` checks that every reference exists on its date
and every excerpt occurs in the referenced text. All 81 pass, and each temporal pair falls on two
different versions of the exactly dated text.

> **All 81 questions are drafts (`bozza`) pending verification by a domain expert.** A machine
> check does not show that an answer is legally right, so the numbers below are indicative.
> `eval export-xlsx` and `eval import-xlsx` handle the review round trip; the workbook for the
> expert is written to `eval/review/` (not committed).

**Retrieval** (no LLM). Each question is searched at its own date, over the 73 questions with
expected references. Comma level means the retrieved unit matches an expected reference
(`50/1/b` is found by unit `50/1`); article level means the right article was retrieved.

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical (Italian full-text) | 0.192 | 0.546 | 0.751 | 0.355 | 0.610 | 0.788 | 0.387 |
| Semantic (e5-base) | 0.534 | 0.799 | 0.899 | 0.684 | 0.884 | 0.931 | 0.783 |
| **Hybrid (RRF, default)** | 0.660 | 0.836 | 0.886 | 0.786 | 0.870 | 0.918 | 0.827 |

Run `20261011T165405Z_retrieval`, on the exactly dated corpus: 73 questions, all drafts,
k = 1, 3, 5, 10. Dating the changes exactly moved no question across a version, so the numbers
are the same as on the earlier corpus. The full report,
with results by kind and by topic, is in [eval/results/latest.md](eval/results/latest.md). With
hybrid search, 71% of questions get an expected comma in first place (hit@1) and 88% within the
top 5 (hit@5). Semantic search alone has the best recall@10. The weakest group is the
multi-reference questions (hybrid recall@5 0.556 over 9 questions). Only 4 questions cite an
article explicitly, so reference pinning barely affects these numbers.

**Answers** (correctness, citation precision and recall, abstention accuracy, latency, cost):
*pending: run with an API key*
(`appalti-qa eval answers --provider anthropic --judge-provider anthropic --all --publish`).
Without a key the fake provider and a word-overlap judge produce placeholder numbers only, and
`--publish` refuses to publish them. Answers retrieve their sources exactly as the app does; the
retrieval metrics of the run come from separate calls. Costs use list prices; an answer served by
a model without a known price is counted apart instead of hiding the total.

Every run is written to `eval/results/` as JSON and markdown. Only a run made with `--publish`
replaces `latest.json`, the one shown on the app's *Valutazione* page.

## Quick start

Requirements: Docker. For local development, also Python 3.12. About 2 GB of disk are needed for
the embedding model and the CPU build of PyTorch.

### Docker Compose

```bash
cp .env.example .env                              # optional; defaults run offline
docker compose up -d --build                      # db (127.0.0.1:5433) + app (localhost:8000)
docker compose exec app appalti-qa download --refine  # exactly dated snapshots -> ./data/raw/akn
docker compose exec app appalti-qa ingest         # parse, version, load
docker compose exec app appalti-qa embed          # embeddings on CPU (~15-25 min the first time)
```

Open <http://localhost:8000>.

### Local development

```bash
docker compose up -d db
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"

.venv/bin/appalti-qa download --refine   # polite, cached; --refine dates every change exactly
.venv/bin/appalti-qa ingest
.venv/bin/appalti-qa embed
.venv/bin/appalti-qa serve               # http://127.0.0.1:8000

.venv/bin/appalti-qa eval validate --db
.venv/bin/appalti-qa eval retrieval --all          # lexical, semantic and hybrid
.venv/bin/appalti-qa eval answers --all            # fake provider unless a key is set
# add --publish to update eval/results/latest.json (refused for fake providers and judges)

.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest                         # DB tests use a throwaway database on the compose server
```

To answer with Claude, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env`. For Groq,
set `LLM_PROVIDER=groq` and `GROQ_API_KEY`. A provider without a key falls back to the fake
provider, and the interface says so.

### JSON API

```bash
curl -s localhost:8000/api/v1/answer -H 'Content-Type: application/json' \
  -d '{"question": "Quali sono le soglie di rilevanza europea per i lavori?", "date": "2026-10-11"}'
curl -s 'localhost:8000/api/v1/search?q=revisione%20prezzi&date=2024-06-01&k=5'
curl -s 'localhost:8000/api/v1/units/50/1/b?date=2026-10-11'
```

OpenAPI documentation is served at `/api/docs`.

## Configuration

Environment variables, also read from `.env` (see [.env.example](.env.example)):

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://appalti:appalti@localhost:5433/appalti` | PostgreSQL with pgvector |
| `POSTGRES_PASSWORD` | `appalti` | database password used by Docker Compose (local development only; must match `DATABASE_URL`) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | sentence-transformers model; `fake` for offline tests |
| `EMBEDDING_DIM` | `768` | must match the model; re-create the schema and re-embed when it changes |
| `LLM_PROVIDER` | `fake` | `fake`, `anthropic` or `groq` |
| `LLM_MODEL` / `LLM_EFFORT` | `claude-opus-5-5` / `medium` | Claude model and effort |
| `LLM_MAX_TOKENS` | `16000` | output ceiling (reasoning tokens count too) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | comparison model on Groq |
| `ANTHROPIC_API_KEY`, `GROQ_API_KEY` | empty | provider keys |
| `JUDGE_PROVIDER` / `JUDGE_MODEL` | `fake` / `claude-opus-5-5` | evaluation judge (`fake` = word overlap) |
| `RETRIEVAL_MODE` | `hybrid` | `lexical`, `semantic` or `hybrid` |
| `DEFAULT_TOP_K` / `RRF_K` | `8` / `60` | sources per answer, RRF constant |
| `MIN_RELEVANCE_SCORE` | `0.0` | abstain without calling the model when the best source scores below this (not calibrated yet) |
| `MAX_QUESTION_CHARS` | `500` | question length limit |
| `RATE_LIMIT_PER_IP_PER_HOUR` / `RATE_LIMIT_GLOBAL_PER_DAY` | `20` / `500` | answering limits (0 disables) |
| `RATE_LIMIT_SEARCH_PER_IP_PER_HOUR` | `120` | per-client limit of `GET /api/v1/search` (0 disables); `k` is capped at 20 |
| `CLIENT_HASH_SALT` | random per process | salt for hashing client addresses in the usage log |
| `FORWARDED_ALLOW_IPS` | empty | reverse proxies trusted for `X-Forwarded-For` (`*`: only the rightmost hop is used) |

## Limitations

- **Not legal advice.** This is an information tool. The official text on
  [Normattiva](https://www.normattiva.it) and the Gazzetta Ufficiale prevails, and answers do not
  replace a professional's judgement.
- **Scope.** D.Lgs. 36/2023 with its allegati and amendments only. It does not cover ANAC
  guidelines and resolutions, case law, regional laws, ministerial decrees, or other acts the Code
  refers to.
- **Granularity.** A change is dated to the first day Normattiva serves it and credited per
  date, not per comma: when several acts take effect on the same day, the latest published one is
  credited (the doubt is logged), and a few changes that only re-lay out a table or a repeal note
  count as amendments. A change of an article heading alone is not versioned. Allegati without
  articles are split into blocks in document order (`p1`, `p2`, ...), which may not match their
  printed numbering.
- **Evaluation.** The questions are drafts awaiting expert review, and answer quality has not yet
  been measured with a real model.
- **Abstention** depends on the model: the relevance threshold that would let retrieval alone
  decline a question is not calibrated yet.
- **Rate limits** (answers, and search on the JSON API) are kept in process: they reset on
  restart and apply per worker.

## Data source and licensing

The normative text comes from [Normattiva](https://www.normattiva.it), the official Italian
legislation portal, as Akoma Ntoso 3.0 XML (OASIS LegalDocML). Official legislative texts are not
covered by copyright (art. 5, Legge 22 aprile 1941, n. 633). The raw XML is not committed:
`appalti-qa download` fetches it politely, one request at a time, and caches it in
`data/raw/akn/`. The consolidated text published on Normattiva is not the official version: only
the Gazzetta Ufficiale is authoritative.

The code is released under the MIT License, see [LICENSE](LICENSE).

## Project layout

```text
src/appalti_qa/
  config.py, db.py, keys.py, models.py, cli.py   settings, DB access, canonical keys, shared models, CLI
  ingest/        Normattiva download, Akoma Ntoso parsing, versioning, loading
  retrieval/     embeddings, search, reference extraction, Normattiva links
  llm/           Anthropic, Groq and fake providers, prompts
  answer/        answer orchestration and abstention
  web/           FastAPI app, Jinja2 templates, static assets
  evaluation/    dataset, metrics, judges, reports, xlsx review, eval CLI
sql/schema.sql   tables, exclusion constraint on validity intervals, GIN and HNSW indexes
eval/            dataset.yaml (questions), README_dataset.md, results/, review/ (expert workbook, not committed)
docs/            architecture (module contract), decisions, per-module notes
tests/           pytest suite per module; DB tests run on a throwaway database
```
