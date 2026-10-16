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

<p align="center">
  <img src="docs/images/answer.png" width="640" alt="An answer to &quot;Fino a che importo posso affidare direttamente un servizio?&quot; at 17 October 2026: art. 50, comma 1, lett. b) allows direct award of services under 140,000 euro, with numbered citations linked to the cited commi, each with its validity and the act that amended it">
</p>

<p align="center">
  <img src="docs/images/versions.png" width="640" alt="Page of art. 108, comma 4: the text in force at the chosen date and a table of its four versions, from the original text of 1 April 2023 to the version introduced by L. 30 ottobre 2025, n. 164">
</p>

*Screenshots of a local run (17 October 2026). Above: an answer with its cited commi, each with
its validity and amending act. Below: a comma and its versions over time.*

## Features

- **Point-in-time text.** 20 Normattiva consolidated snapshots, one for each day the text
  changed from 2023-04-01 to 2026-08-26 (checked through 2026-10-11), are parsed into 2,814 units
  (commi, and the allegati) with 3,213 versions. Each version records its validity interval,
  dated to the first day Normattiva serves it, and, when Normattiva marks the change as an
  amendment, the act that made it. Article headings are dated too.
- **Hybrid retrieval in PostgreSQL.** Italian full-text search (`tsvector`) and local
  multilingual embeddings (`intfloat/multilingual-e5-base` on CPU, pgvector HNSW), fused with
  reciprocal rank fusion and filtered by date. Explicit references such as "art. 50, comma 1,
  lett. b)" are recognised and pinned first, and acronyms the code spells out ("DEC", "DGUE",
  "OEPV") are searched with its wording. Long commi and tables are also embedded as
  sub-passages (one per lettera, or overlapping windows); passage-level search is measured and
  available, but not the default, because it did not beat the baseline on held-out questions.
- **Reranking.** For answers, 40 candidates from lexical, unit-level and passage-level search are
  reordered by Claude Haiku 5.5, which reads them and puts the useful ones first (structured
  output, about $0.002 per question). On the 36 held-out questions it raises MRR from 0.797 to
  0.976-0.982 and puts an expected comma in the top 5 for 35 and 36 of them in two runs.
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
    A --> K["rerank<br/>Claude Haiku 5.5<br/>40 candidates"]
    K --> R
    R --> DB
    A --> L{"LLM provider"}
    L --> C["Claude<br/>native citations"]
    L --> G["Groq<br/>numbered markers"]
    L --> F["Fake<br/>offline, deterministic"]
    V["evaluation<br/>dataset · metrics · judge · report"] --> A
    V --> R
    V -. "eval/results/latest.json<br/>latest_retrieval.json" .-> W
```

| Package | Role |
| --- | --- |
| `appalti_qa.ingest` | download, parse and version the Normattiva snapshots; load PostgreSQL |
| `appalti_qa.retrieval` | embeddings, lexical / semantic / hybrid search, lookups, version history |
| `appalti_qa.llm` | Anthropic, Groq and fake providers behind one protocol; the reranker |
| `appalti_qa.answer` | the answer flow: validation, retrieval and reranking, generation, abstention |
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
   and the same wording. Where a flat export restarts the comma numbering from 1 inside an
   article (arts. 34, 41 and 62), the numbering of the structured exports is adopted when the
   texts match, so commi such as 41/13 exist from 2023-04-01.
2. **Dates each version by its snapshot's file date**, the first day the text was served. For
   example, D.L. 124/2023, published on 19 September 2023, changes allegato V.3 from 20
   September, and L. 34/2026, published on 23 March 2026, applies from 7 April 2026.
3. **Opens a new version only when the wording changes.** The previous version is closed
   (`valid_to` is exclusive). The new one is an *amendment*, credited to the latest amending
   act in the export's lifecycle by that date, only when the words that changed touch the text
   Normattiva marks as amended (`((...))`, deletion markers, repeal notes) or the comma is
   repealed. Other differences
   (punctuation, a different split into commi, a reworded repeal note, words an export dropped or
   duplicated away from any marked span) are *editorial* versions credited to no act; an export
   artefact that touches marked text still counts as an amendment (see Limitations). A new article heading alone is not a
   new version: the heading is stored with its own dates and the one in force at the query date
   is shown, searched and given to the model. The corpus has 2,593 original, 589 amended and 31
   editorial versions, and 25 rows where only the heading changed (8 articles).

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

**Retrieval.** Each question is searched at its own date, over the 73 questions with expected
references. Comma level means the retrieved unit matches an expected reference (`50/1/b` is found
by unit `50/1`); article level means the right article was retrieved. The first three modes use no
language model; the last is how answers retrieve their sources (see Reranking below).

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | Article recall@5 | Article recall@10 | Article MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lexical (Italian full-text) | 0.192 | 0.546 | 0.751 | 0.355 | 0.610 | 0.788 | 0.387 |
| Semantic (e5-base) | 0.534 | 0.799 | 0.899 | 0.684 | 0.884 | 0.931 | 0.783 |
| Hybrid (RRF, default for search) | 0.660 | 0.836 | 0.886 | 0.786 | 0.870 | 0.918 | 0.826 |
| **Reranked by Claude Haiku 5.5 (default for answers)** | 0.847 | 0.977 | 0.977 | 0.957 | 0.993 | 0.993 | 0.970 |

Run `20261011T212153Z_retrieval` (published), on the exactly dated corpus with the repaired
comma numbering and point-in-time headings: 73 questions, all drafts, k = 1, 3, 5, 10. The first
three rows are identical to the v1 run `20261011T175038Z_retrieval` except the hybrid article MRR
(0.827 before): the numbering repair added commi 41/9-41/15 and 62/8-62/18 before 2024-12-28 and
reordered some lower-ranked candidates. The reranked row is a further run of the configuration
measured in decision 9 (another, `20261011T205507Z_retrieval`, gave MRR 0.946 and recall@5
0.973). The full report, with results by kind and by topic, is in
[eval/results/latest_retrieval.md](eval/results/latest_retrieval.md). With hybrid search 71% of
questions get an expected comma in first place (hit@1) and 88% within the top 5 (hit@5); with
reranking, 92% and 100%. The weakest group is the multi-reference questions (recall@5 0.556
hybrid, 0.815 reranked, over 9 questions). Only 4 questions cite an article explicitly, so
reference pinning barely affects these numbers.

**Retrieval experiment: passage-level embeddings** ([decision 8](docs/decisions.md)). e5-base is
weak on long, list-structured commi: for "Fino a che importo posso affidare direttamente un
servizio?" the cosine rank of 50/1 (2,135 characters, lettere a-e) is 61, and fusion with that
noisy list keeps it out of the hybrid top 10 although lexical search finds it at rank 2-4. Long
units are therefore also embedded as sub-passages (intro, one passage per lettera with the
intro, overlapping windows of about 1,000 characters for long blocks and tables), and a version
can score its best passage. The method was fixed before looking at any number: questions with
an odd id are the development set (37), even ids the test set (36); 60 hybrid variants
(granularity, a CSLS-style hubness correction, lexical weight in RRF, fusion depth) were
compared on the development set only, a rule stated in advance chose one, and it would become
the default only if it beat the baseline on the test set (higher MRR, recall@5 not lower).

| Variant | Dev R@5 | Dev MRR | Test R@5 | Test MRR | All 73 R@5 | All 73 MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid, whole units, plain RRF (v1, default) | 0.820 | 0.775 | 0.852 | 0.797 | 0.836 | 0.786 |
| Hybrid, passages, hubness 0.5, lexical weight 0.5, depth 8k (chosen on dev) | 0.901 | 0.816 | 0.861 | 0.748 | 0.881 | 0.782 |
| Semantic, whole units | 0.793 | 0.666 | 0.806 | 0.701 | 0.799 | 0.684 |
| Semantic, passages | 0.874 | 0.708 | 0.819 | 0.726 | 0.847 | 0.717 |

**The chosen variant does not beat the baseline on the test set, so the default is unchanged.**
Passages fix the motivating cases (50/1 for the example query moves from outside the hybrid top
10 to rank 2; Q021, "Fino a che importo posso affidare direttamente un lavoro di
manutenzione?", from outside the top 10 to rank 5) and raise recall@10 on all 73 questions from
0.886 to 0.927 (multi-reference recall@5 from 0.556 to 0.704), but they also lift other
list-structured units, and on the test set the right comma more often drops from first to
second or third (6 questions worse, 2 better). With 37 questions the development set was too
small to choose among 60 variants without overfitting. The full grid is in
[eval/results/sweeps](eval/results/sweeps), and `appalti-qa eval sweep --all --split dev` (or
`test`) reproduces it.

<p align="center">
  <img src="docs/images/evaluation-retrieval.png" width="720" alt="Evaluation page, retrieval by search mode over 73 questions: lexical recall@5 54.6% and MRR 0.355, semantic 79.9% and 0.684, hybrid 83.6% and 0.786, hybrid reranked by a model 97.7% and 0.957">
</p>

**Reranking** ([decision 9](docs/decisions.md)). For answers, a wider pool of 40 candidates
(lexical, unit-level and passage-level lists fused by RRF) is reordered by Claude Haiku 5.5, which
reads each candidate and returns the useful ones, most useful first. One configuration was fixed
and committed before any measurement, with the same adoption rule as above, and measured in two
runs because the model is not deterministic. It holds in both, so reranking is the default for
answers (`GET /api/v1/search` stays plain hybrid search, which costs nothing).

| Retrieval | Dev R@5 | Dev MRR | Test R@5 | Test MRR | All 73 R@1 | All 73 R@5 | All 73 MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid (no model) | 0.820 | 0.775 | 0.852 | 0.797 | 0.660 | 0.836 | 0.786 |
| **Reranked, run 1** | 0.964 | 0.919 | 0.954 | 0.976 | 0.833 | 0.959 | 0.947 |
| **Reranked, run 2** | 0.973 | 0.926 | 0.972 | 0.982 | 0.847 | 0.973 | 0.953 |

20 questions improve, 3 move from first to second place, and in a spot check the example query
above ranks 50/1 first. Multi-reference questions remain the hardest (recall@5 0.556 → 0.778).
Each call reads about 16,000 tokens: $0.0018 per question at list prices. The test set had
already been used for decision 8, so new questions are needed to confirm the size of the gain.
Results are in [eval/results/rerank](eval/results/rerank).

**Answers.** Claude Opus 5.5 (effort medium) answers all 81 questions from 8 reranked sources;
Claude Opus 5.5 also grades each answer against the expected one with an Italian rubric
(`correct` 1, `partial` 0.5, `wrong` 0).

| Kind | n | Correctness | Citation precision | Citation recall | Abstention accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Answerable | 52 | 0.971 | 0.522 | 1.000 | 1.000 |
| Temporal | 12 | 0.958 | 0.583 | 1.000 | 1.000 |
| Unanswerable | 8 | 1.000 | - | - | 0.625 |
| Multi-reference | 9 | 0.833 | 0.576 | 0.778 | 1.000 |
| **All** | **81** | **0.957** | **0.538** | **0.973** | **0.963** |

Run `20261011T205828Z_anthropic_claude-opus-5-5` (published, all 81 questions, drafts). No answer
is graded wrong; 7 are partial, each omitting a condition the expected answer includes (for
example the 50% cap on contract changes in Q078). Reading the numbers:

- *Citation precision* counts a citation as right only when it matches an expected reference.
  The expected lists hold the minimum needed, and the model also cites related commi (for the
  direct-award question, 50/1 and the 62/6 rule on maintenance works), so the low figure
  measures extra citations more than wrong ones.
- *Abstention*: 3 of the 8 unanswerable questions get a partial answer instead of "the sources do
  not allow an answer". Each says what the Code does not cover (the regional platform, who signs
  the determina, the criminal penalty) and quotes only what it does, as the prompt asks; the
  judge rates all three correct, but they are not abstentions.
- *Self-grading*: the judge is the same model as the answerer, and the expected answers are
  drafts, so the expert review may move these numbers.
- *Latency* (retrieval, reranking and generation): median 7.8 s, 95th percentile 13.5 s.
- *Cost* at list prices: $3.09 for the answers including reranking ($0.038 per answer), $1.01
  for the judge.

Answers retrieve their sources exactly as the app does; the retrieval metrics of the run come
from separate calls. Without a key the fake provider and a word-overlap judge produce placeholder
numbers only, and `--publish` refuses to publish them. Costs use list prices; an answer that
the API served through a fallback model is charged for every attempt at its own model's rates,
and an answer involving a model without a known price is counted apart instead of hiding the
total.

Every run is written to `eval/results/` as JSON and markdown. Only a run made with `--publish`
is published: an answer run replaces `latest.json`, the one shown on the app's *Valutazione*
page; a retrieval run replaces `latest_retrieval.json` and also `latest.json` unless that holds
an answer run, so a retrieval run never hides published answer numbers (the page then takes the
comparison of the search modes from `latest_retrieval.json`). Publishing is refused for fake
providers, judges or embedder, and for semantic or hybrid runs on a corpus not fully embedded
with the configured model.

## Quick start

Requirements: Docker. For local development, also Python 3.12. About 2 GB of disk are needed for
the embedding model and the CPU build of PyTorch.

### Docker Compose

```bash
cp .env.example .env                              # optional; defaults run offline
docker compose up -d --build                      # db (127.0.0.1:5433) + app (localhost:8000)
docker compose exec app appalti-qa download --refine  # exactly dated snapshots -> ./data/raw/akn
docker compose exec app appalti-qa ingest         # parse, version, load
docker compose exec app appalti-qa embed          # embeddings on CPU (~25-35 min the first time)
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
.venv/bin/appalti-qa eval sweep --all --split dev  # retrieval variants on the development split
.venv/bin/appalti-qa eval answers --all            # fake provider unless a key is set
# add --publish to update eval/results/latest.json (refused for fake providers, judges or
# embedder, or a corpus not fully embedded); retrieval --publish also writes latest_retrieval.json

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
| `FUSION_DEPTH` / `LEXICAL_WEIGHT` | `4` / `1.0` | hybrid fuses two lists of `FUSION_DEPTH * k`; weight of the lexical list in RRF |
| `SEMANTIC_GRANULARITY` / `HUBNESS_WEIGHT` | `unit` / `0.0` | `passage`: a version scores its best sub-passage, less `HUBNESS_WEIGHT` times its hub score (see decision 8) |
| `RERANK_PROVIDER` / `RERANK_MODEL` | `anthropic` / `claude-haiku-5-5` | answers rerank their sources with this model when `ANTHROPIC_API_KEY` is set (decision 9); `none` turns it off; search alone is never reranked |
| `RERANK_POOL` / `RERANK_MAX` | `40` / `15` | candidates the reranker reads (lexical, unit and passage lists fused) and how many it may put first |
| `MIN_RELEVANCE_SCORE` | `0.0` | abstain without calling the model when the best source scores below this (not calibrated yet) |
| `MAX_QUESTION_CHARS` | `500` | question length limit |
| `RATE_LIMIT_PER_IP_PER_HOUR` / `RATE_LIMIT_GLOBAL_PER_DAY` | `20` / `500` | answering limits (0 disables) |
| `RATE_LIMIT_SEARCH_PER_IP_PER_HOUR` | `120` | per-client limit of `GET /api/v1/search` (0 disables); `k` is capped at 20 |
| `CLIENT_HASH_SALT` | random per process | salt for hashing client addresses in the usage log |
| `FORWARDED_ALLOW_IPS` | empty | reverse proxies trusted for `X-Forwarded-For` (`*`: only the rightmost hop is used) |
| `REAL_IP_HEADER` | empty | header in which the outermost trusted proxy names the client, read only when every hop is trusted (behind Cloudflare: `CF-Connecting-IP`) |

## Limitations

- **Not legal advice.** This is an information tool. The official text on
  [Normattiva](https://www.normattiva.it) and the Gazzetta Ufficiale prevails, and answers do not
  replace a professional's judgement.
- **Scope.** D.Lgs. 36/2023 with its allegati and amendments only. It does not cover ANAC
  guidelines and resolutions, case law, regional laws, ministerial decrees, or other acts the Code
  refers to.
- **Granularity.** A change is dated to the first day Normattiva serves it and credited per
  date, not per comma: when several acts take effect on the same day, the latest published one is
  credited (the doubt is logged). Allegati without articles are split into blocks in document
  order (`p1`, `p2`, ...), which may not match their printed numbering.
- **Amendment or editorial.** The distinction rests on Normattiva's markers: a change an act made
  without a marker would be editorial, and marked text that differs only between the 2024 flat
  and the 2025 structured exports counts as an amendment (11 units on 2025-03-15 are credited to
  D.L. 25/2025, several of them likely of this kind, e.g. a duplicated word removed next to a
  marked span in all. I.7, 40/2; see
  [docs/modules/ingest.md](docs/modules/ingest.md)). Defects of a single export, such as a
  sentence of art. 125, comma 1, missing from the 2024-12-31 text, are not repaired.
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
  retrieval/     embeddings, passages, search, reference extraction, Normattiva links
  llm/           Anthropic, Groq and fake providers, prompts
  answer/        answer orchestration and abstention
  web/           FastAPI app, Jinja2 templates, static assets
  evaluation/    dataset, metrics, judges, reports, xlsx review, eval CLI
sql/schema.sql   tables, exclusion constraint on validity intervals, GIN and HNSW indexes
eval/            dataset.yaml (questions), README_dataset.md, results/ (sweeps/ for retrieval variants), review/ (expert workbook, not committed)
docs/            architecture (module contract), decisions, per-module notes
tests/           pytest suite per module; DB tests run on a throwaway database
```
