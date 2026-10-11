# Decision records

Short records of the choices that shape the project. Each states the context, the decision and
its consequences.

## 1. Source: Akoma Ntoso snapshots from Normattiva

**Context.** Answers must quote the exact text of D.Lgs. 36/2023 *as in force on a given date*. The
code has been amended many times since 2023 (D.L. 51/2023, the 2024 correttivo D.Lgs. 209/2024,
and later acts), so a single "current" text cannot answer questions about past procedures.

**Decision.** Use the consolidated Akoma Ntoso 3.0 XML exported by Normattiva, one snapshot per
amendment date, and derive point-in-time versions of every comma by diffing consecutive snapshots.

**Consequences.** Official, structured text with stable element ids (`art_50__para_1`) and the list
of amending acts in the metadata; no PDF or HTML scraping. Versions are only as fine-grained as the
snapshots: a change takes effect on the snapshot's effective date. Raw files stay out of git and
are re-downloadable with a polite, cached downloader.

## 2. Storage and search: PostgreSQL 16 + pgvector

**Context.** We need relational point-in-time queries (validity intervals, amendment events),
Italian full-text search and vector search, filtered by date, in a footprint a public
administration can run on one server.

**Decision.** A single PostgreSQL instance with the `italian` text-search configuration (generated
`tsvector`, GIN index) and pgvector (HNSW, cosine). Validity intervals are enforced by an exclusion
constraint.

**Consequences.** One system to operate, back up and secure; hybrid retrieval is a SQL join rather
than a sync between stores. The corpus (a few thousand commi times their versions) is far below
the scale where a dedicated vector database would pay off.

## 3. Embeddings: multilingual-e5-base, local, on CPU

**Context.** Questions and text are Italian. Sending legal text and user questions to a third-party
embedding API adds cost and a data flow to justify; the deployment target has no GPU and limited
disk.

**Decision.** `intfloat/multilingual-e5-base` (768 dimensions, about 1.1 GB) through
sentence-transformers on CPU, with the `query: ` / `passage: ` prefixes and normalised vectors.
The model name and dimension are settings: `BAAI/bge-m3` (1024 dimensions) is the configured
upgrade path when hardware allows (re-create the schema with the new dimension and re-embed).

**Consequences.** Data stays on the server, embedding cost is zero and the corpus embeds once
offline. Unchanged texts are embedded once (deduplicated by hash). Tests use a deterministic fake
embedder so CI never downloads a model.

## 4. Interface: Bootstrap Italia and WCAG 2.1 AA

**Context.** The intended users are public-sector staff (RUP, buyers, legal offices). Italian
public administrations follow the AgID design guidelines and must meet accessibility requirements
(Legge 4/2004, WCAG 2.1 AA).

**Decision.** Server-rendered FastAPI + Jinja2 pages with htmx and Bootstrap Italia, the design
system of Designers Italia, targeting WCAG 2.1 AA.

**Consequences.** The interface looks and behaves like other PA services, works without a
JavaScript build chain, and degrades to plain form posts. Accessibility is a requirement checked in
review, not a later polish.

## 5. Generation: Claude with native citations, Groq models for comparison

**Context.** A legal answer is useful only if every claim can be traced to the text. Free-form
"[1]" markers produced by a model can point to the wrong source or quote text that is not there.

**Decision.** The primary provider is Claude (`claude-opus-5-5`) through the Anthropic Messages API
with native citations: each retrieved comma is a document, and the API returns the exact cited span
and its document index. Open-weight models served by Groq (default `openai/gpt-oss-120b`) are a
second provider, with numbered-source markers parsed into citations, to compare quality, latency
and cost. A deterministic fake provider is the default so the system runs and is tested without
keys. The model abstains when the sources do not support an answer.

**Consequences.** Citations from Claude are verifiable by construction; Groq citations are best
effort and the evaluation measures the difference. Providers sit behind one protocol, so adding or
swapping one does not touch retrieval, the UI or the evaluation.

## 6. Evaluation is part of the product

**Context.** A question-answering system on legislation that is not measured cannot be trusted or
improved, and claims like "it answers correctly" are meaningless without a dataset.

**Decision.** Ship a versioned dataset of Italian questions (answerable, temporal, unanswerable,
multi-reference) with expected canonical references, verified by a human through an xlsx round
trip; measure retrieval (recall@k, MRR), citations (precision/recall), abstention and answer
correctness (LLM judge with a deterministic fallback); publish results as JSON, a markdown report
and a page in the app.

**Consequences.** Retrieval and prompt changes are compared on numbers. The dataset's quality is
explicit (`bozza` / `verificata` / `corretta`), and the results page shows users the system's
measured limits.

## 7. Versions take effect on the expression date, not the download date

**Context.** The lifecycle of a Normattiva export lists the amending acts by *publication* date,
and the downloader requests the text in force on each of those dates. Most amendments take effect
later (a decree-law the next day, a law after a 15-day vacatio legis), so the export requested on
a publication date often still shows the previous text, and the change appears only in the next
export. Dating versions by file name would put changes in force weeks early or late (for example,
L. 34/2026, published on 23 March 2026, applies from 7 April 2026).

**Decision.** Each export's FRBRExpression date records when its consolidated text came into
force. When that date falls after the previous file's date and before the file's own date, it is
the version's `valid_from`; otherwise the file date is used. A version is credited to the latest
amending act in the lifecycle of the export that introduced it.

**Consequences.** Point-in-time answers follow Normattiva's own dating. When several acts take
effect between two exports, all changes are credited to the latest act. Crediting individual commi
would need Normattiva's per-article update notes.
