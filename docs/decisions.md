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

## 7. Versions take effect on the first day Normattiva serves them, and are judged on its markers

**Context.** The lifecycle of a Normattiva export lists the amending acts by *publication* date,
and the downloader requests the text in force on each of those dates. Most amendments take effect
later (a decree-law the next day, a law after a 15-day vacatio legis), so the export requested on
a publication date often still shows the previous text. The FRBRExpression date of an export
should say when its text came into force, but it is stale in several exports: 20230919 and
20230929 both say 2023-07-06, so the changes of D.L. 124/2023, in force on 20 September 2023,
would be dated 29 September; 20250401 still says 2024-12-31, so the changes of D.L. 25/2025
would be dated 1 April 2025 instead of 15 March. Point-in-time answers in those gaps quoted the
previous text.

**Decision.** Dates are discovered, not read from metadata. `appalti-qa download --refine`
compares consecutive exports on their normalised content (unit keys, headings and wording,
ignoring layout and metadata) and, where it differs, downloads the text in force on intermediate
days by bisection until the first day of each new content is known; a probe that differs from
both ends is an intermediate version and both halves are searched. Only boundary snapshots are
kept, named by that first day, and `refined.json` records what has been verified so later runs
do not probe again. Versioning uses the file date as `valid_from` and only logs disagreements
with the FRBRExpression date.

What changed is judged on Normattiva's own evidence. A change of words counts as an `amendment`
only when it touches the text the export marks as amended (`((...))` inside `<ins>`, deletion
markers, repeal notes) or the unit becomes repealed; it is credited to the amending act whose
lifecycle event is the latest one dated on or before `valid_from` in the lifecycle of the export
that introduced it. Any other difference (punctuation or capitalisation, a different split into
commi, a reworded repeal note, words an export dropped or duplicated, table cells in another
order) is an `editorial` version (`unit_versions.change_kind`) credited to no act. A change of the
article heading alone is not a new version of the text: it opens a `heading` row with the same
text and the new heading, credited only when the new heading is marked, so headings, full-text
vectors and embeddings are those in force at every date. The comma numbering of the flat exports,
which restart it from 1 in a few articles, is taken from the structured exports when the texts
match (word-sequence similarity of at least 0.5), and kept as exported otherwise.

**Consequences.** Every version is dated to the day Normattiva starts serving it (20 boundaries
from 2023-04-01 to 2026-08-26 as of 2026-10-11, found with 84 probes), at the cost of a few dozen
requests (2 seconds apart) when new amendments appear; a probe cap bounds a run and the gaps it
leaves are logged, recorded and resumed by the next run. The search assumes the text
does not change and change back between two equal probes. On the corpus the rules give 2,593
original, 589 amended and 31 editorial versions and 25 heading rows; the earlier rules, which
credited any change of words and repaired the numbering of art. 34 only, gave 2,575, 630 and 10
(details in `docs/modules/ingest.md`). Commi 41/9-41/15 and 62/8-62/18 now exist from
2023-04-01 instead of appearing on 2024-12-28 as 20 amendments of D.L. 201/2024. The marker rule
is only as reliable as the markers: an unmarked change made by an act is editorial, and marked
text that differs only between the flat and the structured export (11 units on 2025-03-15,
credited to D.L. 25/2025, several of them likely of this kind) is an amendment. Crediting is still
per change date, not per comma: when two acts take effect on the same day, or an act with a
vacatio legis takes effect after another act published later, the latest-published act is
credited and the doubt is logged; crediting individual commi would need Normattiva's per-article
update notes.

## 8. Retrieval is tuned on a fixed development split and reported on a held-out test split

**Context.** With e5-base, semantic search is weak on long, list-structured commi: one vector
for a comma with five lettere, or for an allegato table cut at 512 tokens, averages its parts.
For "Fino a che importo posso affidare direttamente un servizio?" the exact cosine rank of 50/1
(2,135 characters, lettere a-e) is 61; similarities are compressed (0.82-0.86) and a few passages
(140-bis/2, 49/6, all. II.4/2/1, the II.12 tables) sit near every query. Lexical search finds 50/1
at rank 2-4, but reciprocal rank fusion with the noisy semantic list pushes it out of the top 10.
Changes meant to fix this have parameters, and with 73 questions it is easy to pick the ones
that happen to suit them.

**Decision.** The method is fixed before any number is looked at.

1. *Split.* Questions with expected references and an odd numeric id are the development set
   (37), those with an even id the test set (36) (`evaluation.dataset.split_of`). The rule
   depends only on the id, so it is deterministic and independent of the results.
2. *Variants* (`evaluation.sweep.default_grid`), each with every parameter explicit: semantic
   granularity `unit` (one vector per version, as before) or `passage` (max-sim over fixed
   sub-passages: the intro and one passage per lettera, windows of about 1,000 characters with
   up to 200 overlapping; these sizes were set from the problem, not tuned, since each change
   needs a re-embedding); for passages, a hubness correction weight
   λ in {0, 0.5, 1} (score − λ · mean similarity to the 10 nearest probe queries, the corpus
   headings embedded as queries; λ = 0.5 ranks like CSLS); lexical weight w in RRF in {0.5, 1,
   1.5, 2, 3}; fusion depth (each list `d * k` long) in {2, 4, 8}; `RRF_K` stays 60. 60 hybrid
   variants, plus lexical and semantic-only rows for reference.
3. *Selection rule*, applied on the development set only (`choose`): the highest comma-level
   MRR among hybrid variants; variants within 0.01 of it are separated by recall@5, those within
   0.01 of the best recall@5 by the fewest parameters changed from the baseline, then by name.
4. *Adoption rule* (`beats`): the chosen variant becomes the default only if, on the test set,
   its MRR is higher than the baseline's (`hybrid unit w1 d4`) and its recall@5 is not lower.
   Otherwise the baseline stays the default and the result is reported as negative.

**Consequences.** `appalti-qa eval sweep --all --split dev` and `--split test` reproduce the
comparison from the database; results are kept in `eval/results/sweeps/`. The test set is small
(36 questions; one question moving by one rank changes MRR by up to 0.014), and the six temporal
pairs ask the same question at two dates, so each pair has one question in each split: the test
set is not fully independent of the development set for them.

**Result.** On the development set the rule chose `hybrid passage w0.5 d8 h0.5` (passages,
CSLS-style hubness correction, lexical weight 0.5, lists of 8k): MRR 0.816 and recall@5 0.901
against 0.775 and 0.820 for the baseline. On the test set it does not beat the baseline: MRR
0.748 against 0.797, recall@5 0.861 against 0.852. **The baseline stays the default**
(`semantic_granularity=unit`, `lexical_weight=1`, `fusion_depth=4`); the passages are built and
the new settings are available, but off.

| Variant | Dev R@5 | Dev MRR | Test R@5 | Test MRR | All R@5 | All MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid unit w1 d4 (baseline, default) | 0.820 | 0.775 | 0.852 | 0.797 | 0.836 | 0.786 |
| hybrid passage w0.5 d8 h0.5 (chosen on dev) | 0.901 | 0.816 | 0.861 | 0.748 | 0.881 | 0.782 |
| semantic unit | 0.793 | 0.666 | 0.806 | 0.701 | 0.799 | 0.684 |
| semantic passage | 0.874 | 0.708 | 0.819 | 0.726 | 0.847 | 0.717 |

What the numbers say:

- Passage-level embeddings fix the failure that motivated them. For "Fino a che importo posso
  affidare direttamente un servizio?" (at 2026-10-11) 50/1 is outside the top 10 of the
  baseline and second with the chosen variant; Q021 goes from outside the top 10 to rank 5.
  Semantic search alone improves on both splits (MRR +0.042 dev, +0.025 test).
- In hybrid search they move more expected references into the top 5 and 10 (recall@10 on all
  73 questions 0.927 against 0.886; answerable recall@5 0.923 against 0.885, multi-reference
  0.704 against 0.556), but they also lift other list-structured units, and with the lexical
  list weighted 0.5 the right comma more often lands second or third instead of first: on the
  test set 6 questions get worse (Q022, Q052 and Q076 from rank 1 to 2 or 3) and 2 better.
  On the development set it was 6 better and 2 worse. With 37 questions per side the selection
  overfitted the development set.
- Some variants do score above the baseline on the test set (for example `hybrid unit w1 d8`,
  MRR 0.800), but picking one of them now would be choosing on the test set, so the default does
  not change. A new selection needs new questions, or a cross-validated rule fixed in advance.

Results: `eval/results/sweeps/20261011T182205Z_sweep_dev`, `20261011T182509Z_sweep_test` and
`20261011T182915Z_sweep_all` (`.json`, `.md`).

## 9. A small model reorders a wider pool of candidates (fixed before measuring)

**Context.** Decision 8 left two facts: the expected comma is often retrieved but not in the top
5 (recall@10 0.886, MRR 0.786), and passage-level search finds units that unit-level search
misses (50/1 for the direct-award question), while hurting the order of the others. Neither
list alone gives the answer model the right commi first. A language model that reads the
candidates can judge relevance in a way that rank fusion of similarity scores cannot.

**Decision.** One configuration, fixed here before any number is looked at; nothing is tuned.

1. *Candidates.* For the question at its date, three ranked lists of 40 units each: lexical (as
   in hybrid search), semantic on whole units, and semantic on passages (max-sim, no hubness
   correction). They are fused with reciprocal rank fusion (k = 60, equal weights) and the 40
   best are the candidates. Units cited in the question ("art. 50, comma 1") are pinned first,
   as in hybrid search.
2. *Reranker.* Claude Haiku 5.5 (`claude-haiku-5-5`, default effort) reads the question, the
   date and every candidate (label, heading and text, cut at 3,000 characters) and returns, as
   structured output, the candidates useful to answer, most useful first, at most 15.
3. *Ranking.* Pinned units, then the reranker's list, then the other candidates in fused order.
   When the call fails or is declined, the fused order is used and the failure is counted.
4. *Scope.* Only answers are reranked. `GET /api/v1/search` keeps plain hybrid search: it has no
   daily limit and must not spend on model calls.
5. *Measure.* Comma-level retrieval metrics on the development set, the test set and all
   questions, from two independent runs (the model is not deterministic).
6. *Adoption rule.* Reranking becomes the default for answers only if, in both runs, on the test
   set, its MRR is higher than the baseline's (`hybrid unit w1 d4`) and its recall@5 is not
   lower. Otherwise it stays off and the result is reported as negative.

**Caveat.** The test set is no longer untouched: decision 8 evaluated 60 variants on it, and the
idea of adding the passage list comes from those results. This configuration was not selected
among alternatives on it, but a confirmation on new questions is still needed.

**Result.** The rule holds in both runs, so **reranking is the default for answers**
(`RERANK_PROVIDER=anthropic`; without an API key answers keep plain hybrid search).

| Run | Dev R@5 | Dev MRR | Test R@5 | Test MRR | All 73 R@5 | All 73 R@10 | All 73 MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hybrid (baseline) | 0.820 | 0.775 | 0.852 | 0.797 | 0.836 | 0.886 | 0.786 |
| Reranked, run 1 | 0.964 | 0.919 | 0.954 | 0.976 | 0.959 | 0.977 | 0.947 |
| Reranked, run 2 | 0.973 | 0.926 | 0.972 | 0.982 | 0.973 | 0.977 | 0.953 |

- In run 1, 20 questions rank an expected comma higher than the baseline, 50 the same and 3
  lower (Q001, Q049, Q055: from first to second place, with the expected comma still in the
  top 5). Recall@1 rises from 0.660 to 0.833 (run 1) and 0.847 (run 2); the share of questions
  with an expected comma in the top 5 from 88% to 99% and 100%.
- By kind (run 2): answerable recall@5 0.885 → 1.000, temporal 0.833 → 1.000, multi-reference
  0.556 → 0.778 (MRR 0.731 → 0.944).
- In a spot check, the direct-award question that motivated decisions 8 and 9 ranks 50/1 first
  (outside the baseline top 10; tenth in the candidate pool).
- 146 calls, no failure. A call reads about 16,000 input tokens and writes about 200: $0.0018
  per question at list prices, $0.25 for both runs.
- The two runs differ by a few ranks (the model is not deterministic); the gap to the baseline
  is an order of magnitude larger. Three later runs on all 73 questions agree: MRR 0.946 and
  0.957 (`eval/results/20261011T205507Z_retrieval`, `20261011T212153Z_retrieval`) and 0.958 in
  the answer run.

Results: `eval/results/rerank/` (`20261011T203936Z` and `20261011T204123Z` for run 1 on dev and
test, `20261011T204301Z` and `20261011T204453Z` for run 2).

## 10. Acronyms are searched with the code's wording

**Context.** Users write the acronyms of practice, but the code spells most of them out. "DEC"
never occurs in it (the text says "direttore dell'esecuzione del contratto"), nor do DGUE, FVOE,
OEPV, CCNL or RTI. "Voglio sapere i compiti di un DEC e i requisiti per essere DEC" therefore
retrieved allegato II.6 fragments and tables; the reranker found none of the 40 candidates
useful, and the model rightly abstained. Only three acronyms occur in the dataset questions
(RUP, which the code uses, CCNL and CAM), so the evaluation could not show the gap.

**Decision.** A fixed glossary of 21 acronyms (`retrieval/acronyms.py`), each mapped to the
code's own wording, which search appends after the first occurrence of the acronym (lexical and
semantic search, hybrid search and the reranking pool). Matching is case-sensitive on whole
words, and an acronym whose expansion is already in the question is left alone. The model still
reads the question as written. Ambiguous acronyms are left out ("DL" is both *decreto-legge* and
*direttore dei lavori*).

**Result.** On the 73 questions only Q005 (CCNL) changes: hybrid MRR 0 → 0.25 and recall@5
0 → 1, so hybrid recall@5 goes from 0.836 to 0.849 and MRR from 0.786 to 0.789. The published
runs above predate the glossary. For the DEC question the reranked sources are now allegato II.14
art. 31, art. 114 commi 1, 7 and 8 and allegato I.2 art. 8, and the answer cites them. Questions
written with acronyms belong in the next set of evaluation questions.

