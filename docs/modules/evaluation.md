# Evaluation

Package `appalti_qa.evaluation` measures how well the system finds and cites the right provisions
of D.Lgs. 36/2023 and how good its answers are, on the question set in `eval/dataset.yaml`. It
runs offline by default: with no API key, answers come from the fake provider and correctness is
scored by a lexical-overlap judge.

| Module | Content |
| --- | --- |
| `dataset.py` | `EvalItem`, `load_dataset`, `validate_dataset`, `select_items`, corpus lookups, `embedding_coverage` |
| `metrics.py` | `retrieval_metrics`, `answer_metrics`, `answer_cost`, grouping by kind and category |
| `judge.py` | `JudgeVerdict`, `LLMJudge` (Italian rubric, JSON verdict), `OverlapJudge`, `get_judge` |
| `pricing.py` | token prices, `cost_usd`, `usage_cost` (each model attempt at its own rate) |
| `runner.py` | run ids, results documents for retrieval and answer runs |
| `report.py` | `write_results`, `published_names`, `render_report` (markdown), `render_comparison` |
| `xlsx.py` | `export_xlsx` / `import_xlsx`: the human review round trip |
| `yaml_writer.py` | writes the dataset back, rewriting only changed lines and keeping comments |
| `cli.py` | `cli_main`, the `appalti-qa eval` commands |

## Dataset

The format is in `docs/architecture.md` (section 8) and `eval/README_dataset.md`. `status` is
`bozza`, `verificata` or `corretta`, plus `scartata` for questions the reviewer discarded: they
stay in the file (ids are stable) but are never evaluated.

`appalti-qa eval validate` runs the static checks (unique ids, valid keys, `unanswerable` exactly
when there are no refs, `multi` with two or more refs, excerpt present and at most 300
characters). With `--db` it also checks, for each question at its `vigenza` date, that every ref
matches a unit in force and not repealed, and that the excerpt occurs (whitespace-normalised) in
the text of the referenced units.

Runs evaluate only verified questions (`verificata` or `corretta`) unless `--all` is given; every
results file records how many of the evaluated questions are verified, and the report flags runs
that include drafts.

## Metrics

**Retrieval** (no LLM; questions with expected refs, each retrieved at its own `vigenza` date):

- comma level: an expected ref is found when a retrieved unit matches it with
  `keys.ref_matches` (`50/1/b` is found by unit `50/1`, `50` by any comma of article 50);
- article level: the article of each distinct expected ref (`50`, `all:I.1/2`; `all:I.3` for
  allegato blocks) is looked up among the articles of the retrieved units;
- `recall@k`: share of expected refs (or articles) found in the top k; `hit@k`: at least one
  found; `mrr`: reciprocal rank of the first one found. Defaults: k = 1, 3, 5, 10.

`appalti-qa eval retrieval` computes them for each mode (lexical, semantic, hybrid), and records
the retrieval settings (`semantic_granularity`, `hubness_weight`, `lexical_weight`,
`fusion_depth`, `rrf_k`) in the run's configuration.

**Tuning retrieval: split and sweep.** Retrieval parameters are chosen on a fixed split
(`dataset.split_of`, decision 8): questions with expected refs and an odd id are the development
set (37), those with an even id the test set (36). `--split dev|test` restricts `eval retrieval`
(a split run cannot be published). `appalti-qa eval sweep --all --split dev` evaluates the fixed
grid of `evaluation/sweep.py` (lexical, semantic per granularity, and 60 hybrid variants) and
applies the selection rule stated there before any run (best development MRR; ties within 0.01
by recall@5, then by fewest changes from the baseline); `--split test --chosen-from DEV.json`
reports the same grid on the test set and says whether the chosen variant beats the baseline
(higher MRR, recall@5 not lower), the condition for making it the default. `--only VARIANT`
(repeatable) restricts the grid. Sweeps are written to `eval/results/sweeps/` and never
published. Query embeddings are cached within a sweep.

**Answers** (`appalti-qa eval answers`, alias `run`): each question goes through
`answer.answer_question` with the configured provider, and its sources come from the same
`search(question, date, k, mode)` call the app makes. The retrieval metrics of the run come from
separate calls at the largest cutoff (10), so a question with refs is searched twice; the
`retrieved` field of each item still lists the sources the model received.

- `correctness`: judge score, 1 correct, 0.5 partial, 0 wrong. An abstention scores 1 on an
  unanswerable question and 0 otherwise, without calling the judge.
- `citation_precision`: share of cited units that match an expected ref (answers with
  citations); `citation_recall`: share of expected refs cited (abstentions count as 0). Both are
  computed on questions with expected refs only.
- `abstention_accuracy`: the system abstains on unanswerable questions and answers the others;
  `abstention_rate` is reported too.
- `latency_p50_ms`, `latency_p95_ms` (nearest rank), token totals, `cost_usd` and
  `judge_cost_usd` (every attempt of a call that fell back to another model, see Cost);
  `cost_unknown` and `judge_cost_unknown` count the answers and judgements that used a model
  without a known price, which the totals leave out.
- A failing answer or judge call (timeout, rate limit, unparsable verdict) is recorded on the item
  (`error`, `judge_error`), counted, and left out of the averages it affects.

Every group (overall, `by_kind`, `by_category`) carries `n`.

## Judges

`get_judge()` resolves `JUDGE_PROVIDER` like the answer provider: without a key it falls back to
`OverlapJudge`. `LLMJudge` sends an Italian rubric through `LLMProvider.complete` and expects
`{"verdict": "correct" | "partial" | "wrong", "reason": "..."}`; replies wrapped in prose or code
fences, and Italian labels, are accepted. `OverlapJudge` is the share of the expected answer's
content words (accent-folded, stopwords removed, six-letter stems, numbers kept) present in the
answer: 50% or more is correct, 25% or more partial. It is a pipeline check, not a quality
measure, and the report says so.

## Cost

List prices per million tokens, without prompt-caching discounts (tokens written to or read from
the cache count as input):

| Model | Input | Output |
| --- | --- | --- |
| `claude-opus-5-5` | $4.00 | $20.00 |
| `claude-sonnet-5-5` | $2.00 | $10.00 |
| `claude-haiku-5-5` (prompts up to 100K tokens) | $0.10 | $0.50 |
| `claude-fable-5-1` | $10.00 | $50.00 |
| `claude-opus-5`, `claude-opus-4-8` (refusal fallback of Claude Opus 5.5) | $5.00 | $25.00 |
| `claude-sonnet-5` (refusal fallback of Claude Sonnet 5.5) | $2.00 | $10.00 |
| Groq models (free tier), fake provider | $0 | $0 |

Requests to Claude opt in to the server-side refusal fallback, so one call can run on more than
one model: a request the requested model declines, even after it has started writing, is re-run
on its fallback model, and each attempt is billed at its own model's rate. The response's
top-level usage covers only the attempt that produced the reply; `usage.iterations` lists every
attempt with its model. `AnthropicProvider` reports the tokens of all attempts and keeps them per
model in `Generation.iterations`, and the evaluation prices each attempt separately
(`usage_cost`): the judge from its own call, the answers through a `UsageMeter` around the
answering provider, since an `Answer` carries only the model that wrote it and the token totals.
Every reported attempt is priced, so a decline before any output that turns out not to be billed
is counted anyway, and the total is an upper bound in that case.

Known prices are summed; a call that used an Anthropic model missing from the table is counted in
`cost_unknown` instead of adding a wrong number, and the total is `null` only when no answer has a
known price.

## Results

`write_results` writes `eval/results/<run_id>.json` and `<run_id>.md`, where the run id is
`<UTC timestamp>_<provider>_<model>` (for example `20261011T120000Z_anthropic_claude-opus-5-5`).
The JSON follows `docs/architecture.md` and adds, per item, the expected refs, the answer text
and citations, citation precision and recall, tokens, cost and the judge verdict. Retrieval-only
runs (`<timestamp>_retrieval.json`) add `retrieval_by_mode` and per-item rankings for each mode.

Only `--publish` updates the published files (`published_names`):

| Run | Published as |
| --- | --- |
| `answers --publish` | `latest.json` / `latest.md`, read by the web page `/valutazione` |
| `retrieval --publish` | `latest_retrieval.json` / `latest_retrieval.md`, the stable report with every mode; also `latest.json` / `latest.md` unless they hold an answer run |

So the per-mode retrieval report survives later answer runs, and a retrieval run never replaces
published answers on the web page (useful before any answer run exists). When `latest.json`
holds an answer run, the web page takes the comparison of the modes from
`latest_retrieval.json`.

Publishing is refused before anything runs (exit 2, nothing written) when the numbers would not
describe the system:

- `answers --publish`, when the answering provider or the judge resolves to the fake provider
  (placeholder numbers);
- `retrieval --publish` with the semantic or hybrid mode, and `answers --publish` retrieving with
  one of them, when `EMBEDDING_MODEL=fake` or fewer than 99% of the versions searchable on the
  evaluated dates (in force on at least one question's `vigenza`, not repealed) have an embedding
  from the configured model (`embedding_coverage`; with `SEMANTIC_GRANULARITY=passage`, all
  their passages too). Semantic search only returns versions embedded by that model, so it
  would come back empty or partial and hybrid search would degrade to lexical. Lexical-only runs
  need no embeddings.

```bash
appalti-qa eval validate --db
appalti-qa eval retrieval --all --k 1,3,5,10 --publish # all three modes -> latest_retrieval.md
appalti-qa eval answers --provider anthropic --judge-provider anthropic --publish
appalti-qa eval answers --provider groq --all --delay 2 # free tier rate limits
appalti-qa eval report eval/results/latest.json
appalti-qa eval compare eval/results/2026*.json
```

## Human review

`appalti-qa eval export-xlsx revisione.xlsx [--with-answers RESULTS.json]` writes a workbook for
the reviewer: one row per question with Italian headers, wrapped text, frozen header and ID
column, an "Esito" dropdown (OK / Da correggere / Scartare), a free-text "Correzione" column, an
instructions sheet and, when the database is reachable, the text in force of the referenced units
("Testo vigente"). A hidden sheet keeps the exported values.

`appalti-qa eval import-xlsx revisione.xlsx [--dry-run] [--db]` merges the review into the YAML:

| Reviewer action | Result |
| --- | --- |
| OK, no edits | `verificata` (`corretta` stays `corretta`) |
| cells edited (question, answer, refs, excerpt, date, kind, category, difficulty) | edits applied, `corretta` |
| Da correggere, no edits | `bozza` |
| Scartare | `scartata` |
| text in "Correzione" | appended to `notes` as `Revisione: ...` |
| no Esito | row ignored (a warning if cells were edited) |

Edits are detected against the exported values, so a field changed in the YAML after the export
is reported as a conflict instead of being overwritten. Changed questions go through the static
checks (and the corpus checks with `--db`) before anything is written; on any problem the file is
left untouched. The file keeps its order, layout and comments: only the changed lines are
rewritten, and importing the same workbook twice changes nothing.
