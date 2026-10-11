# Evaluation dataset

`dataset.yaml` holds the questions used to measure retrieval and answer quality. They are written
in Italian, phrased the way a RUP, a buyer at a health authority or a bidding company would ask
them. Each answer is grounded in the consolidated text of D.Lgs. 36/2023 as exported from
Normattiva (Akoma Ntoso snapshots in `data/raw/akn/`).

## How it was built

1. Every distinct Normattiva snapshot was parsed into canonical units (article / comma / lettera,
   plus allegati), and consecutive snapshots were diffed to find provisions whose text changed.
2. Questions were drafted across the whole Code: principles, RUP, digitalisation, programming
   and design, thresholds, below-threshold awards, ordinary procedures, requirements and
   exclusion, soccorso istruttorio, guarantees, award criteria, subcontracting, execution,
   concessions and PPP, disputes, and some allegati.
3. A script checked every entry: each `expected_refs` key follows the grammar in
   `appalti_qa.keys` and exists in the snapshot in force on the `vigenza` date, and each
   `source_excerpt` appears verbatim in the referenced text (with the Normattiva `((` `))`
   markers removed and whitespace collapsed).

The questions are a first draft (`status: bozza`). A machine check does not show that an answer
is legally correct, so a person still has to review each one.

## Kinds

| kind | count | meaning |
|---|---|---|
| `answerable` | 52 | answer is in the text in force on 2026-10-11 |
| `temporal` | 12 | the answer depends on the date; mostly pairs of the same question at two `vigenza` dates (before and after D.Lgs. 209/2024 and later amendments) |
| `unanswerable` | 8 | plausible questions the Code does not answer (regional rules, internal ASL organisation, tax or criminal law, figures set elsewhere); the system should say so instead of guessing |
| `multi` | 9 | the answer needs two or more commi or articles |

## Fields

| field | content |
|---|---|
| `id` | `Q001`, `Q002`, ... in file order |
| `question` | the question, in Italian |
| `kind` | `answerable`, `temporal`, `unanswerable` or `multi` |
| `category` | topic tag (`affidamento_diretto`, `soglie`, `procedure`, `rup`, `garanzie`, `subappalto`, `esecuzione`, `digitalizzazione`, `principi`, `concessioni`, `contenzioso`, ...) |
| `vigenza` | ISO date; the answer refers to the text in force on that date. A change applies from the first day Normattiva serves the new text (the date of its boundary snapshot after `appalti-qa download --refine`, decision 7 in `docs/decisions.md`); a temporal question dated just before a change must fall on the earlier version |
| `expected_answer` | 1-3 sentences in Italian |
| `expected_refs` | canonical unit keys (`50/1/b`, `60/2`, `all:I.3/p1`, `all:II.1/2/2`); a lettera ref matches its comma unit by prefix; `[]` for unanswerable |
| `source_excerpt` | verbatim excerpt (at most 300 characters) of the referenced text, to make the reviewer's job faster |
| `difficulty` | `easy`, `medium` or `hard` |
| `status` | `bozza` (draft), `verificata` (checked as is), `corretta` (checked after edits) or `scartata` (discarded by the reviewer: kept in the file, never evaluated) |
| `notes` | caveats for the reviewer, amending act for temporal items |

## Human verification workflow

1. **Export**: `appalti-qa eval export-xlsx review.xlsx` writes one row per question with Italian
   column headers, wrapped text and a frozen header row. With the database loaded, a
   "Testo vigente" column shows the referenced units as in force on the `vigenza` date; add
   `--with-answers RESULTS.json` to include the system's answers from a run.
2. **Review**: a domain expert reads each row against the cited text and sets the **Esito**
   dropdown: `OK` (the row is right as is), `Da correggere` (edit the cells, or describe the fix in
   the **Correzione** column) or `Scartare` (the question should not be used).
3. **Import**: `appalti-qa eval import-xlsx review.xlsx` (use `--dry-run` first) merges the sheet
   back into `dataset.yaml`, keeping ids, order and comments. Edited rows become `corretta`, rows
   marked `OK` without edits become `verificata`, `Scartare` becomes `scartata`, and `Da correggere`
   without edits stays `bozza` with the Correzione text appended to `notes`. The import stops
   without writing if `dataset.yaml` changed after the export or an edited row fails validation
   (`--db` also checks edited rows against the corpus).
4. **Check**: `appalti-qa eval validate --db` re-runs the static checks and the corpus checks
   (every ref in force on its date, every excerpt found in the referenced text).

Published results count **only** questions with status `verificata` or `corretta` (`--all`
includes drafts and marks the report accordingly). Every reported metric states how many
verified questions it is based on.
