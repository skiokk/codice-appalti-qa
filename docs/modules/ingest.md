# Ingest

Package `appalti_qa.ingest` turns the Normattiva Akoma Ntoso exports of D.Lgs. 36/2023 into
point-in-time versions of every retrieval unit and loads them into PostgreSQL.

```bash
appalti-qa download            # Normattiva -> data/raw/akn/23G00044_YYYYMMDD.xml (cached, polite)
appalti-qa download --refine   # ... then date every change exactly, keep only boundary files
appalti-qa ingest              # parse every file, build versions, load (replaces the act's data)
```

| Module | Content |
| --- | --- |
| `download.py` | `download_snapshots`, `NormattivaFetcher`: session cookie, `dataVigenza` per lifecycle date, caching, spacing between requests |
| `refine.py` | `refine_snapshots`: bisection of the change dates, boundary files only |
| `manifest.py` | `refined.json`: boundaries already verified, last day checked, gaps left by the probe cap |
| `akn.py` | `parse_snapshot`: act metadata, lifecycle events, units of the body and the allegati |
| `articles.py` | units of the code body (structured and flat exports) |
| `annexes.py` | units of the allegati (articles, numbered blocks, tables) |
| `numbering.py` | finds commi and lettere inside running text |
| `text.py` | `normalize_text`, `text_hash`, `content_fingerprint`, `word_fingerprint`, repeal detection |
| `labels.py` | Italian labels of amending acts from AKN hrefs |
| `versioning.py` | `build_versions`, `content_digest`, `effective_date` (the file date; expression date checked) |
| `loader.py` | `load_corpus`: one transaction, embeddings preserved by `(text_hash, heading)` |
| `commands.py` | `cli_download`, `cli_ingest` |

## Two export shapes

Exports up to 2024-12-31 are *flat*: each article is one paragraph of running text
(`"Art. 50."`, then `"1. ... a) ... b) ... 2. ..."`), and the allegati are flat text in every
snapshot. Exports from 2025-03-15 (the first boundary of 2025) have one `<paragraph>` per comma
and one `<point>` per lettera.
In flat text a label opens a new comma or lettera only when it follows sentence punctuation (or the
end of a `((...))` insertion), continues the numbering (`2` or `1-bis` after `1`), and is not inside
`«...»` quotes. Three defects of the old exports are repaired: a comma wrongly numbered `1.` takes
the expected number when the next comma carries on from it, articles replaced as a whole (old text
followed by the new one) keep only the last copy, and the final `1.`, `2.` of art. 34 (a
per-article override in `articles.py`) become commi 5 and 6, as in the structured exports.
Arts. 205 and 207 carry a similar run inside a comma in *both* shapes and are left as exported, so
their keys do not change with the format switch.

Unchanged commi produce the same wording in both shapes, so the 2025 format switch creates no
versions: the trimmed fixtures `20241231` (flat) and `20250401` (structured) have the same
`content_digest`.

## Change dates: refine by bisection

Normattiva files are requested on the dates in the act's lifecycle, which are the *publication*
dates of the amending acts, and the FRBRExpression date that should say when an export's text came
into force is stale in several exports (decision 7). `download --refine` finds the dates instead:

1. every pair of consecutive files whose `content_digest` differs (unit keys, headings and wording,
   layout and metadata ignored) is searched by bisection on `dataVigenza`: a probe equal to the
   older file moves the lower bound, one equal to the newer file the upper bound, and one unlike
   both is an intermediate version, so both halves are searched;
2. when every change is a one-day step, only the first file of each run of equal content is kept,
   named by that first day; probes that are not boundaries and repeated files are deleted;
3. `refined.json` records the boundaries, the last day checked (`verified_through`) and any gap
   left open by `--max-probes` (default 150, logged); the next run skips verified boundaries,
   resumes open gaps and reuses probe files left by an interrupted run, and `download` does not
   request lifecycle dates up to `verified_through` again.

Requests reuse one session cookie (the act's page is visited first), are 2 seconds apart, and an
HTML page in place of XML renews the session once and then fails.

## Versions

`valid_from` of a version is the date in the file name of the snapshot that introduced it (exact
after a refine). The FRBRExpression date is only compared with it: a stale date is logged at
`INFO`, an expression date between the previous file and this one at `WARNING` (the directory was
not refined). Snapshots whose `content_digest` repeats the previous one only refresh the stored
rendering.

A version changes when the wording changes (`content_fingerprint`: whitespace, dots, hyphens,
quotes, accents typed as apostrophes and `((...))` markers ignored); `valid_to` is exclusive. A
change of the article heading alone does not create a version: the open version takes the new
heading. Each version has a `change_kind`:

| `change_kind` | when | `changed_by` |
| --- | --- | --- |
| `original` | first snapshot | none |
| `amendment` | the words changed, or the unit appeared or reappeared | latest act in the export's lifecycle dated on or before `valid_from` |
| `editorial` | only punctuation or capitalisation changed (same `word_fingerprint`), or the article's text is split into commi differently with the same words in document order (art. 62 commi 14-15 and all. I.7 art. 13-bis in the flat exports) | none |

Doubtful attributions are logged: no act published since the previous change date (an act with a
vacatio legis overtaken by a later one), several acts in one interval, an act published by
`valid_from` but missing from the export's own lifecycle.

## Corpus figures (refined 2026-10-11)

| | |
| --- | --- |
| refine | 84 probes, 88 redundant files deleted, 20 boundary files, no gap left by the cap |
| boundaries | 2023-04-01, 2023-05-30, 2023-07-06, 2023-09-20, 2023-11-23, 2023-11-29, 2024-03-02, 2024-12-28, 2024-12-31, 2025-03-15, 2025-04-02, 2025-04-19, 2025-05-14, 2025-05-21, 2025-07-20, 2025-09-10, 2025-11-06, 2026-01-01, 2026-04-07, 2026-08-26 |
| events | 22 (original act + 21 amendments) |
| units / versions | 2,814 / 3,215 (352 units with more than one version) |
| change kinds | 2,575 original, 630 amendment, 10 editorial |
| in force on 2026-10-11 | 2,811 versions (1,319 commi, 1 article, 1,491 allegato units), 29 repealed |

Before the refine, the files requested on the publication dates put D.L. 124/2023 on 2023-09-29
(in force 2023-09-20), D.L. 201/2024 together with D.Lgs. 209/2024 on 2024-12-31 (in force
2024-12-28) and D.L. 25/2025 on 2025-04-01 (in force 2025-03-15); the stale expression dates of
20230919, 20230929, 20231123 and 20250401 could not correct them.

## Known source quirks

- Art. 34 commi 5-6 are numbered `1.`/`2.` in the pre-2025 flat exports; the per-article override
  restores 34/5 and 34/6.
- Allegati without articles (II.2, II.5, II.6, II.9, ...) are split into blocks `p1`, `p2`, ... in
  document order; where an allegato restarts its numbering in each part, `pN` does not match the
  printed number. Very long tables (all. II.12, I.13 Tabella A) stay single units.
- The amending act is credited per change date, not per comma. On 2023-11-23 the only change (the
  tables of all. II.4, re-laid out with cells in a different order) is credited to the Circolare
  MIT 298/2023, the only act in the lifecycle at that date; on 2025-03-15 the note "COMMA
  SOPPRESSO" of 221/1 becomes "PERIODO SOPPRESSO" and counts as an amendment by D.L. 25/2025.
