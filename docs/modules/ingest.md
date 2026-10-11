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
| `articles.py` | units of the code body (structured and flat exports); both readings of a restarted numbering |
| `alignment.py` | `align_numbering`: decides the comma numbering of flat articles against the structured exports |
| `annexes.py` | units of the allegati (articles, numbered blocks, tables) |
| `numbering.py` | finds commi and lettere inside running text |
| `text.py` | `normalize_text`, `text_hash`, `content_fingerprint`, `word_fingerprint`, `marked_words`, repeal detection |
| `labels.py` | Italian labels of amending acts from AKN hrefs |
| `versioning.py` | `build_versions`, `content_digest`, `effective_date` (the file date; expression date checked) |
| `loader.py` | `load_corpus`: one transaction, embeddings preserved by `(text_hash, heading)` |
| `commands.py` | `cli_download`, `cli_ingest` |

## Two export shapes

Exports up to 2024-12-31 are *flat*: each article (art. 128 excepted) is one paragraph of running
text (`"Art. 50."`, then `"1. ... a) ... b) ... 2. ..."`), and the allegati are flat text in every
snapshot. Exports from 2025-03-15 (the first boundary of 2025) have one `<paragraph>` per comma
and one `<point>` per lettera.
In flat text a label opens a new comma or lettera only when it follows sentence punctuation (or the
end of a `((...))` insertion), continues the numbering (`2` or `1-bis` after `1`), and is not inside
`«...»` quotes. An article replaced as a whole (old text followed by the new one) keeps only the
last copy. Text between the article heading and comma 1 is the rest of the heading: the flat
exports put "Governance dei servizi" of art. 221 there, the structured ones in the heading.

Unchanged commi produce the same wording in both shapes, so the 2025 format switch creates no
versions: the trimmed fixtures `20241231` (flat) and `20250401` (structured) have the same
`content_digest`.

### Restarted comma numbering

Some flat exports number a few commi from `1.` again inside an article. The parser splits such
an article twice: as exported (the run stays inside the previous comma) and with every run read
as the next commi (a run is accepted when the label after it carries on from it, or it ends the
article). `alignment.align_numbering` then compares the article with the same article in the
nearest snapshot that exports it structured, where Normattiva numbers each comma. The
renumbered reading is adopted when every comma it changes exists there with a word-sequence
similarity of at least 0.5 (difflib ratio on the word lists) and it matches better than the
reading as exported; otherwise, or without a structured counterpart, the export's numbering is
kept and a warning is logged. The decisions are in `VersionedCorpus.numbering` and summarised by
`appalti-qa ingest`. A renumbered comma is stored with its corrected label (the 2023 text of 41/9
starts with `9.` where the flat export reads `1.`), so quoted spans of these commi before
2024-12-28 follow the structured numbering rather than the flat export literally.

On the corpus (9 flat snapshots, 2023-04-01 to 2024-12-31) five articles restart their
numbering:

| article | flat text | decision | lowest similarity |
| --- | --- | --- | ---: |
| 34 | commi 5-6 read `1.`, `2.` at the end (all 9 flat snapshots) | renumbered, 34/5-34/6 | 1.00 |
| 41 | commi 9-10 read `1.`, `2.`, so 11-15 were merged into comma 8 (7 snapshots up to 2024-03-02) | renumbered, 41/9-41/15 | 0.64 |
| 62 | commi 8-9 read `1.`, `2.`, so 10-18 were merged into comma 7 (7 snapshots up to 2024-03-02) | renumbered, 62/8-62/18 | 0.65 |
| 205 | a quoted `1.`, `2.` inside comma 5 (all 9) | kept: the structured export has no 205/6 | 0 |
| 207 | a quoted `1.` inside comma 2 (all 9) | kept: the structured export has no 207/3 | 0 |

The lowest similarities of arts. 41 and 62 come from commi that changed before the structured
counterpart (41/15 and 62/15, amended by D.Lgs. 209/2024; 62/15 is also split differently from
62/14 in the flat text); the other renumbered commi match it at 0.8 to 1.0.
The 2024-12-28 and 2024-12-31 exports number arts. 41 and 62 right. Before this rule, 41/9-41/15
and 62/8-62/18 appeared only on 2024-12-28 as 20 amendments credited to D.L. 201/2024 (with
41/8 and 62/7), and point-in-time lookups of, for example, 41/13 (costo del lavoro) or 62/14
before that date found nothing; they now exist from 2023-04-01 with the original text (41/13 is
amended on 2024-12-31 by D.Lgs. 209/2024). The allegati, flat in every export, have no
structured counterpart: they keep the local repair of a single comma misnumbered `1.` between
two that continue the numbering.

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
quotes, accents typed as apostrophes and `((...))` markers ignored); `valid_to` is exclusive.
Each row has a `change_kind`:

| `change_kind` | when | `changed_by` |
| --- | --- | --- |
| `original` | first snapshot | none |
| `amendment` | the words that changed touch Normattiva's amendment markers, or the unit becomes repealed (a new unit: any of its text is marked) | latest act in the export's lifecycle dated on or before `valid_from` |
| `editorial` | only punctuation or capitalisation changed (same `word_fingerprint`), the article's words are split into commi differently, or the words changed outside any marker | none |
| `heading` | the text is the previous row's, only the article heading changed | the act as above when the new heading touches markers, otherwise none |

**Amendment markers.** Normattiva wraps text inserted or replaced by an amending act in
`<ins>((...))</ins>`, in the flat and in the structured exports alike, marks a deletion with
`((...))` and a repealed comma with a marked note (`((COMMA ABROGATO DAL ...))`); the markers
stay in later exports. The parser keeps each unit's text with its markers (`ParsedUnit.marked`;
a comma that starts inside a span opened by the previous one, as when D.Lgs. 209/2024 inserted
116/4-bis and 4-ter together, gets an opening marker). Versioning compares the old and the new
words in order (`marked_words`, difflib opcodes): a difference touches the markers when an
inserted or replaced word is marked in the new text, a removed or replaced word was marked in
the old one, or words were removed from inside a span of the new text (art. 63, comma 8,
replaced by a marked text shorter than the old one). A deletion marker counts as a word, and a
repeal or suppression note counts as one word whatever its wording, so "COMMA SOPPRESSO" reworded
as "PERIODO SOPPRESSO" is editorial. A label moving in or out of a span (`((5. Ai fini` vs
`5. ((Ai fini`) changes no word.

**Headings.** A change of the heading alone does not create a version of the text. It closes the
row and opens one with the same text and the new heading (`change_kind = 'heading'`), so every
row keeps the heading in force over its interval: the heading shown, given to the model and
indexed with the text (`tsv`, embedding) is the one in force at the query date, and a later
heading never rewrites an earlier row. Retrieval reports the validity of the text across such
rows and `history` folds them into the version as `heading_changes`. A heading rewritten without
new words (spacing, quotes) only refreshes the rendering.

Doubtful attributions are logged: no act published since the previous change date (an act with a
vacatio legis overtaken by a later one), several acts in one interval, an act published by
`valid_from` but missing from the export's own lifecycle.

## Corpus figures (refined 2026-10-11)

| | |
| --- | --- |
| refine | 84 probes, 88 redundant files deleted, 20 boundary files, no gap left by the cap |
| boundaries | 2023-04-01, 2023-05-30, 2023-07-06, 2023-09-20, 2023-11-23, 2023-11-29, 2024-03-02, 2024-12-28, 2024-12-31, 2025-03-15, 2025-04-02, 2025-04-19, 2025-05-14, 2025-05-21, 2025-07-20, 2025-09-10, 2025-11-06, 2026-01-01, 2026-04-07, 2026-08-26 |
| events | 22 (original act + 21 amendments) |
| units / text versions | 2,814 / 3,213 (352 units with more than one version), plus 25 `heading` rows: 3,238 rows |
| change kinds | 2,593 original, 589 amendment, 31 editorial; 25 heading rows (24 credited: 17 to D.Lgs. 209/2024, 7 to D.L. 73/2025) |
| in force on 2026-10-11 | 2,811 rows (1,319 commi, 1 article, 1,491 allegato units), 29 repealed |

Amendments and editorial versions by boundary, with the rules above (after) and with the earlier
rules that credited every change of words and repaired art. 34 only (before):

| boundary | before: amendment / editorial | after: amendment / editorial / heading |
| --- | --- | --- |
| 2023-04-01 | 2,575 original | 2,593 original (+18: 41/9-15, 62/8-18) |
| 2023-05-30 | 1 / 0 | 1 / 0 / 0 |
| 2023-07-06 | 0 / 1 | 0 / 1 / 0 |
| 2023-09-20 | 1 / 0 | 1 / 0 / 0 |
| 2023-11-23 | 1 / 0 | 0 / 1 / 0 |
| 2023-11-29 | 2 / 0 | 2 / 0 / 0 |
| 2024-03-02 | 5 / 0 | 2 / 3 / 0 |
| 2024-12-28 | 21 / 0 | 1 / 0 / 0 |
| 2024-12-31 | 528 / 0 | 528 / 0 / 18 |
| 2025-03-15 | 28 / 6 | 11 / 23 / 0 |
| 2025-04-02 to 2025-05-14 | 3 / 0 | 3 / 0 / 0 |
| 2025-05-21 | 16 / 1 | 16 / 1 / 7 |
| 2025-07-20 | 18 / 1 | 18 / 1 / 0 |
| 2025-09-10 to 2026-08-26 | 6 / 1 | 6 / 1 / 0 |

What moved, so that a reviewer can judge each class:

- 2024-12-28: the 20 renumbering artefacts of arts. 41 and 62 are gone; 63/4 remains, a marked
  insertion credited to D.L. 201/2024.
- 2023-11-23: the tables of all. II.4, re-laid out with cells in another order, are editorial and
  no longer credited to the Circolare MIT 298/2023.
- 2024-03-02: three annex commi that only gained a stray trailing label (`14.`, `11.`, `29.`) are
  editorial.
- 2024-12-31: unchanged count. 63/8 stays an amendment (comma replaced by a marked text shorter
  than the old one); 116/4-ter and all. I.7, 6-bis/2 are amendments because they open inside the
  marked span of the comma before them.
- 2025-03-15, 17 former amendments now editorial: eight repeal notes reworded ("COMMA SOPPRESSO"
  to "PERIODO SOPPRESSO", 221/1 among them; in 106/8 the flat export also kept the replaced words
  "e secondo"), five notes of all. I.7 whose date was corrected, a label moving out of a span
  (all. II.4, 5/5 and 11/2), a sentence glued to a comma number in the flat text (119/2) and a
  sentence of 125/1 that the 2024-12-31 export had lost.
- 2025-03-15, the 11 amendments left, credited to D.L. 25/2025, all touch marked text: 209/1 and
  all. I.5, 8/1 (a deletion marker replaced by words, a new "((9))"), all. I.7, 40/2 (a
  duplicated "il" next to a marked span), all. I.9, 1/10 (marked text of the flat export
  removed), all. I.11, 8/2-bis, 8/2-ter and 8/4 (new marked text), all. II.2-bis, 16/3 and
  tables C and D (marked notes moved between them), all. V.2, 1/5 (marked text replaced).
  Several look like differences between the flat and the structured export rather than changes
  made by D.L. 25/2025; the markers cannot tell them apart.

Before the refine, the files requested on the publication dates put D.L. 124/2023 on 2023-09-29
(in force 2023-09-20), D.L. 201/2024 together with D.Lgs. 209/2024 on 2024-12-31 (in force
2024-12-28) and D.L. 25/2025 on 2025-04-01 (in force 2025-03-15); the stale expression dates of
20230919, 20230929, 20231123 and 20250401 could not correct them.

## Known source quirks and limits

- Allegati without articles (II.2, II.5, II.6, II.9, ...) are split into blocks `p1`, `p2`, ... in
  document order; where an allegato restarts its numbering in each part, `pN` does not match the
  printed number. Very long tables (all. II.12, I.13 Tabella A) stay single units.
- The amending act is credited per change date, not per comma: when several acts take effect on
  the same day, the latest published one is credited.
- The amendment rule is only as good as the markers. Words changed outside them are editorial
  even if an act changed them (a deletion Normattiva did not mark); marked text that only moved
  or differs between export shapes counts as an amendment (the 2025-03-15 cases above). Removing
  words right next to the end of a marked span that also ends the text counts as an amendment.
- Defects of a single export are not repaired: the 2024-12-31 flat export lacks the last
  sentence of 125/1, which therefore has a version without it from 2024-12-31 to 2025-03-15.
- Art. 221's heading includes "Governance dei servizi" from 2023-04-01: the flat exports carry
  those words between the heading and comma 1, the structured ones in the heading.
- The heading of art. 140 lost "e di protezione civile" on 2025-05-21 (D.L. 73/2025); Normattiva
  marks it with a `((...))` paragraph after the heading, which is read as part of the heading.
- The structured exports (from 2025-03-15) split art. 62 between commi 14 and 15 at "di cui
  all'articolo 15.", reading the article number as the label of comma 15: 62/14 ends with
  "all'articolo" and 62/15 opens with the last two sentences of comma 14 ("Se la procedura di
  aggiudicazione ...") before its own "15. Fermi restando ...". The flat exports split it after
  them, so the 2025-03-15 versions of 62/14 and 62/15 are editorial re-splits. This is not
  repaired.
