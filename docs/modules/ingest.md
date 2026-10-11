# Ingest

Package `appalti_qa.ingest` turns the Normattiva Akoma Ntoso exports of D.Lgs. 36/2023 into
point-in-time versions of every retrieval unit and loads them into PostgreSQL.

```bash
appalti-qa download     # Normattiva -> data/raw/akn/23G00044_YYYYMMDD.xml (cached, polite)
appalti-qa ingest       # parse every file, build versions, load (replaces the act's data)
```

| Module | Content |
| --- | --- |
| `download.py` | `download_snapshots`: session cookie, `dataVigenza` per lifecycle date, caching, spacing between requests |
| `akn.py` | `parse_snapshot`: act metadata, lifecycle events, units of the body and the allegati |
| `articles.py` | units of the code body (structured and flat exports) |
| `annexes.py` | units of the allegati (articles, numbered blocks, tables) |
| `numbering.py` | finds commi and lettere inside running text |
| `text.py` | `normalize_text`, `text_hash`, repeal detection |
| `labels.py` | Italian labels of amending acts from AKN hrefs |
| `versioning.py` | `build_versions`, `effective_date` |
| `loader.py` | `load_corpus`: one transaction, embeddings preserved by `(text_hash, heading)` |
| `commands.py` | `cli_download`, `cli_ingest` |

## Two export shapes

Exports up to 2024-12-31 are *flat*: each article is one paragraph of running text
(`"Art. 50."`, then `"1. ... a) ... b) ... 2. ..."`), and the allegati are flat text in every
snapshot. Exports from 2025-04-01 have one `<paragraph>` per comma and one `<point>` per lettera.
In flat text a label opens a new comma or lettera only when it follows sentence punctuation (or the
end of a `((...))` insertion), continues the numbering (`2` or `1-bis` after `1`), and is not inside
`«...»` quotes. Two defects of the old exports are repaired: a comma wrongly numbered `1.` takes the
expected number when the next comma carries on from it, and articles replaced as a whole (old text
followed by the new one) keep only the last copy.

Unchanged commi produce byte-identical text in both shapes, so the 2025 format switch creates no
versions.

## Effective dates and versions

Normattiva files are requested on the dates in the act's lifecycle, which are the *publication*
dates of the amending acts. The FRBRExpression date of an export says when the text it contains
came into force. When it falls strictly after the previous file's date and before the file's own
date, it is used as the effective date (`20260811` -> 2026-04-07, L. 34/2026 after its vacatio
legis; `20261011` -> 2026-08-26, D.Lgs. 149/2026).

A version changes when the wording (ignoring line breaks and markers) or the heading changes;
`valid_to` is exclusive. The amending act of a version is the latest amendment in the lifecycle of
the snapshot that introduced it; when several acts enter one snapshot, the latest is credited.

## Corpus figures (24 files, 2026-10-11)

| | |
| --- | --- |
| files / distinct snapshots | 24 / 17 (7 byte-identical files skipped) |
| events | 22 (original act + 21 amendments) |
| units / versions | 2,814 / 3,242 (376 units with more than one version) |
| in force on 2026-10-11 | 2,811 versions (1,319 commi, 1 article, 1,491 allegato units), 29 repealed |

## Known source quirks

- Art. 34 commi 5-6 are numbered `1.`/`2.` in the pre-2025 flat exports, so until 2025-04-01 they
  sit inside 34/4; similar defects (art. 62, all. I.7 art. 13-bis) show up as versions on
  2025-04-01.
- Allegati without articles (II.2, II.5, II.6, II.9, ...) are split into blocks `p1`, `p2`, ... in
  document order; where an allegato restarts its numbering in each part, `pN` does not match the
  printed number. Very long tables (all. II.12, I.13 Tabella A) stay single units.
- The amending act is credited per snapshot, not per comma.
