# Ingest fixtures

Trimmed copies of three Normattiva Akoma Ntoso exports of D.Lgs. 31 marzo 2023, n. 36
(`23G00044_YYYYMMDD.xml`, the date being the `dataVigenza` requested):

| file | export format | in force |
| --- | --- | --- |
| `23G00044_20240302.xml` | flat: one `<p>` per article | before D.Lgs. 209/2024 |
| `23G00044_20241231.xml` | flat | after D.Lgs. 209/2024 (the 2024 correttivo) |
| `23G00044_20250401.xml` | structured: `<paragraph>` per comma, `<point>` per lettera | after D.L. 25/2025 |

Each file keeps, unchanged, the `identification`, `publication`, `lifecycle` and `references`
metadata (without the ELI `preservation` blocks), the `docTitle`, articles 1, 6, 50, 109 and
225-bis, and the allegato docs "Indice allegati" (shortened), I.3, I.4 (art. 1 and Tabella A) and
II.2-bis (art. 1, from 2024-12-31). The `analysis` section and the per-doc metadata of the
allegati are dropped. The legal text is reproduced as published on Normattiva and is not
authoritative.
