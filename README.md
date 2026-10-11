# codice-appalti-qa

Question answering over the Italian Public Procurement Code (D.Lgs. 31 marzo 2023, n. 36,
*Codice dei contratti pubblici*). Answers quote the exact article and comma they rely on, refer to
the text **in force at a chosen date**, and the system's quality is measured on a verified dataset.

- **Point-in-time text**: every comma is versioned across Normattiva's consolidated snapshots, with
  the amending act that changed it.
- **Grounded answers**: hybrid retrieval (Italian full-text + local multilingual embeddings in
  PostgreSQL/pgvector) and Claude with native citations; abstains when the sources do not answer.
- **Measured**: retrieval, citation, abstention and correctness metrics on a human-verified
  dataset of Italian questions.
- **PA-ready interface**: Bootstrap Italia, WCAG 2.1 AA, rate-limited public demo.

Design and module contract: [docs/architecture.md](docs/architecture.md). Decisions:
[docs/decisions.md](docs/decisions.md).

## Quick start

Requirements: Python 3.12, Docker.

```bash
docker compose up -d db                       # PostgreSQL 16 + pgvector on localhost:5433
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"
cp .env.example .env                          # defaults run offline with the fake LLM provider

.venv/bin/appalti-qa download                 # Normattiva Akoma Ntoso snapshots -> data/raw/akn
.venv/bin/appalti-qa ingest                   # parse, version and load into PostgreSQL
.venv/bin/appalti-qa embed                    # local embeddings (CPU)
.venv/bin/appalti-qa serve                    # http://127.0.0.1:8000
.venv/bin/appalti-qa eval run                 # evaluation on eval/dataset.yaml
```

To use Claude or Groq, set `LLM_PROVIDER` and the matching API key in `.env`. Without a key the
fake provider is used.

## Development

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest                              # DB tests need `docker compose up -d db`
```

## Disclaimer

This is an information tool, not legal advice. Always check the official text on
[Normattiva](https://www.normattiva.it).

## License

MIT, see [LICENSE](LICENSE). Normative texts come from Normattiva; official acts are not covered by
copyright (art. 5, Legge 633/1941).
