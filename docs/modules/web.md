# Web

Package `appalti_qa.web`: a server-rendered FastAPI application with Jinja2, htmx and Bootstrap
Italia, plus a JSON API. `create_app()` works with no arguments (the Docker image runs
`uvicorn appalti_qa.web:create_app --factory`); `appalti-qa serve` runs it locally.

| Module | Content |
| --- | --- |
| `app.py` | `create_app`, `cli_serve`, security headers, error pages |
| `pages.py` | HTML routes |
| `api.py` | `/api/v1` routes and `/healthz` |
| `asking.py` | question flow shared by the form and the API: validation, rate limits, logging |
| `backend.py` | `DatabaseBackend`: corpus lookups via `retrieval` and `qa_log` writes |
| `clients.py` | client identification (salted hash of the address bucket, trusted proxies) |
| `querycache.py` | `CachedQueryEmbedder`: LRU cache of query embeddings for `/api/v1/search` |
| `ratelimit.py` | in-process sliding windows |
| `views.py`, `formatting.py`, `results.py` | view models, Italian formatting, evaluation page data |
| `templates/`, `static/` | Jinja2 templates, CSS, htmx configuration and error handling |

## Routes

| Route | Content |
| --- | --- |
| `GET /` | question form, "Testo vigente al" date (default today, bounded by the corpus range), example questions |
| `POST /chiedi` | answer: htmx partial, or the full page without JavaScript |
| `GET /norma/{key}?data=` | text in force at the date and version history of a unit |
| `GET /valutazione` | latest evaluation run (`eval/results/latest.json`); after an answer run, the retrieval modes are compared from `latest_retrieval.json` |
| `GET /informazioni` | "Come funziona": sources, method, limits, privacy, accessibility |
| `POST /api/v1/answer` (alias `POST /api/ask`) | `{"question", "date"?}` -> `Answer` |
| `GET /api/v1/search?q=&date=&k=&mode=` | `list[Hit]`, `k` at most 20 |
| `GET /api/v1/units/{key}?date=` | `list[Hit]` |
| `GET /healthz` | `{"status": "ok", "db": bool}` |
| `GET /api/docs` | OpenAPI documentation |

## Behaviour

- Rate limits on the answering endpoints: `RATE_LIMIT_PER_IP_PER_HOUR` per client and
  `RATE_LIMIT_GLOBAL_PER_DAY` overall (0 disables). `GET /api/v1/search` has a separate
  per-client budget, `RATE_LIMIT_SEARCH_PER_IP_PER_HOUR` (default 120), and repeated queries reuse
  their cached embedding. All limits answer 429 with `Retry-After` and an Italian message; invalid
  requests do not use up the quota. Limits are in process (reset on restart, per worker).
- Clients are identified by a salted SHA-256 of the address (`CLIENT_HASH_SALT`); IPv6 addresses
  are reduced to their /64 network first, so one host cannot rotate addresses to evade the limits.
  `X-Forwarded-For` is read only from peers listed in `FORWARDED_ALLOW_IPS`; with `*` only the
  rightmost hop (the one the immediate proxy appended) is used, unlike uvicorn's leftmost. Raw
  IPs are never stored.
- Every answer, and every failed attempt, is written to `qa_log` in a background task; a logging
  failure never breaks the response.
- Validation errors return 422 with Italian messages; `LLMError` or a database error return 503
  with a generic message (details only in the log). An unexpected error returns 500; for htmx
  requests it is a partial with `role="alert"` swapped into the page. Other failed htmx requests
  (a proxy's 502 or 504, a network error, a timeout) are reported by `app.js`, which writes an
  Italian alert into the target and resets `aria-busy`, the loading indicator and the submit
  button.
- The version history on `/norma/{key}` names the act that introduced each version; editorial
  versions (changes outside Normattiva's amendment markers: punctuation, layout, split into commi,
  reworded notes) are labelled "Modifica redazionale", and a heading that changed while the text
  did not is listed in the notes of that version with its date and act. The heading above each
  text in force is the one in force at the chosen date.

## Security and accessibility

- Bootstrap Italia 2.18.3 CSS and htmx 2.0.11 are pinned with SRI hashes; the Bootstrap Italia
  JavaScript bundle is not used, so the site works fully without JavaScript.
- Content Security Policy without inline scripts, `nosniff`, Referrer-Policy. Model output is
  always escaped; the only markup added is the link of each `[n]` marker to its source and the
  few Markdown forms the model writes (paragraphs, bullet and numbered lists, `**bold**`,
  `### headings`), rebuilt as tags around the escaped text (`views.answer_blocks`).
- WCAG 2.1 AA: `lang="it"`, landmarks, skip link, one `h1`, labelled controls with hints and
  inline errors (`aria-invalid`, `aria-describedby`, error summary), visible focus, an
  `aria-live="polite"` answer region, a text loading indicator, tables with captions, and states
  always stated in text, never by colour alone.
