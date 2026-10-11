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
| `clients.py` | client identification (salted hash, trusted proxies) |
| `ratelimit.py` | in-process sliding windows |
| `views.py`, `formatting.py`, `results.py` | view models, Italian formatting, evaluation page data |
| `templates/`, `static/` | Jinja2 templates, CSS, htmx configuration |

## Routes

| Route | Content |
| --- | --- |
| `GET /` | question form, "Testo vigente al" date (default today, bounded by the corpus range), example questions |
| `POST /chiedi` | answer: htmx partial, or the full page without JavaScript |
| `GET /norma/{key}?data=` | text in force at the date and version history of a unit |
| `GET /valutazione` | latest evaluation run (`eval/results/latest.json`) |
| `GET /informazioni` | "Come funziona": sources, method, limits, privacy, accessibility |
| `POST /api/v1/answer` (alias `POST /api/ask`) | `{"question", "date"?}` -> `Answer` |
| `GET /api/v1/search?q=&date=&k=&mode=` | `list[Hit]` |
| `GET /api/v1/units/{key}?date=` | `list[Hit]` |
| `GET /healthz` | `{"status": "ok", "db": bool}` |
| `GET /api/docs` | OpenAPI documentation |

## Behaviour

- Rate limits on answering endpoints only: `RATE_LIMIT_PER_IP_PER_HOUR` per client and
  `RATE_LIMIT_GLOBAL_PER_DAY` overall (0 disables); 429 with `Retry-After` and an Italian message.
  Invalid requests do not use up the quota. Limits are in process (reset on restart, per worker).
- Clients are identified by a salted SHA-256 of the address (`CLIENT_HASH_SALT`).
  `X-Forwarded-For` is read only from peers listed in `FORWARDED_ALLOW_IPS`. Raw IPs are never
  stored.
- Every answer, and every failed attempt, is written to `qa_log` in a background task; a logging
  failure never breaks the response.
- Validation errors return 422 with Italian messages; `LLMError` or a database error return 503
  with a generic message (details only in the log).

## Security and accessibility

- Bootstrap Italia 2.18.3 CSS and htmx 2.0.11 are pinned with SRI hashes; the Bootstrap Italia
  JavaScript bundle is not used, so the site works fully without JavaScript.
- Content Security Policy without inline scripts, `nosniff`, Referrer-Policy. Model output is
  always escaped; the only markup added is the link of each `[n]` marker to its source.
- WCAG 2.1 AA: `lang="it"`, landmarks, skip link, one `h1`, labelled controls with hints and
  inline errors (`aria-invalid`, `aria-describedby`, error summary), visible focus, an
  `aria-live="polite"` answer region, a text loading indicator, tables with captions, and states
  always stated in text, never by colour alone.
