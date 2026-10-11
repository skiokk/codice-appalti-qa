"""FastAPI application factory and the `serve` command.

Configuration comes from `Settings`; the reverse proxies whose `X-Forwarded-For` header is
trusted come from `trusted_proxies` or, by default, `Settings.forwarded_allow_ips`
(FORWARDED_ALLOW_IPS: comma-separated addresses or networks, the same convention as uvicorn).
When it is empty, the header is ignored and the TCP peer identifies the client. Unlike uvicorn,
"*" makes only the rightmost hop count (see `clients`), so clients cannot choose their address.
"""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from appalti_qa import __version__
from appalti_qa.config import Settings, get_settings
from appalti_qa.db import close_pool
from appalti_qa.models import Answer
from appalti_qa.web import api, pages
from appalti_qa.web.asking import AnswerFn, AskService
from appalti_qa.web.backend import Backend, DatabaseBackend
from appalti_qa.web.clients import ClientHasher, TrustedProxies
from appalti_qa.web.context import WebContext
from appalti_qa.web.ratelimit import RateLimiter
from appalti_qa.web.results import RESULTS_PATH
from appalti_qa.web.templating import STATIC_DIR, build_templates

DOCS_PATH = "/api/docs"
_CDN = "https://cdn.jsdelivr.net"
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        f"script-src 'self' {_CDN}",
        f"style-src 'self' {_CDN}",
        f"font-src 'self' {_CDN} data:",
        f"img-src 'self' {_CDN} data:",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "object-src 'none'",
    ]
)
_HTTP_MESSAGES = {
    404: ("Pagina non trovata", "La pagina richiesta non esiste o è stata spostata."),
    405: ("Operazione non consentita", "Questa pagina non supporta l'operazione richiesta."),
}


def _default_answer_fn(settings: Settings) -> AnswerFn:
    def answer(question: str, at_date: date) -> Answer:
        from appalti_qa.answer import answer_question

        return answer_question(question, at_date, settings)

    return answer


def _is_api(request: Request) -> bool:
    path = request.url.path
    return path.startswith("/api/") or path == "/healthz"


def _http_error_handler(request: Request, exc: HTTPException) -> Response:
    title, message = _HTTP_MESSAGES.get(
        exc.status_code, ("Richiesta non valida", "Non è stato possibile completare la richiesta.")
    )
    if _is_api(request):
        return JSONResponse({"detail": message}, exc.status_code, headers=exc.headers)
    ctx: WebContext = request.app.state.web
    return pages.render_error(ctx, request, exc.status_code, title, message)


def _server_error_handler(request: Request, exc: Exception) -> Response:
    message = "Si è verificato un errore imprevisto. Riprova più tardi."
    if _is_api(request):
        return JSONResponse({"detail": message}, 500)
    ctx: WebContext = request.app.state.web
    if request.headers.get("hx-request") == "true":  # swapped into the result region
        return pages.render_error_partial(ctx, request, 500, "Errore del servizio", message)
    return pages.render_error(ctx, request, 500, "Errore del servizio", message)


async def _security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    if not request.url.path.startswith(DOCS_PATH):  # Swagger UI needs an inline script
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    close_pool()


def create_app(
    settings: Settings | None = None,
    *,
    answer_fn: AnswerFn | None = None,
    backend: Backend | None = None,
    results_path: Path = RESULTS_PATH,
    trusted_proxies: Iterable[str] | None = None,
) -> FastAPI:
    """Build the application. Without arguments it uses the environment's settings, the
    database pool and `appalti_qa.answer.answer_question`."""
    settings = settings or get_settings()
    backend = backend or DatabaseBackend()
    proxies = (
        TrustedProxies(trusted_proxies)
        if trusted_proxies is not None
        else TrustedProxies.parse(settings.forwarded_allow_ips)
    )
    salt = settings.client_hash_salt.get_secret_value() if settings.client_hash_salt else None
    limiter = RateLimiter(settings.rate_limit_per_ip_per_hour, settings.rate_limit_global_per_day)
    search_limiter = RateLimiter(settings.rate_limit_search_per_ip_per_hour, global_per_day=0)
    app = FastAPI(
        title="Codice Appalti QA",
        description="Domande sul Codice dei contratti pubblici (D.Lgs. 36/2023) con fonti citate.",
        version=__version__,
        docs_url=DOCS_PATH,
        swagger_ui_oauth2_redirect_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    app.state.web = WebContext(
        settings=settings,
        templates=build_templates(),
        backend=backend,
        asker=AskService(settings, answer_fn or _default_answer_fn(settings), backend, limiter),
        trusted_proxies=proxies,
        hasher=ClientHasher(salt),
        results_path=results_path,
        search_limiter=search_limiter,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(pages.router)
    app.include_router(api.router)
    app.add_exception_handler(RequestValidationError, api.validation_error_handler)
    app.add_exception_handler(HTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _server_error_handler)
    app.middleware("http")(_security_headers)
    return app


def cli_serve(args: argparse.Namespace) -> int:
    """`appalti-qa serve`: run the app with uvicorn (the app itself reads X-Forwarded-For)."""
    import uvicorn

    uvicorn.run(
        "appalti_qa.web:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=False,
        log_level=get_settings().log_level.lower(),
    )
    return 0
