"""HTML pages. The question form posts normally and is enhanced with htmx (partial responses)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse

from appalti_qa.keys import is_valid_key, key_label, sort_key
from appalti_qa.models import UnitVersion
from appalti_qa.web.asking import (
    InvalidInput,
    RateLimited,
    Unavailable,
    configured_model,
    parse_date,
    today,
)
from appalti_qa.web.context import Context, WebContext
from appalti_qa.web.results import load_results
from appalti_qa.web.views import answer_view

router = APIRouter(default_response_class=HTMLResponse)

EXAMPLE_QUESTIONS = (
    "Fino a quale importo si può affidare direttamente un contratto di servizi e forniture?",
    "Quali requisiti deve avere il responsabile unico del progetto?",
    "Entro quali limiti è ammesso il subappalto?",
)


@dataclass(slots=True)
class QuestionForm:
    """State of the question form: submitted values, errors and the selectable dates."""

    question: str
    date: str
    min_date: date | None
    max_date: date
    errors: dict[str, str] = field(default_factory=dict)


def _form(ctx: WebContext, question: str = "", day: str = "") -> QuestionForm:
    first, last = ctx.asker.date_bounds()
    return QuestionForm(question, day or today().isoformat(), first, last)


def _render(
    ctx: WebContext,
    request: Request,
    template: str,
    context: dict[str, Any],
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> HTMLResponse:
    page = {
        "max_chars": ctx.settings.max_question_chars,
        "current_path": request.url.path,
        **context,
    }
    return ctx.templates.TemplateResponse(request, template, page, status_code, headers)


def render_error(
    ctx: WebContext, request: Request, status_code: int, title: str, message: str
) -> HTMLResponse:
    return _render(ctx, request, "errore.html", {"title": title, "message": message}, status_code)


def render_error_partial(
    ctx: WebContext, request: Request, status_code: int, title: str, message: str
) -> HTMLResponse:
    """The error as an alert for htmx requests, which swap it into the result region."""
    page = {"title": title, "message": message}
    return _render(ctx, request, "errore_htmx.html", page, status_code)


@router.get("/", name="home")
def home(ctx: Context, request: Request, domanda: str = "") -> HTMLResponse:
    form = _form(ctx, domanda[: ctx.settings.max_question_chars])
    return _render(ctx, request, "index.html", {"form": form, "examples": EXAMPLE_QUESTIONS})


@router.post("/chiedi", name="ask")
def ask(
    ctx: Context,
    request: Request,
    tasks: BackgroundTasks,
    domanda: Annotated[str, Form()] = "",
    data: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Answer page; htmx requests get only the result region plus the refreshed form fields."""
    form = _form(ctx, domanda, data)
    outcome: dict[str, Any] = {}
    status_code, headers = 200, None
    try:
        question, day = ctx.asker.validate(domanda, data)
        answer = ctx.asker.ask(question, day, ctx.client_hash(request), tasks)
        outcome["result"] = answer_view(answer)
    except InvalidInput as exc:
        form.errors = exc.errors
        status_code = 422
    except RateLimited as exc:
        outcome["notice"] = exc.message
        status_code, headers = 429, {"Retry-After": str(exc.retry_after_s)}
    except Unavailable as exc:
        outcome["failure"] = exc.message
        status_code = 503
    htmx = request.headers.get("hx-request") == "true"
    template = "esito_htmx.html" if htmx else "index.html"
    context = {"form": form, "examples": EXAMPLE_QUESTIONS, **outcome}
    return _render(ctx, request, template, context, status_code, headers)


@dataclass(frozen=True, slots=True)
class UnitHistory:
    unit_key: str
    versions: list[UnitVersion]


def _histories(versions: list[UnitVersion]) -> list[UnitHistory]:
    by_unit: dict[str, list[UnitVersion]] = defaultdict(list)
    for version in versions:
        by_unit[version.unit_key].append(version)
    return [
        UnitHistory(key, sorted(by_unit[key], key=lambda v: v.valid_from, reverse=True))
        for key in sorted(by_unit, key=sort_key)
    ]


@router.get("/norma/{key:path}", name="norma")
def norma(ctx: Context, request: Request, key: str, data: str = "") -> HTMLResponse:
    """Text of a unit (or of all units under an article) in force at a date, with its history."""
    if not is_valid_key(key):
        message = f"«{key}» non è un riferimento valido, ad esempio 50/1 per art. 50, comma 1."
        return render_error(ctx, request, 404, "Riferimento non valido", message)
    try:
        day = parse_date(data) if data.strip() else today()
    except ValueError:
        message = "Inserisci la data nel formato AAAA-MM-GG, ad esempio 2024-12-31."
        return render_error(ctx, request, 422, "Data non valida", message)
    try:
        hits = ctx.backend.lookup(key, day)
        versions = ctx.backend.history(key)
    except psycopg.Error:
        message = "L'archivio dei testi non è al momento disponibile. Riprova più tardi."
        return render_error(ctx, request, 503, "Servizio non disponibile", message)
    if not hits and not versions:
        message = f"Nessuna disposizione corrisponde a «{key_label(key)}»."
        return render_error(ctx, request, 404, "Disposizione non trovata", message)
    page = {"key": key, "day": day, "hits": hits, "histories": _histories(versions)}
    return _render(ctx, request, "norma.html", page)


@router.get("/valutazione", name="evaluation")
def evaluation(ctx: Context, request: Request) -> HTMLResponse:
    results = load_results(ctx.results_path)
    return _render(ctx, request, "valutazione.html", {"results": results})


@router.get("/informazioni", name="about")
def about(ctx: Context, request: Request) -> HTMLResponse:
    provider, model = configured_model(ctx.settings)
    page = {
        "provider": provider,
        "model": model,
        "available": ctx.backend.available_range(),
        "settings": ctx.settings,
    }
    return _render(ctx, request, "informazioni.html", page)
