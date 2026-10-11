"""JSON API (`/api/v1`) and health check. Error bodies use FastAPI's shape with Italian messages."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from appalti_qa.config import RetrievalMode
from appalti_qa.keys import is_valid_key
from appalti_qa.models import Answer, Hit
from appalti_qa.web.asking import InvalidInput, RateLimited, Unavailable, today, wait_text
from appalti_qa.web.context import Context

router = APIRouter()

DB_UNAVAILABLE = "L'archivio dei testi non è al momento disponibile. Riprova più tardi."
MAX_SEARCH_K = 20  # each search embeds the query and ranks candidates on CPU

_PYDANTIC_MESSAGES = {
    "missing": "Campo obbligatorio.",
    "string_type": "Deve essere un testo.",
    "date_from_datetime_parsing": "Data non valida: usa il formato AAAA-MM-GG.",
    "date_from_datetime_inexact": "Data non valida: usa il formato AAAA-MM-GG.",
    "date_parsing": "Data non valida: usa il formato AAAA-MM-GG.",
    "date_type": "Data non valida: usa il formato AAAA-MM-GG.",
    "int_parsing": "Deve essere un numero intero.",
    "int_type": "Deve essere un numero intero.",
    "greater_than_equal": "Deve essere almeno {ge}.",
    "less_than_equal": "Deve essere al massimo {le}.",
    "literal_error": "Valore non ammesso: usa uno tra {expected}.",
    "json_invalid": "Il corpo della richiesta non è un JSON valido.",
    "model_attributes_type": "Il corpo della richiesta deve essere un oggetto JSON.",
    "dict_type": "Il corpo della richiesta deve essere un oggetto JSON.",
}


class AskPayload(BaseModel):
    question: str = Field(description="Domanda in italiano.")
    date: dt.date | None = Field(
        default=None, description="Data del testo vigente (AAAA-MM-GG); oggi se assente."
    )


def _detail(location: Sequence[str | int], message: str, kind: str) -> dict[str, Any]:
    return {"loc": list(location), "msg": message, "type": kind}


def _italian(error: dict[str, Any]) -> dict[str, Any]:
    template = _PYDANTIC_MESSAGES.get(error["type"], "Valore non valido.")
    ctx = error.get("ctx") or {}
    context = {key: str(value).replace(" or ", " o ") for key, value in ctx.items()}
    try:
        message = template.format(**context)
    except KeyError:
        message = "Valore non valido."
    return _detail(error["loc"], message, error["type"])


def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse({"detail": [_italian(error) for error in exc.errors()]}, 422)


def _invalid(exc: InvalidInput) -> JSONResponse:
    details = [
        _detail(["body", field], message, "value_error") for field, message in exc.errors.items()
    ]
    return JSONResponse({"detail": details}, 422)


@router.post("/api/v1/answer", response_model=Answer, tags=["domande"])
@router.post("/api/ask", response_model=Answer, tags=["domande"])
def answer(
    payload: AskPayload, ctx: Context, request: Request, tasks: BackgroundTasks
) -> Answer | JSONResponse:
    """Answer a question from the text in force at `date`, with citations (rate limited)."""
    try:
        question, day = ctx.asker.validate(payload.question, payload.date)
        return ctx.asker.ask(question, day, ctx.client_hash(request), tasks)
    except InvalidInput as exc:
        return _invalid(exc)
    except RateLimited as exc:
        return JSONResponse(
            {"detail": exc.message}, 429, headers={"Retry-After": str(exc.retry_after_s)}
        )
    except Unavailable as exc:
        return JSONResponse({"detail": exc.message}, 503)


@router.get("/api/v1/search", response_model=list[Hit], tags=["norma"])
def search(
    ctx: Context,
    request: Request,
    q: Annotated[str, Query(description="Testo da cercare.")],
    date: Annotated[dt.date | None, Query(description="Data del testo vigente.")] = None,
    k: Annotated[
        int | None, Query(ge=1, le=MAX_SEARCH_K, description="Numero di risultati.")
    ] = None,
    mode: Annotated[RetrievalMode | None, Query(description="Tipo di ricerca.")] = None,
) -> list[Hit] | JSONResponse:
    """Units in force at `date` that best match `q` (no answer generation; rate limited)."""
    query = q.strip()
    limit = ctx.settings.max_question_chars
    if not query or len(query) > limit:
        message = f"Il testo da cercare deve avere tra 1 e {limit} caratteri."
        return JSONResponse({"detail": [_detail(["query", "q"], message, "value_error")]}, 422)
    exceeded = ctx.search_limiter.acquire(ctx.client_hash(request))
    if exceeded is not None:
        message = (
            f"Hai raggiunto il limite di {exceeded.limit} ricerche all'ora. "
            f"Potrai fare una nuova ricerca tra circa {wait_text(exceeded.retry_after_s)}."
        )
        headers = {"Retry-After": str(exceeded.retry_after_s)}
        return JSONResponse({"detail": message}, 429, headers=headers)
    try:
        return ctx.backend.search(
            query,
            date or today(),
            min(k or ctx.settings.default_top_k, MAX_SEARCH_K),
            mode or ctx.settings.retrieval_mode,
        )
    except psycopg.Error:
        return JSONResponse({"detail": DB_UNAVAILABLE}, 503)


@router.get("/api/v1/units/{key:path}", response_model=list[Hit], tags=["norma"])
def units(
    ctx: Context,
    key: str,
    date: Annotated[dt.date | None, Query(description="Data del testo vigente.")] = None,
) -> list[Hit] | JSONResponse:
    """Text in force at `date` of the units matching `key` ("50", "50/1", "all:I.3/p1")."""
    if not is_valid_key(key):
        message = "Riferimento non valido: usa ad esempio 50, 50/1 o 50/1/b."
        return JSONResponse({"detail": [_detail(["path", "key"], message, "value_error")]}, 422)
    try:
        hits = ctx.backend.lookup(key, date or today())
    except psycopg.Error:
        return JSONResponse({"detail": DB_UNAVAILABLE}, 503)
    if not hits:
        return JSONResponse({"detail": "Nessuna disposizione in vigore alla data indicata."}, 404)
    return hits


@router.get("/healthz", tags=["servizio"])
def healthz(ctx: Context) -> dict[str, Any]:
    return {"status": "ok", "db": ctx.backend.ping()}
