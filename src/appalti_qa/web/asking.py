"""The question flow shared by the HTML form and the JSON API.

Validate the input (Italian messages), apply the rate limits, call the answer function and queue
the qa_log entry. Failures are typed exceptions that each interface renders in its own way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import psycopg
from starlette.background import BackgroundTasks

from appalti_qa.config import Settings
from appalti_qa.llm import LLMError
from appalti_qa.models import Answer
from appalti_qa.web.backend import Backend, QALogEntry
from appalti_qa.web.formatting import format_date
from appalti_qa.web.ratelimit import LimitExceeded, RateLimiter

logger = logging.getLogger(__name__)

AnswerFn = Callable[[str, date], Answer]

QUESTION = "question"
DATE = "date"

UNAVAILABLE_MESSAGE = (
    "Il servizio non è al momento disponibile. Riprova tra qualche minuto; se il problema "
    "persiste, consulta il testo ufficiale su Normattiva."
)


def today() -> date:
    """Today's date in Italy, where the code applies (server date if time zones are missing)."""
    try:
        return datetime.now(ZoneInfo("Europe/Rome")).date()
    except ZoneInfoNotFoundError:
        return date.today()


def parse_date(value: str) -> date:
    """ISO ("2026-10-11") or Italian ("11/10/2026") notation; ValueError otherwise."""
    value = value.strip()
    try:
        return date.fromisoformat(value)
    except ValueError:
        return datetime.strptime(value, "%d/%m/%Y").date()


class InvalidInput(Exception):
    """Field name -> Italian message."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__(errors)
        self.errors = errors


class RateLimited(Exception):
    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_s = retry_after_s


class Unavailable(Exception):
    def __init__(self, message: str = UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


def configured_model(settings: Settings) -> tuple[str, str]:
    """(provider, model) that answers by default: the fake provider when no key is set."""
    provider = settings.effective_provider()
    models = {"anthropic": settings.llm_model, "groq": settings.groq_model}
    return provider, models.get(provider, "fake")


def wait_text(retry_after_s: int) -> str:
    """Italian approximation of a wait in whole minutes: "1 minuto", "12 minuti"."""
    minutes = max(1, round(retry_after_s / 60))
    return "1 minuto" if minutes == 1 else f"{minutes} minuti"


def _limit_message(limit: LimitExceeded) -> str:
    wait = wait_text(limit.retry_after_s)
    if limit.scope == "client":
        return (
            f"Hai raggiunto il limite di {limit.limit} domande all'ora per questo servizio "
            f"dimostrativo. Potrai fare una nuova domanda tra circa {wait}."
        )
    return (
        "Il servizio ha raggiunto il numero massimo di domande previsto per oggi. "
        f"Riprova tra circa {wait}."
    )


class AskService:
    def __init__(
        self,
        settings: Settings,
        answer_fn: AnswerFn,
        backend: Backend,
        limiter: RateLimiter,
    ) -> None:
        self.settings = settings
        self.answer_fn = answer_fn
        self.backend = backend
        self.limiter = limiter

    def date_bounds(self) -> tuple[date | None, date]:
        """Dates a question may refer to: from the first text available up to today."""
        available = self.backend.available_range()
        if available is None:
            return None, today()
        first, last = available
        return first, max(today(), last)

    def _date_problem(self, day: date) -> str | None:
        first, last = self.date_bounds()
        if first is not None and day < first:
            return f"Il testo del Codice è disponibile a partire dal {format_date(first)}."
        if day > last:
            return f"La data non può essere successiva al {format_date(last)}."
        return None

    def validate(self, question: str, at_date: date | str | None) -> tuple[str, date]:
        """The cleaned question and date (default today), or InvalidInput with Italian messages.

        `at_date` may be the raw text of a form field.
        """
        errors: dict[str, str] = {}
        text = question.replace("\r\n", "\n").strip()
        limit = self.settings.max_question_chars
        if not text:
            errors[QUESTION] = "Scrivi la domanda."
        elif len(text) > limit:
            errors[QUESTION] = (
                f"La domanda può contenere al massimo {limit} caratteri: ne contiene {len(text)}."
            )
        day = today()
        if isinstance(at_date, date):
            day = at_date
        elif at_date and at_date.strip():
            try:
                day = parse_date(at_date)
            except ValueError:
                errors[DATE] = "Inserisci una data valida, ad esempio 11/10/2026."
        if DATE not in errors and (problem := self._date_problem(day)):
            errors[DATE] = problem
        if errors:
            raise InvalidInput(errors)
        return text, day

    def ask(self, question: str, at_date: date, client_hash: str, tasks: BackgroundTasks) -> Answer:
        """Answer a validated question; the qa_log entry is written after the response."""
        exceeded = self.limiter.acquire(client_hash)
        if exceeded is not None:
            raise RateLimited(_limit_message(exceeded), exceeded.retry_after_s)
        try:
            answer = self.answer_fn(question, at_date)
        except ValueError as exc:  # the answer service validates the question too
            raise InvalidInput({QUESTION: str(exc)}) from exc
        except (LLMError, psycopg.Error) as exc:
            logger.error("Answering failed: %s: %s", type(exc).__name__, exc)
            provider, model = configured_model(self.settings)
            entry = QALogEntry(
                client_hash=client_hash,
                question=question,
                at_date=at_date,
                provider=provider,
                model=model,
                error=type(exc).__name__,
            )
            tasks.add_task(self.backend.log, entry)
            raise Unavailable() from exc
        tasks.add_task(self.backend.log, QALogEntry.from_answer(client_hash, answer))
        return answer
