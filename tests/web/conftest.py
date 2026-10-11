"""Fixtures for the web tests: a stub answer function and an in-memory backend (no DB)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from appalti_qa.config import RetrievalMode, Settings
from appalti_qa.models import Answer, Citation, Event, Hit, UnitVersion
from appalti_qa.web import create_app
from appalti_qa.web.backend import QALogEntry

FIRST_DAY = date(2023, 4, 1)
LAST_SNAPSHOT = date(2026, 10, 11)
AMENDMENT = Event(
    date=date(2024, 12, 31),
    amending_urn="/akn/it/act/DECRETO_LEGISLATIVO/stato/2024-12-31/209/!main",
    label="D.Lgs. 31 dicembre 2024, n. 209",
)


def make_hit(unit_key: str = "50/1", text: str | None = None, **fields: object) -> Hit:
    values: dict[str, object] = {
        "unit_key": unit_key,
        "version_id": 1,
        "heading": "Procedure per l'affidamento",
        "text": text or f"1. Testo di prova del comma {unit_key}. Secondo periodo.",
        "score": 1.0,
        "valid_from": AMENDMENT.date,
        "normattiva_url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36~art50!vig=2026-10-11",
        "modified_by": [AMENDMENT],
    }
    return Hit(**(values | fields))


def make_answer(question: str, at_date: date, **fields: object) -> Answer:
    """An answer citing the first of two sources."""
    cited, other = make_hit("50/1"), make_hit("1/1", heading="Principio del risultato")
    values: dict[str, object] = {
        "question": question,
        "at_date": at_date,
        "text": "Si procede mediante affidamento diretto [1].",
        "citations": [
            Citation(unit_key="50/1", cited_text="Testo di prova del comma 50/1.", source_index=0)
        ],
        "sources": [cited, other],
        "model": "fake",
        "provider": "fake",
        "latency_ms": 1234,
    }
    return Answer(**(values | fields))


class StubAnswer:
    """Answer function that records its calls; set `answer` or `error` to steer it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []
        self.answer: Answer | None = None
        self.error: Exception | None = None

    def __call__(self, question: str, at_date: date) -> Answer:
        self.calls.append((question, at_date))
        if self.error is not None:
            raise self.error
        return self.answer or make_answer(question, at_date)


class FakeBackend:
    """In-memory Backend; `fail=True` simulates an unreachable database."""

    def __init__(self) -> None:
        self.range: tuple[date, date] | None = (FIRST_DAY, LAST_SNAPSHOT)
        self.hits: list[Hit] = []
        self.versions: list[UnitVersion] = []
        self.logged: list[QALogEntry] = []
        self.lookups: list[tuple[str, date]] = []
        self.searches: list[tuple[str, date, int, RetrievalMode]] = []
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise psycopg.OperationalError("database unreachable")

    def available_range(self) -> tuple[date, date] | None:
        return None if self.fail else self.range

    def lookup(self, ref: str, at_date: date) -> list[Hit]:
        self._check()
        self.lookups.append((ref, at_date))
        return self.hits

    def history(self, ref: str) -> list[UnitVersion]:
        self._check()
        return self.versions

    def search(self, query: str, at_date: date, k: int, mode: RetrievalMode) -> list[Hit]:
        self._check()
        self.searches.append((query, at_date, k, mode))
        return self.hits[:k]

    def ping(self) -> bool:
        return not self.fail

    def log(self, entry: QALogEntry) -> None:
        self.logged.append(entry)


@pytest.fixture
def stub_answer() -> StubAnswer:
    return StubAnswer()


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def results_path(tmp_path: Path) -> Path:
    return tmp_path / "latest.json"


MakeClient = Callable[..., TestClient]


@pytest.fixture
def make_client(
    settings: Settings, stub_answer: StubAnswer, backend: FakeBackend, results_path: Path
) -> MakeClient:
    """Build a TestClient; keyword arguments override settings, plus `trusted_proxies` and
    `peer` (the client address seen by the app)."""

    def make(
        *,
        trusted_proxies: tuple[str, ...] = (),
        peer: str = "198.51.100.7",
        **overrides: object,
    ) -> TestClient:
        app = create_app(
            settings.model_copy(update=overrides),
            answer_fn=stub_answer,
            backend=backend,
            results_path=results_path,
            trusted_proxies=trusted_proxies,
        )
        return TestClient(app, client=(peer, 50000))

    return make


@pytest.fixture
def client(make_client: MakeClient) -> TestClient:
    return make_client()


@pytest.fixture
def hit_factory() -> Callable[..., Hit]:
    return make_hit


@pytest.fixture
def answer_factory() -> Callable[..., Answer]:
    return make_answer
