from contextlib import AbstractContextManager, nullcontext
from datetime import date

import psycopg
import pytest
from fastapi.testclient import TestClient

from appalti_qa import retrieval
from appalti_qa.models import Answer
from appalti_qa.web import create_app
from appalti_qa.web.backend import RANGE_RETRY_S, DatabaseBackend, QALogEntry


def unreachable() -> AbstractContextManager[psycopg.Connection]:
    raise psycopg.OperationalError("connection refused")


def entry(**fields: object) -> QALogEntry:
    values: dict[str, object] = {
        "client_hash": "a" * 64,
        "question": "Domanda?",
        "at_date": date(2025, 1, 1),
        "provider": "fake",
        "model": "fake",
    }
    return QALogEntry(**(values | fields))


def test_unreachable_database_degrades() -> None:
    now = [0.0]
    backend = DatabaseBackend(unreachable, clock=lambda: now[0])
    assert backend.ping() is False
    assert backend.available_range() is None
    backend.log(entry())  # best effort: no exception
    with pytest.raises(psycopg.Error):
        backend.lookup("50/1", date(2025, 1, 1))


def test_available_range_is_cached(monkeypatch) -> None:
    now = [0.0]
    calls: list[object] = []

    def fake_range(*, conn: object) -> tuple[date, date]:
        calls.append(conn)
        return date(2023, 4, 1), date(2026, 10, 11)

    monkeypatch.setattr(retrieval, "available_range", fake_range)
    backend = DatabaseBackend(lambda: nullcontext("conn"), clock=lambda: now[0])
    assert backend.available_range() == (date(2023, 4, 1), date(2026, 10, 11))
    now[0] += RANGE_RETRY_S
    backend.available_range()
    assert calls == ["conn"]


def test_corpus_queries_use_the_given_connection(monkeypatch) -> None:
    seen: list[tuple] = []
    monkeypatch.setattr(
        retrieval,
        "lookup",
        lambda refs, at_date, *, conn: seen.append(("lookup", refs, at_date, conn)) or [],
    )
    monkeypatch.setattr(
        retrieval,
        "history",
        lambda ref, *, conn: seen.append(("history", ref, conn)) or [],
    )
    monkeypatch.setattr(
        retrieval,
        "search",
        lambda q, d, k, mode, *, conn: seen.append(("search", q, d, k, mode, conn)) or [],
    )
    backend = DatabaseBackend(lambda: nullcontext("conn"))
    day = date(2025, 1, 1)
    backend.lookup("50/1", day)
    backend.history("50")
    backend.search("soglie", day, 5, "lexical")
    assert seen == [
        ("lookup", ["50/1"], day, "conn"),
        ("history", "50", "conn"),
        ("search", "soglie", day, 5, "lexical", "conn"),
    ]


@pytest.mark.db
def test_log_and_ping_on_real_schema(db_conn: psycopg.Connection) -> None:
    backend = DatabaseBackend(lambda: nullcontext(db_conn))
    assert backend.ping() is True
    backend.log(entry(cited_keys=["50/1", "1/1"], input_tokens=10, latency_ms=5))
    backend.log(entry(error="LLMError"))
    rows = db_conn.execute(
        "SELECT client_hash, cited_keys, input_tokens, error FROM qa_log ORDER BY id"
    ).fetchall()
    assert rows == [("a" * 64, ["50/1", "1/1"], 10, None), ("a" * 64, [], 0, "LLMError")]


@pytest.mark.db
def test_answers_are_logged_without_ip(seeded_conn: psycopg.Connection, settings) -> None:
    def answer(question: str, at_date: date) -> Answer:
        return Answer(
            question=question,
            at_date=at_date,
            text="Non trovato.",
            abstained=True,
            model="fake",
            provider="fake",
        )

    app = create_app(
        settings,
        answer_fn=answer,
        backend=DatabaseBackend(lambda: nullcontext(seeded_conn)),
        trusted_proxies=(),
    )
    client = TestClient(app, client=("203.0.113.77", 50000))
    response = client.post("/api/v1/answer", json={"question": "Domanda?", "date": "2024-06-01"})
    assert response.status_code == 200
    row = seeded_conn.execute("SELECT client_hash, at_date, abstained FROM qa_log").fetchone()
    assert row[1:] == (date(2024, 6, 1), True)
    assert len(row[0]) == 64 and "203.0.113.77" not in row[0]


@pytest.mark.db
def test_norma_page_and_units_api_on_seeded_corpus(
    seeded_conn: psycopg.Connection, settings, stub_answer
) -> None:
    app = create_app(
        settings,
        answer_fn=stub_answer,
        backend=DatabaseBackend(lambda: nullcontext(seeded_conn)),
        trusted_proxies=(),
    )
    client = TestClient(app)
    before = client.get("/norma/50/1/b", params={"data": "2024-06-01"})
    assert before.status_code == 200
    assert "140.000 euro." in before.text and "ingegneria" not in before.text.split("Versioni")[0]
    assert "D.Lgs. 31 dicembre 2024, n. 209" in before.text  # history lists the amendment
    after = client.get("/api/v1/units/50", params={"date": "2025-01-01"})
    [hit] = after.json()
    assert hit["unit_key"] == "50/1"
    assert "ingegneria e architettura" in hit["text"]
    assert hit["modified_by"][0]["label"] == "D.Lgs. 31 dicembre 2024, n. 209"
    page = client.get("/").text
    assert 'min="2023-04-01"' in page
