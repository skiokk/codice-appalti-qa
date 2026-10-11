from datetime import date

import pytest
from fastapi.testclient import TestClient

from appalti_qa.llm import LLMError
from appalti_qa.models import Answer
from appalti_qa.web.asking import today


@pytest.mark.parametrize("path", ["/api/v1/answer", "/api/ask"])
def test_answer_endpoint(client: TestClient, stub_answer, backend, path: str) -> None:
    response = client.post(path, json={"question": "Quando si affida?", "date": "2025-01-15"})
    assert response.status_code == 200
    answer = Answer.model_validate(response.json())
    assert answer.at_date == date(2025, 1, 15)
    assert answer.cited_keys == ["50/1"]
    assert answer.sources[0].normattiva_url.startswith("https://www.normattiva.it/")
    assert stub_answer.calls == [("Quando si affida?", date(2025, 1, 15))]
    assert len(backend.logged) == 1


def test_answer_defaults_to_today(client: TestClient, stub_answer) -> None:
    assert client.post("/api/v1/answer", json={"question": "Domanda?"}).status_code == 200
    assert stub_answer.calls[0][1] == today()


@pytest.mark.parametrize(
    ("body", "field", "message"),
    [
        ({}, "question", "Campo obbligatorio."),
        ({"question": "Domanda?", "date": "31-12-2024"}, "date", "Data non valida"),
        ({"question": "   "}, "question", "Scrivi la domanda."),
        ({"question": "x" * 501}, "question", "al massimo 500 caratteri"),
        ({"question": "Domanda?", "date": "2020-01-01"}, "date", "a partire dal 01/04/2023"),
    ],
)
def test_answer_validation_is_italian(
    client: TestClient, stub_answer, body: dict, field: str, message: str
) -> None:
    response = client.post("/api/v1/answer", json=body)
    assert response.status_code == 422
    [detail] = response.json()["detail"]
    assert detail["loc"][-1] == field
    assert message in detail["msg"]
    assert stub_answer.calls == []


def test_answer_rejects_malformed_json(client: TestClient) -> None:
    response = client.post(
        "/api/v1/answer", content=b"{nope", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "JSON" in response.json()["detail"][0]["msg"]


def test_answer_rate_limits(make_client) -> None:
    client = make_client(rate_limit_per_ip_per_hour=0, rate_limit_global_per_day=1)
    assert client.post("/api/v1/answer", json={"question": "Domanda?"}).status_code == 200
    response = client.post("/api/v1/answer", json={"question": "Domanda?"})
    assert response.status_code == 429
    assert "massimo di domande previsto per oggi" in response.json()["detail"]
    assert 0 < int(response.headers["Retry-After"]) <= 86400


def test_answer_provider_failure(client: TestClient, stub_answer) -> None:
    stub_answer.error = LLMError("rate limited by upstream")
    response = client.post("/api/v1/answer", json={"question": "Domanda?"})
    assert response.status_code == 503
    assert "upstream" not in response.json()["detail"]


def test_answer_service_value_error_is_422(client: TestClient, stub_answer) -> None:
    stub_answer.error = ValueError("La domanda è vuota.")
    response = client.post("/api/v1/answer", json={"question": "Domanda?"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "La domanda è vuota."


def test_search(client: TestClient, backend, hit_factory) -> None:
    backend.hits = [hit_factory("50/1"), hit_factory("1/1")]
    response = client.get(
        "/api/v1/search", params={"q": "affidamento diretto", "date": "2025-01-01", "k": 1}
    )
    assert response.status_code == 200
    assert [hit["unit_key"] for hit in response.json()] == ["50/1"]
    assert backend.searches == [("affidamento diretto", date(2025, 1, 1), 1, "hybrid")]


def test_search_validation(client: TestClient, backend) -> None:
    assert client.get("/api/v1/search", params={"q": " "}).status_code == 422
    response = client.get("/api/v1/search", params={"q": "x", "mode": "fuzzy"})
    assert response.status_code == 422
    assert "Valore non ammesso" in response.json()["detail"][0]["msg"]
    assert client.get("/api/v1/search", params={"q": "x", "k": 0}).status_code == 422
    backend.fail = True
    assert client.get("/api/v1/search", params={"q": "x"}).status_code == 503


def test_units(client: TestClient, backend, hit_factory) -> None:
    backend.hits = [hit_factory("all:I.3/p1")]
    response = client.get("/api/v1/units/all:I.3/p1", params={"date": "2024-01-01"})
    assert response.status_code == 200
    assert response.json()[0]["unit_key"] == "all:I.3/p1"
    assert backend.lookups == [("all:I.3/p1", date(2024, 1, 1))]


def test_units_errors(client: TestClient, backend) -> None:
    assert client.get("/api/v1/units/art50").status_code == 422
    missing = client.get("/api/v1/units/999")
    assert missing.status_code == 404
    assert missing.json()["detail"].startswith("Nessuna disposizione")
    backend.fail = True
    assert client.get("/api/v1/units/50").status_code == 503


def test_healthz(client: TestClient, backend) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "db": True}
    backend.fail = True
    assert client.get("/healthz").json() == {"status": "ok", "db": False}


def test_unknown_api_path_is_json_404(client: TestClient) -> None:
    response = client.get("/api/v1/nothing")
    assert response.status_code == 404
    assert response.json() == {"detail": "La pagina richiesta non esiste o è stata spostata."}


def test_openapi_lists_endpoints(client: TestClient) -> None:
    paths = client.get("/api/openapi.json").json()["paths"]
    assert {"/api/v1/answer", "/api/ask", "/api/v1/search", "/healthz"} <= paths.keys()


def test_search_caps_k(make_client, backend, hit_factory) -> None:
    backend.hits = [hit_factory(f"{number}/1") for number in range(1, 31)]
    client = make_client(default_top_k=50)
    assert len(client.get("/api/v1/search", params={"q": "soglie"}).json()) == 20
    assert client.get("/api/v1/search", params={"q": "soglie", "k": 20}).status_code == 200
    response = client.get("/api/v1/search", params={"q": "soglie", "k": 21})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Deve essere al massimo 20."
    assert [k for _, _, k, _ in backend.searches] == [20, 20]


def test_search_rate_limit_is_per_client_and_separate(make_client, backend) -> None:
    client = make_client(rate_limit_search_per_ip_per_hour=2, rate_limit_per_ip_per_hour=1)
    assert client.get("/api/v1/search", params={"q": " "}).status_code == 422  # not counted
    for _ in range(2):
        assert client.get("/api/v1/search", params={"q": "soglie"}).status_code == 200
    response = client.get("/api/v1/search", params={"q": "soglie"})
    assert response.status_code == 429
    assert "limite di 2 ricerche all'ora" in response.json()["detail"]
    assert 0 < int(response.headers["Retry-After"]) <= 3600
    assert len(backend.searches) == 2
    # the answering budget is untouched, and answering does not use the search budget
    assert client.post("/api/v1/answer", json={"question": "Domanda?"}).status_code == 200
    other = make_client(rate_limit_search_per_ip_per_hour=2, peer="198.51.100.8")
    assert other.get("/api/v1/search", params={"q": "soglie"}).status_code == 200


def test_search_rate_limit_zero_disables(make_client) -> None:
    client = make_client(rate_limit_search_per_ip_per_hour=0)
    for _ in range(3):
        assert client.get("/api/v1/search", params={"q": "soglie"}).status_code == 200
