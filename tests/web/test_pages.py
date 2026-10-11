import json
from datetime import date
from pathlib import Path

import lxml.html
import psycopg
from fastapi.testclient import TestClient

from appalti_qa.llm import LLMError
from appalti_qa.models import UnitVersion
from appalti_qa.web.asking import today

HTMX = {"HX-Request": "true"}


def parse(text: str) -> lxml.html.HtmlElement:
    return lxml.html.document_fromstring(text)


def test_home_renders_form(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    page = parse(response.text)
    textarea = page.get_element_by_id("domanda")
    assert textarea.get("maxlength") == "500"
    assert textarea.get("name") == "domanda"
    date_input = page.get_element_by_id("data")
    assert date_input.get("type") == "date"
    assert date_input.get("value") == today().isoformat()
    assert date_input.get("min") == "2023-04-01"
    assert date_input.get("max") == max(today(), date(2026, 10, 11)).isoformat()
    form = page.get_element_by_id("modulo-domanda")
    assert (form.get("method"), form.get("action"), form.get("hx-post")) == (
        "post",
        "/chiedi",
        "/chiedi",
    )
    assert page.get_element_by_id("esito").get("aria-live") == "polite"


def test_home_without_date_range(client: TestClient, backend) -> None:
    backend.fail = True
    date_input = parse(client.get("/").text).get_element_by_id("data")
    assert date_input.get("min") is None
    assert date_input.get("max") == today().isoformat()


def test_example_link_prefills_question(client: TestClient) -> None:
    page = parse(client.get("/", params={"domanda": "Chi nomina il RUP?"}).text)
    assert page.get_element_by_id("domanda").text == "Chi nomina il RUP?"


def test_ask_without_javascript_returns_full_page(client: TestClient, stub_answer, backend) -> None:
    response = client.post(
        "/chiedi", data={"domanda": "  Quando si affida?  ", "data": "2025-01-15"}
    )
    assert response.status_code == 200
    assert stub_answer.calls == [("Quando si affida?", date(2025, 1, 15))]
    page = parse(response.text)
    assert page.find(".//main") is not None
    result = page.get_element_by_id("esito")
    marker = result.xpath(".//a[@class='rif-fonte']")[0]
    assert marker.get("href") == "#fonte-1"
    source = page.get_element_by_id("fonte-1")
    assert "Art. 50, comma 1" in source.text_content()
    assert "In vigore dal 31/12/2024" in source.text_content()
    assert "Modificato da: D.Lgs. 31 dicembre 2024, n. 209" in source.text_content()
    assert "Testo di prova del comma 50/1." in source.xpath(".//details/blockquote")[0].text
    assert source.xpath(".//a[starts-with(@href, 'https://www.normattiva.it/')]")
    assert source.xpath(".//a[@href='/norma/50/1?data=2025-01-15']")
    others = page.get_element_by_id("fonte-2")
    assert "Art. 1, comma 1" in others.text_content()
    assert "non costituisce" in response.text or "non di parere legale" in response.text
    [entry] = backend.logged
    assert entry.cited_keys == ["50/1"]
    assert entry.question == "Quando si affida?"
    assert len(entry.client_hash) == 64 and "198.51.100.7" not in entry.client_hash


def test_ask_with_htmx_returns_partial_and_oob_fields(client: TestClient) -> None:
    response = client.post("/chiedi", data={"domanda": "Domanda?", "data": ""}, headers=HTMX)
    assert response.status_code == 200
    assert "<html" not in response.text
    assert 'id="titolo-risposta"' in response.text
    fragment = lxml.html.fragment_fromstring(response.text, create_parent="div")
    [fields] = fragment.xpath(".//*[@id='campi-domanda']")
    assert fields.get("hx-swap-oob") == "true"
    assert fragment.xpath(".//*[@id='domanda']")[0].text == "Domanda?"


def test_validation_errors_are_linked_to_fields(client: TestClient, stub_answer) -> None:
    response = client.post("/chiedi", data={"domanda": "   ", "data": "31/02/2025"})
    assert response.status_code == 422
    assert stub_answer.calls == []
    page = parse(response.text)
    assert page.findtext(".//title").startswith("Errore: ")
    textarea = page.get_element_by_id("domanda")
    assert textarea.get("aria-invalid") == "true"
    assert "domanda-errore" in textarea.get("aria-describedby").split()
    assert "Scrivi la domanda" in page.get_element_by_id("domanda-errore").text_content()
    date_input = page.get_element_by_id("data")
    assert date_input.get("aria-invalid") == "true"
    assert "data-errore" in date_input.get("aria-describedby").split()
    summary_links = [a.get("href") for a in page.get_element_by_id("esito").xpath(".//a")]
    assert summary_links == ["#domanda", "#data"]


def test_validation_of_length_and_date_range(client: TestClient) -> None:
    response = client.post("/chiedi", data={"domanda": "x" * 501, "data": "2020-01-01"})
    assert response.status_code == 422
    assert "al massimo 500 caratteri" in response.text
    assert "disponibile a partire dal 01/04/2023" in response.text
    future = client.post("/chiedi", data={"domanda": "Domanda?", "data": "2999-01-01"})
    assert future.status_code == 422
    assert "non può essere successiva" in future.text


def test_windows_newlines_count_once(client: TestClient, stub_answer) -> None:
    question = ("riga\r\n" * 100).strip()  # 499 chars once CRLF is normalised to LF
    response = client.post("/chiedi", data={"domanda": question})
    assert response.status_code == 200
    assert stub_answer.calls[0][0] == question.replace("\r\n", "\n")


def test_htmx_validation_error_is_swapped_with_status(client: TestClient) -> None:
    response = client.post("/chiedi", data={"domanda": ""}, headers=HTMX)
    assert response.status_code == 422
    assert "La domanda non è stata inviata" in response.text
    assert 'aria-invalid="true"' in response.text


def test_model_output_is_escaped(
    client: TestClient, stub_answer, answer_factory, hit_factory
) -> None:
    payload = "<script>alert(1)</script>"
    stub_answer.answer = answer_factory(
        payload,
        date(2025, 1, 1),
        text=f"Testo {payload} [1] e [7] e <img src=x onerror=alert(2)>",
        sources=[hit_factory(heading=payload, text=f"1. {payload}")],
    )
    stub_answer.answer.citations[0].cited_text = payload
    response = client.post("/chiedi", data={"domanda": "Domanda?"})
    assert response.status_code == 200
    assert "<script>alert" not in response.text
    assert "<img src=x" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "[7]" in response.text and 'href="#fonte-7"' not in response.text


def test_abstention_notice(client: TestClient, stub_answer, answer_factory) -> None:
    stub_answer.answer = answer_factory(
        "Domanda?", date(2025, 1, 1), abstained=True, citations=[], text="Non trovato."
    )
    response = client.post("/chiedi", data={"domanda": "Domanda?"})
    page = parse(response.text)
    assert "Nessuna risposta fondata sulle fonti" in page.text_content()
    assert "Fonti consultate" in page.text_content()
    assert not page.xpath("//a[@class='rif-fonte']")


def test_llm_failure_is_503_with_generic_message(client: TestClient, stub_answer, backend) -> None:
    stub_answer.error = LLMError("upstream timeout secret-detail")
    response = client.post("/chiedi", data={"domanda": "Domanda?"})
    assert response.status_code == 503
    assert "Servizio non disponibile" in response.text
    assert "secret-detail" not in response.text
    assert backend.logged[0].error == "LLMError"


def test_rate_limit_per_client(make_client, stub_answer) -> None:
    client = make_client(rate_limit_per_ip_per_hour=2)
    for _ in range(2):
        assert client.post("/chiedi", data={"domanda": "Domanda?"}).status_code == 200
    response = client.post("/chiedi", data={"domanda": "Domanda?"}, headers=HTMX)
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    assert "limite di 2 domande all&#39;ora" in response.text
    assert len(stub_answer.calls) == 2


def test_rate_limit_ignores_invalid_requests(make_client) -> None:
    client = make_client(rate_limit_per_ip_per_hour=1)
    assert client.post("/chiedi", data={"domanda": ""}).status_code == 422
    assert client.post("/chiedi", data={"domanda": "Domanda?"}).status_code == 200


def test_norma_page(client: TestClient, backend, hit_factory) -> None:
    backend.hits = [hit_factory("50/1")]
    backend.versions = [
        UnitVersion(
            unit_key="50/1",
            kind="comma",
            text="1. Testo originario.",
            text_hash="a" * 64,
            valid_from=date(2023, 4, 1),
            valid_to=date(2024, 3, 2),
            change_kind="original",
        ),
        UnitVersion(
            unit_key="50/1",
            kind="comma",
            text="1. Testo originario, con una virgola.",
            text_hash="c" * 64,
            valid_from=date(2024, 3, 2),
            valid_to=date(2024, 12, 31),
            change_kind="editorial",
        ),
        UnitVersion(
            unit_key="50/1",
            kind="comma",
            text="1. Testo modificato.",
            text_hash="b" * 64,
            valid_from=date(2024, 12, 31),
            changed_by=backend.hits[0].modified_by[0],
        ),
    ]
    response = client.get("/norma/50/1", params={"data": "2025-02-01"})
    assert response.status_code == 200
    assert backend.lookups == [("50/1", date(2025, 2, 1))]
    page = parse(response.text)
    assert page.findtext(".//h1") == "Art. 50, comma 1"
    rows = page.xpath("//table/tbody/tr")
    assert len(rows) == 3
    acts = [row.xpath("./td[2]")[0].text_content().strip() for row in rows]
    assert "In vigore dal 31/12/2024" in rows[0].text_content()
    assert acts[0] == "D.Lgs. 31 dicembre 2024, n. 209"
    assert "In vigore dal 02/03/2024 al 30/12/2024" in rows[1].text_content()
    assert acts[1].startswith("Modifica redazionale")
    assert "In vigore dal 01/04/2023 al 01/03/2024" in rows[2].text_content()
    assert acts[2] == "Testo originario"


def test_norma_errors(client: TestClient, backend) -> None:
    assert client.get("/norma/art50").status_code == 404
    assert client.get("/norma/50/1", params={"data": "ieri"}).status_code == 422
    missing = client.get("/norma/999/1")
    assert missing.status_code == 404
    assert "Disposizione non trovata" in missing.text
    backend.fail = True
    assert client.get("/norma/50/1").status_code == 503


def test_evaluation_empty_state(client: TestClient) -> None:
    response = client.get("/valutazione")
    assert response.status_code == 200
    assert "Nessuna valutazione pubblicata" in response.text


def test_evaluation_renders_results(client: TestClient, results_path: Path) -> None:
    results = {
        "run_id": "20261011T120000Z_fake_fake",
        "created_at": "2026-10-11T12:00:00Z",
        "dataset": {"path": "eval/dataset.yaml", "size": 120, "verified": 80},
        "config": {"provider": "fake", "model": "fake", "retrieval_mode": "hybrid", "k": 8},
        "retrieval": {
            "overall": {"recall@5": 0.725, "mrr": 0.61234},
            "by_kind": {"temporal": {"recall@5": 0.5}},
            "by_category": {},
        },
        "answer": {
            "overall": {"correctness": 0.8, "latency_p50_ms": 1500, "input_tokens": 12345},
            "by_kind": {},
            "by_category": {"affidamento_diretto": {"correctness": 1.0}},
        },
        "items": [
            {
                "id": "Q001",
                "retrieved": ["50/1"],
                "cited": ["50/1"],
                "abstained": False,
                "score": 1.0,
                "rationale": "<b>ok</b>",
                "latency_ms": 900,
            }
        ],
    }
    results_path.write_text(json.dumps(results), encoding="utf-8")
    response = client.get("/valutazione")
    page = parse(response.text)
    text = page.text_content()
    for expected in (
        "72,5%",
        "0,612",
        "1.500 ms",
        "12.345",
        "Dipendenti dalla data",
        "Affidamento diretto",
        "ibrida",
        "11/10/2026 12:00 UTC",
        "Q001",
    ):
        assert expected in text
    assert all(table.find("caption") is not None for table in page.xpath("//table"))


def test_evaluation_tolerates_broken_file(client: TestClient, results_path: Path) -> None:
    results_path.write_text("{not json", encoding="utf-8")
    assert "Nessuna valutazione pubblicata" in client.get("/valutazione").text


def test_about_page(client: TestClient) -> None:
    response = client.get("/informazioni")
    assert response.status_code == 200
    page = parse(response.text)
    assert page.findtext(".//h1") == "Come funziona"
    for section in ("fonti", "metodo", "limiti", "privacy", "accessibilita"):
        assert page.get_element_by_id(section) is not None
    assert "Non è un parere legale" in page.text_content()
    assert "Normattiva" in page.text_content()


def test_unknown_page_is_italian_404(client: TestClient) -> None:
    response = client.get("/non-esiste")
    assert response.status_code == 404
    assert "Pagina non trovata" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_security_headers(client: TestClient) -> None:
    headers = client.get("/").headers
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_static_assets_are_served(client: TestClient) -> None:
    for path in ("/static/css/app.css", "/static/css/fonts.css", "/static/js/app.js"):
        assert client.get(path).status_code == 200


def test_backend_failure_during_answer_is_503(client: TestClient, stub_answer) -> None:
    stub_answer.error = psycopg.OperationalError("connection refused")
    assert client.post("/chiedi", data={"domanda": "Domanda?"}).status_code == 503


def test_unexpected_htmx_failure_swaps_an_alert(client: TestClient, stub_answer) -> None:
    stub_answer.error = RuntimeError("bug")
    failing = TestClient(client.app, raise_server_exceptions=False)
    response = failing.post("/chiedi", data={"domanda": "Domanda?"}, headers=HTMX)
    assert response.status_code == 500
    assert "<html" not in response.text and "bug" not in response.text
    alert = lxml.html.fragment_fromstring(response.text)
    assert alert.get("role") == "alert"
    assert alert.xpath("string(.//h2)") == "Errore del servizio"
    assert "Riprova più tardi" in alert.text_content()
    page = failing.post("/chiedi", data={"domanda": "Domanda?"})
    assert page.status_code == 500
    assert parse(page.text).xpath("string(//h1)") == "Errore del servizio"


def test_htmx_swaps_server_rendered_errors_only(client: TestClient) -> None:
    meta = parse(client.get("/").text).xpath("//meta[@name='htmx-config']")[0]
    rules = json.loads(meta.get("content"))["responseHandling"]
    swapped = next(rule["code"] for rule in rules if rule["swap"] and "5" in rule["code"])
    assert set(swapped.split("|")) == {"422", "429", "500", "503"}
    assert rules[-1] == {"code": "[45]..", "swap": False, "error": True}


def test_script_reports_failures_that_are_not_swapped(client: TestClient) -> None:
    script = client.get("/static/js/app.js").text
    for event in ("htmx:responseError", "htmx:sendError", "htmx:timeout"):
        assert f'addEventListener("{event}"' in script
    assert 'setAttribute("role", "alert")' in script
    assert 'removeAttribute("aria-busy")' in script
    assert 'classList.remove("htmx-request")' in script
    assert ".innerHTML" not in script  # messages are inserted as text
