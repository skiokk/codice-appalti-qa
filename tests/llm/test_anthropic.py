"""AnthropicProvider against synthetic Messages API responses (mock transport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import anthropic
import httpx2
import pytest

from appalti_qa.config import Settings
from appalti_qa.llm import ABSTAIN_TOKEN, AnthropicProvider, LLMError
from appalti_qa.llm.anthropic_provider import FALLBACK_BETA
from appalti_qa.llm.prompts import SYSTEM_PROMPT, source_title
from appalti_qa.models import Hit

TEST_KEY = "test-placeholder-key"
Handler = Callable[[httpx2.Request], httpx2.Response]


def message(content: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 80,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 300,
        },
        **fields,
    }


def cite(document_index: int, cited_text: str) -> dict[str, Any]:
    return {
        "type": "char_location",
        "cited_text": cited_text,
        "document_index": document_index,
        "document_title": "titolo",
        "start_char_index": 0,
        "end_char_index": len(cited_text),
    }


def text(value: str, *citations: dict[str, Any]) -> dict[str, Any]:
    return {"type": "text", "text": value, "citations": list(citations) or None}


def provider_for(settings: Settings, handler: Handler) -> AnthropicProvider:
    client = anthropic.Anthropic(
        api_key=TEST_KEY,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicProvider(settings, client=client)


def replying(body: dict[str, Any], requests: list[httpx2.Request] | None = None) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return httpx2.Response(200, json=body)

    return handler


def test_request_uses_documents_citations_effort_and_fallback(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    requests: list[httpx2.Request] = []
    provider = provider_for(settings, replying(message([text(ABSTAIN_TOKEN)]), requests))
    provider.generate("Qual è la soglia per i servizi?", hits, at_date=at_date)

    (request,) = requests
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == settings.llm_max_tokens
    assert body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default"
    assert "thinking" not in body and "temperature" not in body
    assert body["system"] == [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    (user,) = body["messages"]
    *documents, question = user["content"]
    assert [d["source"]["data"] for d in documents] == [hit.text for hit in hits]
    assert all(d["citations"] == {"enabled": True} for d in documents)
    assert all(d["source"]["media_type"] == "text/plain" for d in documents)
    assert documents[0]["title"] == source_title(hits[0], at_date)
    assert "D.Lgs. 31 dicembre 2024, n. 209" in documents[0]["context"]
    assert question["type"] == "text"
    assert "11/10/2026" in question["text"] and "Qual è la soglia" in question["text"]


def test_citations_map_to_unit_keys_with_markers(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    lettera_b = "b) affidamento diretto dei servizi e forniture"
    principle = "perseguono il risultato dell'affidamento"
    reply = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            text("Nel testo vigente al 11/10/2026, "),
            text(
                "l'affidamento diretto di servizi è ammesso sotto i 140.000 euro",
                cite(0, lettera_b),
            ),
            text(".\nLe stazioni appaltanti "),
            text(
                "perseguono il risultato",
                cite(1, principle),
                cite(0, lettera_b),
                cite(1, principle),
            ),
            text("."),
        ]
    )
    generation = provider_for(settings, replying(reply)).generate("Soglia?", hits, at_date=at_date)

    assert generation.text == (
        "Nel testo vigente al 11/10/2026, l'affidamento diretto di servizi è ammesso sotto i "
        "140.000 euro [1].\nLe stazioni appaltanti perseguono il risultato [2][1]."
    )
    assert [(c.unit_key, c.source_index, c.cited_text) for c in generation.citations] == [
        ("50/1", 0, lettera_b),
        ("1/1", 1, principle),
    ]
    assert (generation.input_tokens, generation.output_tokens) == (1500, 80)
    assert (generation.provider, generation.model) == ("anthropic", "claude-opus-5-5")
    assert generation.stop_reason == "end_turn"


def test_out_of_range_document_index_is_dropped(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    reply = message([text("Affermazione", cite(7, "testo"))])
    generation = provider_for(settings, replying(reply)).generate("x", hits, at_date=at_date)
    assert generation.text == "Affermazione"
    assert generation.citations == []


def test_fallback_block_is_skipped_and_serving_model_reported(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    reply = message(
        [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal"},
            },
            text("Risposta", cite(0, "Le stazioni appaltanti")),
        ],
        model="claude-opus-4-8",
    )
    generation = provider_for(settings, replying(reply)).generate("x", hits, at_date=at_date)
    assert generation.text == "Risposta [1]"
    assert generation.model == "claude-opus-4-8"


def test_refusal_is_reported_as_abstention(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    reply = message(
        [], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}
    )
    generation = provider_for(settings, replying(reply)).generate("x", hits, at_date=at_date)
    assert generation.text == ABSTAIN_TOKEN
    assert generation.citations == []
    assert generation.stop_reason == "refusal"


def test_no_sources_means_no_request(settings: Settings, at_date: date) -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("unexpected request")

    generation = provider_for(settings, fail).generate("x", [], at_date=at_date)
    assert generation.text == ABSTAIN_TOKEN


def test_complete_returns_plain_text(settings: Settings) -> None:
    requests: list[httpx2.Request] = []
    reply = message([{"type": "thinking", "thinking": "", "signature": "s"}, text('{"score": 1}')])
    generation = provider_for(settings, replying(reply, requests)).complete(
        "Sei un valutatore.", "Valuta.", max_tokens=512
    )
    assert generation.text == '{"score": 1}'
    body = json.loads(requests[0].content)
    assert body["max_tokens"] == 512
    assert body["system"] == [{"type": "text", "text": "Sei un valutatore."}]


def test_complete_refusal_has_no_text(settings: Settings) -> None:
    reply = message([text("parziale")], stop_reason="refusal")
    generation = provider_for(settings, replying(reply)).complete("s", "p")
    assert generation.text == ""
    assert generation.stop_reason == "refusal"


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (400, "rejected the request (HTTP 400): max_tokens: too large"),
        (401, "rejected the credentials (HTTP 401)"),
        (403, "rejected the credentials (HTTP 403)"),
        (404, "rejected the request (HTTP 404)"),
        (429, "rate limit"),
        (500, "HTTP 500"),
        (529, "HTTP 529"),
    ],
)
def test_http_errors_become_llm_errors(
    settings: Settings, hits: list[Hit], at_date: date, status: int, fragment: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        error = {"type": "error", "error": {"type": "x", "message": "max_tokens: too large"}}
        return httpx2.Response(status, json=error)

    with pytest.raises(LLMError) as raised:
        provider_for(settings, handler).generate("x", hits, at_date=at_date)
    assert fragment in str(raised.value)
    assert TEST_KEY not in str(raised.value)


@pytest.mark.parametrize(
    ("exception", "fragment"),
    [(httpx2.ReadTimeout("slow"), "timed out"), (httpx2.ConnectError("down"), "Could not reach")],
)
def test_network_errors_become_llm_errors(
    settings: Settings, hits: list[Hit], at_date: date, exception: Exception, fragment: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exception

    with pytest.raises(LLMError, match=fragment):
        provider_for(settings, handler).generate("x", hits, at_date=at_date)


def test_missing_key_is_an_error(settings: Settings) -> None:
    with pytest.raises(LLMError, match="not configured"):
        AnthropicProvider(settings)


def test_model_override(settings: Settings) -> None:
    client = anthropic.Anthropic(api_key=TEST_KEY)
    assert AnthropicProvider(settings, model="claude-sonnet-5-5", client=client).model == (
        "claude-sonnet-5-5"
    )
