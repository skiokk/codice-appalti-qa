"""GroqProvider against synthetic chat completions (mock transport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import groq
import httpx
import pytest

from appalti_qa.config import Settings
from appalti_qa.llm import ABSTAIN_TOKEN, GroqProvider, LLMError
from appalti_qa.llm.groq_provider import parse_markers
from appalti_qa.llm.prompts import NUMBERED_CITATION_RULES, SYSTEM_PROMPT
from appalti_qa.models import Hit

TEST_KEY = "test-placeholder-key"
Handler = Callable[[httpx.Request], httpx.Response]


def completion(content: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_760_000_000,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 60, "total_tokens": 960},
    }


def provider_for(settings: Settings, handler: Handler) -> GroqProvider:
    client = groq.Groq(
        api_key=TEST_KEY,
        max_retries=0,
        http_client=groq.DefaultHttpxClient(transport=httpx.MockTransport(handler)),
    )
    return GroqProvider(settings, client=client)


def replying(body: dict[str, Any], requests: list[httpx.Request] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, json=body)

    return handler


def test_request_numbers_the_sources(settings: Settings, hits: list[Hit], at_date: date) -> None:
    requests: list[httpx.Request] = []
    provider_for(settings, replying(completion(ABSTAIN_TOKEN), requests)).generate(
        "Qual è la soglia?", hits, at_date=at_date
    )
    (request,) = requests
    assert request.url.path == "/openai/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["max_completion_tokens"] == settings.llm_max_tokens
    system, user = body["messages"]
    assert system == {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{NUMBERED_CITATION_RULES}"}
    assert "[1] Art. 50, comma 1 — D.Lgs. 36/2023 (testo vigente al 11/10/2026)" in user["content"]
    assert "[3] Allegato I.3, punto 1" in user["content"]
    assert user["content"].endswith("Domanda: Qual è la soglia?")


def test_markers_become_citations(settings: Settings, hits: list[Hit], at_date: date) -> None:
    answer = (
        "<think>ragionamento interno</think>Nel testo vigente al 11/10/2026 l'affidamento "
        "diretto di servizi e forniture è ammesso sotto i 140.000 euro [1]. Le stazioni "
        "appaltanti perseguono il risultato [2, 9]."
    )
    generation = provider_for(settings, replying(completion(answer))).generate(
        "Soglia?", hits, at_date=at_date
    )
    assert generation.text == (
        "Nel testo vigente al 11/10/2026 l'affidamento diretto di servizi e forniture è "
        "ammesso sotto i 140.000 euro [1]. Le stazioni appaltanti perseguono il risultato [2]."
    )
    assert [(c.unit_key, c.source_index) for c in generation.citations] == [
        ("50/1", 0),
        ("1/1", 1),
    ]
    assert generation.citations[0].cited_text == (
        "affidamento diretto dei servizi e forniture, ivi compresi i servizi di ingegneria e "
        "architettura, di importo inferiore a 140.000 euro."
    )
    for citation in generation.citations:
        assert citation.cited_text in hits[citation.source_index].text
    assert (generation.input_tokens, generation.output_tokens) == (900, 60)
    assert (generation.provider, generation.model) == ("groq", "openai/gpt-oss-120b")
    assert generation.stop_reason == "stop"


def test_parse_markers_normalises_variants(hits: list[Hit]) -> None:
    text, citations = parse_markers(
        "La procedura aperta dura nove mesi 【3†L1-L2】. Altro [0]. Fine [1; 3].",
        hits,
    )
    assert text == "La procedura aperta dura nove mesi [3]. Altro. Fine [1][3]."
    assert [c.source_index for c in citations] == [2, 0, 2]
    assert citations[0].cited_text == "procedura aperta: nove mesi;"
    # "Fine" shares no word with source 3: its first sentence is quoted.
    assert (
        citations[2].cited_text == "Le gare di appalto si concludono nei seguenti termini massimi:"
    )


def test_abstention_has_no_citations(settings: Settings, hits: list[Hit], at_date: date) -> None:
    generation = provider_for(settings, replying(completion(ABSTAIN_TOKEN))).generate(
        "x", hits, at_date=at_date
    )
    assert generation.text == ABSTAIN_TOKEN
    assert generation.citations == []


def test_complete_returns_plain_text(settings: Settings) -> None:
    generation = provider_for(settings, replying(completion('{"score": 0.5}'))).complete(
        "Sei un valutatore.", "Valuta."
    )
    assert generation.text == '{"score": 0.5}'
    assert generation.citations == []


def test_no_sources_means_no_request(settings: Settings, at_date: date) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected request")

    assert provider_for(settings, fail).generate("x", [], at_date=at_date).text == ABSTAIN_TOKEN


@pytest.mark.parametrize(
    ("status", "fragment"),
    [(401, "credentials"), (429, "rate limit"), (400, "HTTP 400"), (503, "HTTP 503")],
)
def test_http_errors_become_llm_errors(
    settings: Settings, hits: list[Hit], at_date: date, status: int, fragment: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(LLMError) as raised:
        provider_for(settings, handler).generate("x", hits, at_date=at_date)
    assert fragment in str(raised.value)
    assert TEST_KEY not in str(raised.value)


def test_timeout_becomes_llm_error(settings: Settings, hits: list[Hit], at_date: date) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="timed out"):
        provider_for(settings, handler).generate("x", hits, at_date=at_date)


def test_missing_key_is_an_error(settings: Settings) -> None:
    with pytest.raises(LLMError, match="not configured"):
        GroqProvider(settings)
