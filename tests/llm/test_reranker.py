"""AnthropicReranker against synthetic Messages API responses (mock transport, no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import anthropic
import httpx2
import pytest

from appalti_qa.config import Settings
from appalti_qa.llm import LLMError, ModelUsage
from appalti_qa.llm.prompts import RERANK_SYSTEM_PROMPT, RERANK_TEXT_CHARS, rerank_prompt
from appalti_qa.llm.reranker import RANKING_SCHEMA, AnthropicReranker, chosen_indices
from appalti_qa.models import Hit

TEST_KEY = "test-placeholder-key"
Handler = Callable[[httpx2.Request], httpx2.Response]


def reply(text: str, *, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 9000,
            "output_tokens": 120,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1000,
        },
    }


def reranker_for(settings: Settings, handler: Handler) -> AnthropicReranker:
    client = anthropic.Anthropic(
        api_key=TEST_KEY,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicReranker(settings, client=client)


def replying(body: dict[str, Any], requests: list[httpx2.Request] | None = None) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return httpx2.Response(200, json=body)

    return handler


def test_request_uses_haiku_structured_output_and_no_fallback(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    requests: list[httpx2.Request] = []
    reranker = reranker_for(settings, replying(reply('{"commi": [1]}'), requests))
    reranker.rank("Soglia per i servizi?", hits, at_date=at_date, limit=15)

    (request,) = requests
    body = json.loads(request.content)
    assert body["model"] == "claude-haiku-5-5"
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": RANKING_SCHEMA}}
    assert body["system"] == RERANK_SYSTEM_PROMPT
    # Claude Haiku 5.5 rejects a fallback list and non-default sampling parameters
    assert not {"fallbacks", "temperature", "top_p", "thinking"} & body.keys()
    assert "anthropic-beta" not in request.headers
    (user,) = body["messages"]
    assert user["content"] == rerank_prompt("Soglia per i servizi?", hits, at_date, 15)


def test_ranking_is_zero_based_with_usage(
    settings: Settings, hits: list[Hit], at_date: date
) -> None:
    reranker = reranker_for(settings, replying(reply('{"commi": [3, 1]}')))
    ranking = reranker.rank("q", hits, at_date=at_date, limit=15)
    assert ranking.order == [2, 0]
    assert ranking.usage == ModelUsage("claude-haiku-5-5", 10000, 120)


def test_chosen_indices_drop_invalid_and_repeated_numbers() -> None:
    assert chosen_indices([2, 2, 0, 9, "1", True, 3, 1], count=3, limit=15) == [1, 2, 0]
    assert chosen_indices([3, 2, 1], count=3, limit=2) == [2, 1]
    assert chosen_indices([], count=3, limit=15) == []
    with pytest.raises(LLMError):
        chosen_indices(None, count=3, limit=15)


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (reply('{"commi": [1]}', stop_reason="refusal"), 200),
        (reply('{"commi": [1'), 200),
        (reply("[1, 2]"), 200),
        ({"type": "error", "error": {"type": "api_error", "message": "boom"}}, 500),
    ],
    ids=["refusal", "truncated-json", "not-an-object", "server-error"],
)
def test_failures_raise_llm_error(
    settings: Settings, hits: list[Hit], at_date: date, body: dict[str, Any], status: int
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body)

    with pytest.raises(LLMError) as excinfo:
        reranker_for(settings, handler).rank("q", hits, at_date=at_date, limit=15)
    assert TEST_KEY not in str(excinfo.value)


def test_missing_key_is_an_llm_error(settings: Settings) -> None:
    with pytest.raises(LLMError, match="not configured"):
        AnthropicReranker(settings)


def test_prompt_numbers_candidates_and_cuts_long_texts(hits: list[Hit], at_date: date) -> None:
    long = hits[1].model_copy(update={"text": "x" * (RERANK_TEXT_CHARS + 500)})
    prompt = rerank_prompt("Domanda?", [hits[0], long], at_date, 7)
    assert "Data di riferimento: 11/10/2026" in prompt and "Domanda: Domanda?" in prompt
    assert "al massimo 7 commi" in prompt
    assert "[1] Art. 50, comma 1\nRubrica: Procedure per l'affidamento\n1. Le stazioni" in prompt
    assert "[2] Art. 1, comma 1\n" in prompt
    assert "x" * RERANK_TEXT_CHARS + " […]" in prompt
    assert "x" * (RERANK_TEXT_CHARS + 1) not in prompt
