"""Reranking of retrieved units by a small Claude model (decision 9).

The model reads the question and the numbered candidates and returns, as structured output, the
numbers of the useful ones, most useful first. Failures raise `LLMError`; callers fall back to the
retrieval order.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import anthropic

from appalti_qa.config import Settings, get_settings
from appalti_qa.llm.base import LLMError, ModelUsage
from appalti_qa.llm.prompts import RERANK_SYSTEM_PROMPT, rerank_prompt
from appalti_qa.models import Hit

logger = logging.getLogger(__name__)

RERANK_MAX_TOKENS = 8000  # adaptive thinking counts against it; the list itself is short
RANKING_SCHEMA = {
    "type": "object",
    "properties": {"commi": {"type": "array", "items": {"type": "integer"}}},
    "required": ["commi"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Ranking:
    """0-based indices of the candidates the model chose, best first, and the call's usage."""

    order: list[int]
    usage: ModelUsage


class Reranker(Protocol):
    model: str

    def rank(
        self, question: str, candidates: Sequence[Hit], *, at_date: date, limit: int
    ) -> Ranking:
        """The useful candidates, best first, at most `limit`. Raises LLMError on failure."""
        ...


def chosen_indices(numbers: object, count: int, limit: int) -> list[int]:
    """Valid 1-based candidate numbers, deduplicated, as 0-based indices (at most `limit`)."""
    if not isinstance(numbers, list):
        raise LLMError("The reranker returned no list of candidates")
    numbers = [n for n in numbers if isinstance(n, int) and not isinstance(n, bool)]
    valid = (n - 1 for n in numbers if 1 <= n <= count)
    return list(dict.fromkeys(valid))[:limit]


class AnthropicReranker:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = model or self._settings.rerank_model
        self._client = client or self._make_client()

    def _make_client(self) -> anthropic.Anthropic:
        key = self._settings.api_key("anthropic")
        if key is None:
            raise LLMError("Anthropic API key is not configured")
        return anthropic.Anthropic(api_key=key, timeout=self._settings.llm_timeout_s)

    def rank(
        self, question: str, candidates: Sequence[Hit], *, at_date: date, limit: int
    ) -> Ranking:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=RERANK_MAX_TOKENS,
                system=RERANK_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": rerank_prompt(question, candidates, at_date, limit),
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": RANKING_SCHEMA}},
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Anthropic reranking failed (HTTP {exc.status_code}, request {exc.request_id})"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic reranking failed ({type(exc).__name__})") from exc
        usage = message.usage
        model_usage = ModelUsage(
            message.model,
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0),
            usage.output_tokens,
        )
        if message.stop_reason != "end_turn":
            raise LLMError(f"Reranking stopped early ({message.stop_reason})")
        reply = "".join(block.text for block in message.content if block.type == "text")
        try:
            numbers = json.loads(reply).get("commi")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise LLMError("The reranker returned invalid JSON") from exc
        order = chosen_indices(numbers, len(candidates), limit)
        logger.info(
            "Reranked %d candidates with %s: %d chosen, tokens=%d/%d",
            len(candidates),
            message.model,
            len(order),
            model_usage.input_tokens,
            model_usage.output_tokens,
        )
        return Ranking(order, model_usage)
