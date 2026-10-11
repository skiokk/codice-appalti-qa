"""Open-weight models served by Groq, used for comparison.

Sources are numbered in the prompt and the model cites them with "[n]" markers, which are parsed
into citations. The quoted span of a citation is best effort: the sentence of the source with the
highest word overlap with the answer sentence that carries the marker.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date

import groq
from groq.types.chat import ChatCompletion, ChatCompletionMessageParam

from appalti_qa.config import ProviderName, Settings, get_settings
from appalti_qa.llm.base import Generation, LLMError
from appalti_qa.llm.prompts import (
    ABSTAIN_TOKEN,
    NUMBERED_CITATION_RULES,
    SYSTEM_PROMPT,
    numbered_sources,
    question_prompt,
)
from appalti_qa.llm.spans import best_sentence
from appalti_qa.models import Citation, Hit

logger = logging.getLogger(__name__)

# "[1]", "[1, 3]", and the "【1†L2-L4】" style some open-weight models fall back to.
_MARKER = re.compile(
    r"(?P<space>[ \t]*)"
    r"(?:\[(?P<numbers>\d+(?:\s*[,;]\s*\d+)*)\]|【(?P<single>\d+)(?:†[^】]*)?】)"
)
_DIGITS = re.compile(r"\d+")
_REASONING = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANSWER_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def _claim_before(text: str, end: int) -> str:
    """The answer sentence that ends at `end` (the position of a marker), without markers."""
    preceding = _MARKER.sub("", text[:end]).strip()
    pieces = [piece for piece in _ANSWER_SENTENCE.split(preceding) if piece.strip()]
    return pieces[-1] if pieces else ""


def parse_markers(text: str, sources: Sequence[Hit]) -> tuple[str, list[Citation]]:
    """Normalise markers to '[n]', drop out-of-range ones and build the citations they denote."""
    citations: list[Citation] = []
    output: list[str] = []
    pos = 0
    for marker in _MARKER.finditer(text):
        numbers = [int(n) for n in _DIGITS.findall(marker["numbers"] or marker["single"])]
        valid = list(dict.fromkeys(n for n in numbers if 1 <= n <= len(sources)))
        claim = _claim_before(text, marker.start())
        for n in valid:
            hit = sources[n - 1]
            citation = Citation(
                unit_key=hit.unit_key,
                cited_text=best_sentence(claim, hit.text),
                source_index=n - 1,
            )
            if citation not in citations:
                citations.append(citation)
        output.append(text[pos : marker.start()])
        if valid:
            output.append(marker["space"] + "".join(f"[{n}]" for n in valid))
        pos = marker.end()
    output.append(text[pos:])
    return "".join(output), citations


def _token_usage(completion: ChatCompletion) -> tuple[int, int]:
    usage = completion.usage
    return (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)


class GroqProvider:
    name: ProviderName = "groq"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        client: groq.Groq | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = model or self._settings.groq_model
        self._client = client or self._make_client()

    def _make_client(self) -> groq.Groq:
        key = self._settings.api_key("groq")
        if key is None:
            raise LLMError("Groq API key is not configured")
        return groq.Groq(api_key=key, timeout=self._settings.llm_timeout_s)

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        if not sources:
            return Generation(ABSTAIN_TOKEN, [], self.model, self.name)
        completion = self._create(
            [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{NUMBERED_CITATION_RULES}"},
                {
                    "role": "user",
                    "content": (
                        f"{numbered_sources(sources, at_date)}\n\n"
                        f"{question_prompt(question, at_date)}"
                    ),
                },
            ],
            self._settings.llm_max_tokens,
        )
        text, citations = parse_markers(self._answer_text(completion), sources)
        return self._generation(completion, text.strip(), citations)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        completion = self._create(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens,
        )
        return self._generation(completion, self._answer_text(completion), [])

    def _create(
        self, messages: list[ChatCompletionMessageParam], max_tokens: int
    ) -> ChatCompletion:
        try:
            completion = self._client.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=max_tokens
            )
        except groq.APITimeoutError as exc:
            raise LLMError("Groq request timed out") from exc
        except groq.APIConnectionError as exc:
            raise LLMError("Could not reach the Groq API") from exc
        except (groq.AuthenticationError, groq.PermissionDeniedError) as exc:
            raise LLMError(f"Groq rejected the credentials (HTTP {exc.status_code})") from exc
        except groq.RateLimitError as exc:
            raise LLMError("Groq rate limit exceeded (HTTP 429)") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error (HTTP {exc.status_code})") from exc
        except groq.APIError as exc:
            raise LLMError(f"Groq API error ({type(exc).__name__})") from exc
        if not completion.choices:
            raise LLMError("Groq returned no choices")
        logger.info(
            "Groq response: model=%s finish=%s tokens=%d/%d",
            completion.model,
            completion.choices[0].finish_reason,
            *_token_usage(completion),
        )
        return completion

    @staticmethod
    def _answer_text(completion: ChatCompletion) -> str:
        """The message content without inline reasoning (some models emit '<think>' blocks)."""
        return _REASONING.sub("", completion.choices[0].message.content or "").strip()

    def _generation(
        self, completion: ChatCompletion, text: str, citations: list[Citation]
    ) -> Generation:
        finish_reason = completion.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning("Groq answer truncated at the token limit")
        return Generation(
            text, citations, completion.model, self.name, *_token_usage(completion), finish_reason
        )
