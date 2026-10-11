"""Claude through the Anthropic Messages API, with native citations.

Each retrieved unit is sent as its own plain-text document with citations enabled, so every cited
span comes back with the index of its document and the exact quoted text. The request opts in to
server-side refusal fallback; a refusal that survives it is reported as an abstention.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

import anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaRequestDocumentBlockParam,
    BetaTextBlockParam,
)

from appalti_qa.config import ProviderName, Settings, get_settings
from appalti_qa.llm.base import Generation, LLMError
from appalti_qa.llm.prompts import (
    ABSTAIN_TOKEN,
    SYSTEM_PROMPT,
    question_prompt,
    source_context,
    source_title,
)
from appalti_qa.models import Citation, Hit

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _document(hit: Hit, at_date: date) -> BetaRequestDocumentBlockParam:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": hit.text},
        "title": source_title(hit, at_date),
        "context": source_context(hit),
        "citations": {"enabled": True},
    }


def _with_markers(text: str, indices: Sequence[int]) -> str:
    """Append '[n]' markers (n = index + 1) after a cited text block, before trailing spaces."""
    if not indices:
        return text
    body = text.rstrip()
    markers = "".join(f"[{index + 1}]" for index in indices)
    return f"{body} {markers}{text[len(body) :]}"


def _token_usage(message: BetaMessage) -> tuple[int, int]:
    """(input, output) tokens; input includes tokens written to and read from the cache."""
    usage = message.usage
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    return input_tokens, usage.output_tokens


def _api_error_message(exc: anthropic.APIStatusError) -> str:
    """The API's own explanation of a client error (it never echoes credentials)."""
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error")
    return str(error.get("message", "")) if isinstance(error, dict) else ""


class AnthropicProvider:
    name: ProviderName = "anthropic"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model = model or self._settings.llm_model
        self._client = client or self._make_client()

    def _make_client(self) -> anthropic.Anthropic:
        key = self._settings.api_key("anthropic")
        if key is None:
            raise LLMError("Anthropic API key is not configured")
        return anthropic.Anthropic(api_key=key, timeout=self._settings.llm_timeout_s)

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        if not sources:
            return Generation(ABSTAIN_TOKEN, [], self.model, self.name)
        content: list[BetaContentBlockParam] = [_document(hit, at_date) for hit in sources]
        content.append({"type": "text", "text": question_prompt(question, at_date)})
        message = self._create(
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": content}],
            max_tokens=self._settings.llm_max_tokens,
        )
        return self._parse(message, sources)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        message = self._create(
            system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        refused = message.stop_reason == "refusal"
        text = "" if refused else "".join(b.text for b in message.content if b.type == "text")
        return Generation(
            text, [], message.model, self.name, *_token_usage(message), message.stop_reason
        )

    def _create(
        self,
        *,
        system: list[BetaTextBlockParam],
        messages: list[BetaMessageParam],
        max_tokens: int,
    ) -> BetaMessage:
        try:
            message = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": self._settings.llm_effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APITimeoutError as exc:
            raise LLMError("Anthropic request timed out") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API") from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMError(f"Anthropic rejected the credentials (HTTP {exc.status_code})") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit exceeded (HTTP 429)") from exc
        except (anthropic.BadRequestError, anthropic.NotFoundError) as exc:
            detail = _api_error_message(exc)
            raise LLMError(
                f"Anthropic rejected the request (HTTP {exc.status_code}): {detail}"
            ) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(
                f"Anthropic API error (HTTP {exc.status_code}, request {exc.request_id})"
            ) from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error ({type(exc).__name__})") from exc
        logger.info(
            "Claude response: model=%s stop=%s tokens=%d/%d",
            message.model,
            message.stop_reason,
            message.usage.input_tokens,
            message.usage.output_tokens,
        )
        return message

    def _parse(self, message: BetaMessage, sources: Sequence[Hit]) -> Generation:
        input_tokens, output_tokens = _token_usage(message)
        if message.stop_reason == "refusal":
            category = message.stop_details.category if message.stop_details else None
            logger.warning("Claude declined the request (category: %s)", category)
            return Generation(
                ABSTAIN_TOKEN, [], message.model, self.name, input_tokens, output_tokens, "refusal"
            )
        if message.stop_reason == "max_tokens":
            logger.warning(
                "Claude answer truncated at max_tokens=%d", self._settings.llm_max_tokens
            )
        parts: list[str] = []
        citations: list[Citation] = []
        for block in message.content:
            if block.type != "text":  # thinking and fallback marker blocks carry no answer text
                continue
            indices: list[int] = []
            for cited in block.citations or []:
                if cited.type != "char_location" or not 0 <= cited.document_index < len(sources):
                    continue
                index = cited.document_index
                citation = Citation(
                    unit_key=sources[index].unit_key,
                    cited_text=cited.cited_text,
                    source_index=index,
                )
                if citation not in citations:
                    citations.append(citation)
                if index not in indices:
                    indices.append(index)
            parts.append(_with_markers(block.text, indices))
        return Generation(
            "".join(parts).strip(),
            citations,
            message.model,
            self.name,
            input_tokens,
            output_tokens,
            message.stop_reason,
        )
