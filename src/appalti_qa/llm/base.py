"""Provider-neutral types for answer generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from appalti_qa.config import ProviderName
from appalti_qa.models import Citation, Hit


class LLMError(RuntimeError):
    """A provider or network failure. The message never contains keys or prompts."""


@dataclass(frozen=True, slots=True)
class Generation:
    """The raw output of one provider call."""

    text: str  # with "[n]" markers, n = source_index + 1
    citations: list[Citation]
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class LLMProvider(Protocol):
    name: ProviderName
    model: str

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        """Answer `question` from `sources` (the text in force at `at_date`), with citations."""
        ...

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        """A plain completion without sources, e.g. for the evaluation judge."""
        ...
