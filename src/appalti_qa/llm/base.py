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
class ModelUsage:
    """The tokens one model consumed in a call; input includes cache writes and reads."""

    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class Generation:
    """The raw output of one provider call.

    `model` produced the returned text. `input_tokens` and `output_tokens` cover the whole call:
    when a server-side fallback re-runs a declined request on another model, the call has several
    attempts, each billed at its own model's rate, and `iterations` lists them in order.
    """

    text: str  # with "[n]" markers, n = source_index + 1
    citations: list[Citation]
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    iterations: tuple[ModelUsage, ...] = ()  # per-attempt usage, when the provider reports it

    @property
    def model_usage(self) -> tuple[ModelUsage, ...]:
        """Tokens by model attempt: `iterations`, or the whole call on `model`."""
        return self.iterations or (ModelUsage(self.model, self.input_tokens, self.output_tokens),)


class LLMProvider(Protocol):
    name: ProviderName
    model: str

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        """Answer `question` from `sources` (the text in force at `at_date`), with citations."""
        ...

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        """A plain completion without sources, e.g. for the evaluation judge."""
        ...


class UsageMeter:
    """An `LLMProvider` that forwards every call and records the tokens each model consumed.

    Callers that only receive the final product of a call (an `Answer` carries one model and
    token totals) read the per-model breakdown here with `take`.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.name: ProviderName = provider.name
        self.model = provider.model
        self._usage: list[ModelUsage] = []

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        return self._record(self.provider.generate(question, sources, at_date=at_date))

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        return self._record(self.provider.complete(system, prompt, max_tokens=max_tokens))

    def take(self) -> list[ModelUsage]:
        """The usage recorded since the previous `take`, which starts a new count."""
        usage, self._usage = self._usage, []
        return usage

    def _record(self, generation: Generation) -> Generation:
        self._usage.extend(generation.model_usage)
        return generation
