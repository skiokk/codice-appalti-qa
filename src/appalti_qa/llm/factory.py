"""Provider selection from settings."""

from __future__ import annotations

import logging

from appalti_qa.config import ProviderName, Settings, get_settings
from appalti_qa.llm.anthropic_provider import AnthropicProvider
from appalti_qa.llm.base import LLMProvider
from appalti_qa.llm.fake import FakeProvider
from appalti_qa.llm.groq_provider import GroqProvider

logger = logging.getLogger(__name__)


def get_provider(
    settings: Settings | None = None,
    *,
    provider: ProviderName | None = None,
    model: str | None = None,
) -> LLMProvider:
    """The requested provider (default `settings.llm_provider`), or the fake one without its key.

    Default models: anthropic -> `settings.llm_model`, groq -> `settings.groq_model`, fake ->
    "fake". A model requested for a provider that falls back to fake is ignored.
    """
    settings = settings or get_settings()
    requested = provider or settings.llm_provider
    effective = settings.effective_provider(provider)
    if effective == "anthropic":
        return AnthropicProvider(settings, model=model)
    if effective == "groq":
        return GroqProvider(settings, model=model)
    if requested != "fake":
        logger.warning("No API key for provider %r: answering with the fake provider", requested)
        return FakeProvider()
    return FakeProvider(model or "fake")
