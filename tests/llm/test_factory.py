import logging

import pytest

from appalti_qa.config import Settings
from appalti_qa.llm import AnthropicProvider, FakeProvider, GroqProvider, get_provider

PLACEHOLDER = "test-placeholder-key"


def test_default_is_fake(settings: Settings) -> None:
    provider = get_provider(settings)
    assert isinstance(provider, FakeProvider)
    assert (provider.name, provider.model) == ("fake", "fake")


def test_falls_back_to_fake_without_key(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        provider = get_provider(settings, provider="anthropic", model="claude-sonnet-5-5")
    assert isinstance(provider, FakeProvider)
    assert provider.model == "fake"
    assert "No API key for provider 'anthropic'" in caplog.text


def test_anthropic_with_key() -> None:
    settings = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key=PLACEHOLDER)
    provider = get_provider(settings)
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-opus-5-5"
    assert get_provider(settings, model="claude-sonnet-5-5").model == "claude-sonnet-5-5"


def test_groq_with_key() -> None:
    settings = Settings(_env_file=None, groq_api_key=PLACEHOLDER)
    provider = get_provider(settings, provider="groq")
    assert isinstance(provider, GroqProvider)
    assert provider.model == "openai/gpt-oss-120b"


def test_fake_model_override(settings: Settings) -> None:
    assert get_provider(settings, provider="fake", model="fake-2").model == "fake-2"


def test_uses_global_settings_by_default() -> None:
    assert isinstance(get_provider(), FakeProvider)
