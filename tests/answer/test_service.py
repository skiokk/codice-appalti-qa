"""answer_question with stub retrievers and providers (no database, no network)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from appalti_qa.answer import ABSTAIN_MESSAGE, answer_question
from appalti_qa.config import ProviderName, Settings
from appalti_qa.llm import ABSTAIN_TOKEN, FakeProvider, Generation, LLMError
from appalti_qa.models import Citation, Hit

AT = date(2026, 10, 11)


def hit(key: str, score: float, text: str) -> Hit:
    return Hit(
        unit_key=key,
        version_id=1,
        text=text,
        score=score,
        valid_from=date(2023, 4, 1),
        normattiva_url="https://www.normattiva.it/",
    )


HITS = [
    hit("50/1", 0.8, "1. Le stazioni appaltanti procedono all'affidamento diretto."),
    hit("1/1", 0.4, "1. Le stazioni appaltanti perseguono il risultato."),
]


class Retriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, date, int]] = []

    def __call__(self, question: str, at_date: date, k: int) -> list[Hit]:
        self.calls.append((question, at_date, k))
        return self.hits


class ScriptedProvider:
    """Returns a fixed generation and records calls."""

    name: ProviderName = "anthropic"
    model = "scripted"

    def __init__(self, generation: Generation | None = None) -> None:
        self.generation = generation
        self.calls = 0

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        self.calls += 1
        if self.generation is None:
            raise AssertionError("the provider should not be called")
        return self.generation

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        raise NotImplementedError


def scripted(text: str, *citations: Citation, **fields: Any) -> ScriptedProvider:
    return ScriptedProvider(
        Generation(text, list(citations), "scripted-model", "anthropic", **fields)
    )


def test_answers_with_citations(settings: Settings) -> None:
    retriever = Retriever(HITS)
    answer = answer_question(
        "  Come si affida?  ", AT, settings, provider=FakeProvider(), retriever=retriever
    )
    assert retriever.calls == [("Come si affida?", AT, settings.default_top_k)]
    assert answer.question == "Come si affida?"
    assert not answer.abstained
    assert answer.cited_keys == ["50/1", "1/1"]
    assert "[1]" in answer.text and "[2]" in answer.text
    assert answer.sources == HITS
    assert (answer.provider, answer.model) == ("fake", "fake")
    assert answer.input_tokens > 0 and answer.output_tokens > 0
    assert answer.latency_ms >= 0


def test_k_override(settings: Settings) -> None:
    retriever = Retriever(HITS)
    answer_question("Domanda", AT, settings, provider=FakeProvider(), retriever=retriever, k=3)
    assert retriever.calls[0][2] == 3


@pytest.mark.parametrize("question", ["", "   \n "])
def test_empty_question_is_rejected(settings: Settings, question: str) -> None:
    with pytest.raises(ValueError, match="vuota"):
        answer_question(question, AT, settings, provider=FakeProvider(), retriever=Retriever([]))


def test_long_question_is_rejected() -> None:
    settings = Settings(_env_file=None, max_question_chars=10)
    with pytest.raises(ValueError, match="limite di 10 caratteri"):
        answer_question("x" * 11, AT, settings, provider=FakeProvider(), retriever=Retriever([]))


def test_no_hits_abstains_without_calling_the_model(settings: Settings) -> None:
    provider = ScriptedProvider()
    answer = answer_question("Domanda", AT, settings, provider=provider, retriever=Retriever([]))
    assert provider.calls == 0
    assert answer.abstained
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == [] and answer.sources == []
    assert (answer.provider, answer.model) == ("anthropic", "scripted")
    assert (answer.input_tokens, answer.output_tokens) == (0, 0)


def test_low_relevance_abstains_and_keeps_sources() -> None:
    settings = Settings(_env_file=None, min_relevance_score=0.9)
    provider = ScriptedProvider()
    answer = answer_question("Domanda", AT, settings, provider=provider, retriever=Retriever(HITS))
    assert provider.calls == 0
    assert answer.abstained
    assert answer.sources == HITS


@pytest.mark.parametrize("text", [ABSTAIN_TOKEN, f"**{ABSTAIN_TOKEN}**", f"{ABSTAIN_TOKEN}."])
def test_model_abstention(settings: Settings, text: str) -> None:
    citation = Citation(unit_key="50/1", cited_text="Le stazioni", source_index=0)
    provider = scripted(text, citation, input_tokens=100, output_tokens=3)
    answer = answer_question("Domanda", AT, settings, provider=provider, retriever=Retriever(HITS))
    assert answer.abstained
    assert answer.text == ABSTAIN_MESSAGE
    assert answer.citations == []
    assert answer.sources == HITS
    assert (answer.model, answer.input_tokens, answer.output_tokens) == ("scripted-model", 100, 3)


def test_answer_without_valid_citations_abstains(settings: Settings) -> None:
    out_of_range = Citation(unit_key="99/1", cited_text="x", source_index=5)
    provider = scripted("Una risposta [6].", out_of_range)
    answer = answer_question("Domanda", AT, settings, provider=provider, retriever=Retriever(HITS))
    assert answer.abstained
    assert answer.citations == []


def test_invalid_citations_and_markers_are_dropped(settings: Settings) -> None:
    valid = Citation(unit_key="wrong-key", cited_text="Le stazioni appaltanti", source_index=1)
    invalid = Citation(unit_key="99/1", cited_text="x", source_index=4)
    provider = scripted("Prima affermazione [2]. Seconda [5].", valid, invalid)
    answer = answer_question("Domanda", AT, settings, provider=provider, retriever=Retriever(HITS))
    assert not answer.abstained
    assert answer.text == "Prima affermazione [2]. Seconda."
    assert [(c.unit_key, c.source_index) for c in answer.citations] == [("1/1", 1)]


def test_provider_errors_propagate(settings: Settings) -> None:
    class Failing(ScriptedProvider):
        def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
            raise LLMError("Anthropic rate limit exceeded (HTTP 429)")

    with pytest.raises(LLMError):
        answer_question("Domanda", AT, settings, provider=Failing(), retriever=Retriever(HITS))


def test_latency_covers_retrieval_and_generation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr("appalti_qa.answer.service.time.perf_counter", lambda: next(ticks))
    answer = answer_question(
        "Domanda", AT, settings, provider=FakeProvider(), retriever=Retriever(HITS)
    )
    assert answer.latency_ms == 250


def test_default_retriever_uses_search(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_search(query: str, at_date: date, k: int = 8, mode: str = "hybrid", **kw: Any):
        calls.append({"query": query, "at_date": at_date, "k": k, "mode": mode, **kw})
        return HITS

    monkeypatch.setattr("appalti_qa.retrieval.search", fake_search, raising=False)
    answer_question("Domanda", AT, settings, provider=FakeProvider())
    answer_question("Domanda", AT, settings, provider=FakeProvider(), mode="lexical", k=2)
    assert [(c["k"], c["mode"]) for c in calls] == [
        (settings.default_top_k, settings.retrieval_mode),
        (2, "lexical"),
    ]
    assert calls[0]["settings"] is settings


def test_default_provider_is_fake_without_keys(settings: Settings) -> None:
    answer = answer_question("Domanda", AT, settings, retriever=Retriever(HITS))
    assert answer.provider == "fake"
    assert not answer.abstained
