import json
from datetime import date

from appalti_qa.llm import ABSTAIN_TOKEN, FakeProvider
from appalti_qa.models import Hit


def test_abstains_without_sources(at_date: date) -> None:
    generation = FakeProvider().generate("Domanda?", [], at_date=at_date)
    assert generation.text == ABSTAIN_TOKEN
    assert generation.citations == []
    assert (generation.provider, generation.model) == ("fake", "fake")


def test_quotes_the_first_two_sources(hits: list[Hit], at_date: date) -> None:
    generation = FakeProvider().generate("Quali procedure?", hits, at_date=at_date)
    assert generation.text.startswith("Nel testo vigente al 11/10/2026:")
    assert "[1]" in generation.text and "[2]" in generation.text and "[3]" not in generation.text
    assert [c.unit_key for c in generation.citations] == ["50/1", "1/1"]
    assert [c.source_index for c in generation.citations] == [0, 1]
    for citation in generation.citations:
        assert citation.cited_text in hits[citation.source_index].text
        assert f"«{citation.cited_text}»" in generation.text
    assert generation.input_tokens > 0 and generation.output_tokens > 0


def test_is_deterministic(hits: list[Hit], at_date: date) -> None:
    provider = FakeProvider()
    assert provider.generate("x", hits, at_date=at_date) == provider.generate(
        "x", hits, at_date=at_date
    )


def test_complete_returns_json() -> None:
    generation = FakeProvider(model="fake-judge").complete("sistema", "valuta")
    assert json.loads(generation.text) == {"fake": True}
    assert generation.model == "fake-judge"
