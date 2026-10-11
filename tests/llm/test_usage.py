"""Per-model usage of a generation and the UsageMeter wrapper."""

from __future__ import annotations

from datetime import date

from appalti_qa.llm import FakeProvider, Generation, ModelUsage, UsageMeter
from appalti_qa.models import Hit


def test_model_usage_defaults_to_the_whole_call() -> None:
    generation = Generation("x", [], "claude-opus-5-5", "anthropic", 100, 20)
    assert generation.model_usage == (ModelUsage("claude-opus-5-5", 100, 20),)
    attempts = (ModelUsage("claude-opus-5-5", 100, 5), ModelUsage("claude-opus-5", 100, 20))
    fell_back = Generation("x", [], "claude-opus-5", "anthropic", 200, 25, "end_turn", attempts)
    assert fell_back.model_usage == attempts


def test_usage_meter_records_every_call(hits: list[Hit], at_date: date) -> None:
    fake = FakeProvider(model="fake-model")
    meter = UsageMeter(fake)
    assert (meter.name, meter.model) == ("fake", "fake-model")

    generation = meter.generate("Qual è la soglia?", hits, at_date=at_date)
    assert generation == fake.generate("Qual è la soglia?", hits, at_date=at_date)
    judged = meter.complete("Sei un valutatore.", "Valuta.", max_tokens=64)
    assert meter.take() == [
        ModelUsage("fake-model", generation.input_tokens, generation.output_tokens),
        ModelUsage("fake-model", judged.input_tokens, judged.output_tokens),
    ]
    assert meter.take() == []  # a new count starts after each take
