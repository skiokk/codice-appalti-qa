"""Token prices used to report the cost of an evaluation run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Anthropic first-party list prices, standard tier, without prompt-caching discounts. The
# server-side refusal fallback can answer with another model, so the models it may route to are
# priced too.
ANTHROPIC_PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(0.10, 0.50),  # prompts up to 100K tokens
    # fallback-eligible models
    "claude-fable-5-1": Price(10.00, 50.00),
    "claude-opus-5": Price(5.00, 25.00),
    "claude-opus-4-8": Price(5.00, 25.00),
    "claude-sonnet-5": Price(2.00, 10.00),
}
# The fake provider is free; Groq models are used on the free tier.
FREE_PROVIDERS: frozenset[str] = frozenset({"fake", "groq"})


def price_for(provider: str, model: str) -> Price | None:
    """The price of a model, or None when it is unknown."""
    if provider in FREE_PROVIDERS:
        return Price(0.0, 0.0)
    return ANTHROPIC_PRICES.get(model) if provider == "anthropic" else None


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of one call in USD, or None when the model has no known price."""
    price = price_for(provider, model)
    if price is None:
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1e6
