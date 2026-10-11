"""Deterministic offline provider, used by tests and whenever no API key is configured."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from appalti_qa.config import ProviderName
from appalti_qa.llm.base import Generation
from appalti_qa.llm.prompts import ABSTAIN_TOKEN, format_date, question_prompt
from appalti_qa.llm.spans import first_sentence
from appalti_qa.models import Citation, Hit

_QUOTED_SOURCES = 2


def _word_count(*texts: str) -> int:
    return sum(len(text.split()) for text in texts)


class FakeProvider:
    """Quotes the first sentence of the top one or two sources; abstains without sources.

    Token counts are word counts, so downstream aggregation can be tested deterministically.
    """

    name: ProviderName = "fake"

    def __init__(self, model: str = "fake") -> None:
        self.model = model

    def generate(self, question: str, sources: Sequence[Hit], *, at_date: date) -> Generation:
        prompt = question_prompt(question, at_date)
        if not sources:
            return self._generation(ABSTAIN_TOKEN, [], prompt)
        parts = [f"Nel testo vigente al {format_date(at_date)}:"]
        citations = []
        for index, hit in enumerate(sources[:_QUOTED_SOURCES]):
            quote = first_sentence(hit.text)
            if not quote:
                continue
            parts.append(f"{hit.label}: «{quote}» [{index + 1}]")
            citations.append(Citation(unit_key=hit.unit_key, cited_text=quote, source_index=index))
        if not citations:
            return self._generation(ABSTAIN_TOKEN, [], prompt)
        source_text = " ".join(hit.text for hit in sources)
        return self._generation(" ".join(parts), citations, prompt, source_text)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 1024) -> Generation:
        return self._generation('{"fake": true}', [], system, prompt)

    def _generation(self, text: str, citations: list[Citation], *inputs: str) -> Generation:
        return Generation(
            text=text,
            citations=citations,
            model=self.model,
            provider=self.name,
            input_tokens=_word_count(*inputs),
            output_tokens=_word_count(text),
            stop_reason="end_turn",
        )
