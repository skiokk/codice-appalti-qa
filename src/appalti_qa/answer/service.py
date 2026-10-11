"""Question answering: retrieve the units in force at a date, answer with citations or abstain."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from datetime import date

from appalti_qa.config import RetrievalMode, Settings, get_settings
from appalti_qa.llm import ABSTAIN_TOKEN, Generation, LLMProvider, get_provider
from appalti_qa.models import Answer, Citation, Hit

logger = logging.getLogger(__name__)

Retriever = Callable[[str, date, int], list[Hit]]  # (question, at_date, k)

ABSTAIN_MESSAGE = (
    "Le fonti disponibili non consentono di rispondere a questa domanda "
    "con riferimento alla data indicata."
)

_ABSTAIN_PREFIX = re.compile(rf"^\W*{ABSTAIN_TOKEN}")
_MARKER = re.compile(r"[ \t]*\[(\d+)\]")


def _default_retriever(settings: Settings, mode: RetrievalMode) -> Retriever:
    from appalti_qa.retrieval import search

    def retrieve(question: str, at_date: date, k: int) -> list[Hit]:
        return search(question, at_date, k, mode=mode, settings=settings)

    return retrieve


def _validated(question: str, settings: Settings) -> str:
    text = question.strip()
    if not text:
        raise ValueError("La domanda è vuota.")
    if len(text) > settings.max_question_chars:
        raise ValueError(
            f"La domanda supera il limite di {settings.max_question_chars} caratteri ({len(text)})."
        )
    return text


def _valid_citations(citations: Sequence[Citation], sources: Sequence[Hit]) -> list[Citation]:
    """Citations pointing at an actual source, with the unit key taken from that source."""
    return [
        citation.model_copy(update={"unit_key": sources[citation.source_index].unit_key})
        for citation in citations
        if citation.source_index < len(sources)
    ]


def _drop_dangling_markers(text: str, sources: Sequence[Hit]) -> str:
    """Remove '[n]' markers that do not point at a source."""
    return _MARKER.sub(lambda m: m.group() if 1 <= int(m.group(1)) <= len(sources) else "", text)


def answer_question(
    question: str,
    at_date: date,
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    retriever: Retriever | None = None,
    k: int | None = None,
    mode: RetrievalMode | None = None,
) -> Answer:
    """Answer `question` from the text in force at `at_date`.

    Raises ValueError (Italian message) for an empty or too long question and lets `LLMError`
    from the provider propagate. Abstains, keeping the retrieved sources, when nothing relevant is
    retrieved, when the model reports that the sources do not contain the answer, or when no
    citation of the answer points at a source.
    """
    settings = settings or get_settings()
    text = _validated(question, settings)
    provider = provider or get_provider(settings)
    retrieve = retriever or _default_retriever(settings, mode or settings.retrieval_mode)

    started = time.perf_counter()
    sources = retrieve(text, at_date, k or settings.default_top_k)
    if sources and max(hit.score for hit in sources) >= settings.min_relevance_score:
        generation = provider.generate(text, sources, at_date=at_date)
    else:  # nothing relevant to answer from: abstain without calling the model
        generation = Generation(ABSTAIN_TOKEN, [], provider.model, provider.name)
    latency_ms = round((time.perf_counter() - started) * 1000)

    citations = _valid_citations(generation.citations, sources)
    abstained = not citations or bool(_ABSTAIN_PREFIX.match(generation.text))
    answer = Answer(
        question=text,
        at_date=at_date,
        text=ABSTAIN_MESSAGE if abstained else _drop_dangling_markers(generation.text, sources),
        citations=[] if abstained else citations,
        sources=sources,
        abstained=abstained,
        model=generation.model,
        provider=generation.provider,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        latency_ms=latency_ms,
    )
    logger.info(
        "Answered at %s: provider=%s model=%s sources=%d cited=%s abstained=%s latency=%dms",
        at_date,
        answer.provider,
        answer.model,
        len(sources),
        answer.cited_keys,
        abstained,
        latency_ms,
    )
    return answer
