"""Synthetic dataset, stub retriever and stub answers for the evaluation tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pytest

from appalti_qa.evaluation.dataset import EvalItem, load_dataset
from appalti_qa.models import Answer, Citation, Hit

DATASET_YAML = """\
# Synthetic evaluation set for the tests.
# Header comments must survive an import.

- id: Q001
  question: "Fino a che importo posso affidare direttamente un servizio?"
  kind: answerable
  category: affidamento_diretto
  vigenza: "2025-01-01"
  expected_answer: "L'affidamento diretto di servizi e forniture è ammesso sotto i 140.000 euro."
  expected_refs: ["50/1/b"]
  source_excerpt: "b) affidamento diretto dei servizi e forniture"
  difficulty: easy
  status: verificata
  notes: ""

# --- temporal questions ---
- id: Q002
  question: "Nel 2024 l'affidamento diretto comprendeva i servizi di ingegneria?"
  kind: temporal
  category: affidamento_diretto
  vigenza: "2024-06-01"
  expected_answer: "No: prima del correttivo la lettera b) non citava i servizi di ingegneria."
  expected_refs: ["50/1/b"]
  source_excerpt: "affidamento diretto dei servizi e forniture di importo inferiore a 140.000 euro"
  difficulty: medium
  status: corretta
  notes: "Changed by D.Lgs. 209/2024."

- id: Q003
  question: "Chi firma la determina a contrarre nella mia ASL?"
  kind: unanswerable
  category: rup
  vigenza: "2025-01-01"
  expected_answer: "Il codice non lo stabilisce: dipende dall'organizzazione interna."
  expected_refs: []
  source_excerpt: ""
  difficulty: easy
  status: bozza
  notes: "Internal organisation."

- id: Q004
  question: "Quali principi guidano gli affidamenti sotto soglia?"
  kind: multi
  category: principi
  vigenza: "2025-01-01"
  expected_answer: "Il principio del risultato e le procedure dell'articolo 50 sotto soglia."
  expected_refs: ["1/1", "50/1"]
  source_excerpt: "perseguono il risultato dell'affidamento del contratto"
  difficulty: hard
  status: bozza
  notes: ""
"""

# Ranked unit keys returned by the stub retriever, by question id.
RANKINGS: dict[str, list[str]] = {
    "Q001": ["50/2", "50/1", "1/1"],
    "Q002": ["1/1", "2/1"],
    "Q004": ["1/1", "3/1", "50/1"],
}


def _make_hit(key: str, text: str = "Testo della disposizione.") -> Hit:
    return Hit(
        unit_key=key,
        version_id=1,
        text=text,
        score=1.0,
        valid_from=date(2023, 4, 1),
        normattiva_url="https://www.normattiva.it/",
    )


def _make_answer(
    item: EvalItem,
    text: str,
    cited: Sequence[str] = (),
    *,
    abstained: bool = False,
    latency_ms: int = 100,
    provider: str = "fake",
    model: str = "fake",
    tokens: tuple[int, int] = (0, 0),
) -> Answer:
    return Answer(
        question=item.question,
        at_date=item.vigenza,
        text=text,
        citations=[
            Citation(unit_key=key, cited_text="Testo", source_index=i)
            for i, key in enumerate(cited)
        ],
        sources=[_make_hit(key) for key in cited],
        abstained=abstained,
        model=model,
        provider=provider,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        latency_ms=latency_ms,
    )


@pytest.fixture
def make_hit() -> Callable[..., Hit]:
    return _make_hit


@pytest.fixture
def make_answer() -> Callable[..., Answer]:
    return _make_answer


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(DATASET_YAML, encoding="utf-8")
    return path


@pytest.fixture
def items(dataset_path: Path) -> list[EvalItem]:
    return load_dataset(dataset_path)


@pytest.fixture
def stub_retriever(items: list[EvalItem]) -> Callable[[str, date, int], list[Hit]]:
    by_question = {item.question: RANKINGS.get(item.id, []) for item in items}

    def retrieve(question: str, at_date: date, k: int) -> list[Hit]:
        return [_make_hit(key) for key in by_question[question][:k]]

    return retrieve


@pytest.fixture
def stub_answer(items: list[EvalItem]) -> Callable[[str, date], Answer]:
    """Q001 right, Q002 wrong citation, Q003 abstains, Q004 cites one of two refs."""
    by_question = {item.question: item for item in items}

    def answer(question: str, at_date: date) -> Answer:
        item = by_question[question]
        if item.id == "Q001":
            return _make_answer(
                item,
                "L'affidamento diretto di servizi e forniture è ammesso sotto i 140.000 euro [1].",
                ["50/1"],
                latency_ms=100,
                provider="anthropic",
                model="claude-opus-5-5",
                tokens=(1000, 200),
            )
        if item.id == "Q002":
            return _make_answer(
                item, "Si applica il principio del risultato [1].", ["1/1"], latency_ms=200
            )
        if item.id == "Q003":
            return _make_answer(item, "Non trovato.", abstained=True, latency_ms=300)
        return _make_answer(item, "Vale quanto previsto dal codice [1].", ["1/1"], latency_ms=400)

    return answer
