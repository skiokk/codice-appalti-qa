from datetime import date

from appalti_qa.llm import ABSTAIN_TOKEN
from appalti_qa.llm.prompts import (
    NUMBERED_CITATION_RULES,
    SYSTEM_PROMPT,
    format_date,
    numbered_sources,
    question_prompt,
    source_context,
    source_title,
)
from appalti_qa.models import Event, Hit


def test_system_prompt_rules() -> None:
    assert ABSTAIN_TOKEN in SYSTEM_PROMPT
    assert "D.Lgs. 31 marzo 2023, n. 36" in SYSTEM_PROMPT
    assert "Non inventare numeri di articolo" in SYSTEM_PROMPT
    assert "atto modificativo" in SYSTEM_PROMPT
    assert "[1]" in NUMBERED_CITATION_RULES


def test_format_date() -> None:
    assert format_date(date(2026, 1, 5)) == "05/01/2026"


def test_source_title(hits: list[Hit], at_date: date) -> None:
    assert source_title(hits[0], at_date) == (
        "Art. 50, comma 1 — D.Lgs. 36/2023 (testo vigente al 11/10/2026)"
    )
    assert source_title(hits[2], at_date).startswith("Allegato I.3, punto 1 — ")


def test_source_context_lists_heading_validity_and_amendments(hits: list[Hit]) -> None:
    context = source_context(hits[0])
    assert "Rubrica: Procedure per l'affidamento" in context
    assert "Versione in vigore dal 31/12/2024" in context
    assert "D.Lgs. 31 dicembre 2024, n. 209 (dal 31/12/2024)" in context


def test_source_context_closed_interval_and_no_amendments(hits: list[Hit]) -> None:
    original = Event(
        date=date(2023, 3, 31),
        kind="original",
        amending_urn="/akn/it/act/DECRETO_LEGISLATIVO/stato/2023-03-31/36/!main",
        label="D.Lgs. 31 marzo 2023, n. 36",
    )
    hit = hits[1].model_copy(update={"valid_to": date(2024, 12, 31), "modified_by": [original]})
    context = source_context(hit)
    assert "Versione in vigore dal 01/04/2023 al 30/12/2024" in context  # valid_to is exclusive
    assert "Atti modificativi" not in context


def test_question_prompt_carries_date_and_question(at_date: date) -> None:
    prompt = question_prompt("Qual è la soglia?", at_date)
    assert "Data di riferimento: 11/10/2026" in prompt
    assert prompt.endswith("Domanda: Qual è la soglia?")


def test_numbered_sources(hits: list[Hit], at_date: date) -> None:
    rendered = numbered_sources(hits, at_date)
    assert "[1] Art. 50, comma 1 — " in rendered
    assert "[2] Art. 1, comma 1 — " in rendered
    assert "[3] Allegato I.3, punto 1 — " in rendered
    assert hits[0].text in rendered
