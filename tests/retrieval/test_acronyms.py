"""Acronyms of procurement practice expanded with the code's wording before searching."""

from datetime import date

import psycopg
import pytest

from appalti_qa.config import Settings
from appalti_qa.retrieval import FakeEmbedder, search
from appalti_qa.retrieval.acronyms import ACRONYMS, expand_acronyms


def test_first_occurrence_gets_the_code_wording() -> None:
    assert expand_acronyms("I compiti del DEC e i requisiti per essere DEC?") == (
        "I compiti del DEC (direttore dell'esecuzione del contratto) e i requisiti per essere DEC?"
    )


def test_only_whole_words_in_capitals() -> None:
    text = "Il DECRETO del 10 dec. e il dec; l'OEPV"
    assert expand_acronyms(text) == (
        "Il DECRETO del 10 dec. e il dec; l'OEPV (offerta economicamente piu' vantaggiosa)"
    )


def test_synonyms_share_one_expansion_and_spelled_out_terms_are_kept() -> None:
    assert expand_acronyms("RTI o ATI?") == "RTI (raggruppamento temporaneo) o ATI?"
    text = "Il DGUE, cioè il documento di gara unico europeo"
    assert expand_acronyms(text) == text
    assert expand_acronyms("Nessuna sigla qui.") == "Nessuna sigla qui."


def test_every_expansion_is_lower_case_wording() -> None:
    assert all(expansion == expansion.lower() for expansion in ACRONYMS.values())
    assert all(acronym[0].isupper() for acronym in ACRONYMS)


@pytest.mark.db
def test_search_finds_units_by_acronym(seeded_conn: psycopg.Connection, settings: Settings) -> None:
    """'SA' never occurs in the text; 'stazioni appaltanti' does (stemmed alike)."""
    hits = search(
        "SA",
        date(2025, 1, 1),
        5,
        "lexical",
        conn=seeded_conn,
        embedder=FakeEmbedder(),
        settings=settings,
    )
    assert {"1/1", "50/1"} <= {hit.unit_key for hit in hits}
