"""Hand-written sources shared by the provider tests (abridged texts, not authoritative)."""

from __future__ import annotations

from datetime import date

import pytest

from appalti_qa.models import Event, Hit

AMENDMENT = Event(
    date=date(2024, 12, 31),
    amending_urn="/akn/it/act/DECRETO_LEGISLATIVO/stato/2024-12-31/209/!main",
    label="D.Lgs. 31 dicembre 2024, n. 209",
)


@pytest.fixture
def at_date() -> date:
    return date(2026, 10, 11)


@pytest.fixture
def hits() -> list[Hit]:
    return [
        Hit(
            unit_key="50/1",
            version_id=2,
            heading="Procedure per l'affidamento",
            text=(
                "1. Le stazioni appaltanti procedono all'affidamento dei contratti sotto soglia:\n"
                "a) affidamento diretto per lavori di importo inferiore a 150.000 euro;\n"
                "b) affidamento diretto dei servizi e forniture, ivi compresi i servizi di "
                "ingegneria e architettura, di importo inferiore a 140.000 euro."
            ),
            score=0.9,
            valid_from=date(2024, 12, 31),
            normattiva_url="https://www.normattiva.it/uri-res/N2Ls?x~art50!vig=2026-10-11",
            modified_by=[AMENDMENT],
        ),
        Hit(
            unit_key="1/1",
            version_id=1,
            heading="Principio del risultato",
            text=(
                "1. Le stazioni appaltanti e gli enti concedenti perseguono il risultato "
                "dell'affidamento del contratto e della sua esecuzione con la massima "
                "tempestività e il migliore rapporto possibile tra qualità e prezzo."
            ),
            score=0.5,
            valid_from=date(2023, 4, 1),
            normattiva_url="https://www.normattiva.it/uri-res/N2Ls?x~art1!vig=2026-10-11",
        ),
        Hit(
            unit_key="all:I.3/p1",
            version_id=3,
            heading="Termini delle procedure di appalto e di concessione",
            text=(
                "1. Le gare di appalto si concludono nei seguenti termini massimi:\n"
                "a) procedura aperta: nove mesi;\nb) procedura ristretta: dieci mesi."
            ),
            score=0.2,
            valid_from=date(2023, 4, 1),
            normattiva_url="https://www.normattiva.it/uri-res/N2Ls?x!vig=2026-10-11",
        ),
    ]
