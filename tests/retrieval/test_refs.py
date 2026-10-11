import pytest

from appalti_qa.retrieval import extract_refs


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("art. 50, comma 1, lett. b)", ["50/1/b"]),
        ("Art. 50 comma 1 lettera b)", ["50/1/b"]),
        ("art. 50 c. 1 lett. b", ["50/1/b"]),
        ("articolo 225-bis", ["225-bis"]),
        ("art. 225 bis", ["225-bis"]),
        ("art. 225bis", ["225-bis"]),
        ("comma 1-bis dell'articolo 5", ["5/1-bis"]),
        ("artt. 50 e 51", ["50", "51"]),
        ("articoli 50, 51 e 52", ["50", "51", "52"]),
        ("art. 50, commi 1 e 2", ["50/1", "50/2"]),
        ("art. 50, comma 1, lettere a) e b)", ["50/1/a", "50/1/b"]),
        ("art. 11 comma 2 lett. e) ed f)", ["11/2/e", "11/2/f"]),
        ("comma 2 dell'articolo 17", ["17/2"]),
        ("lettera b) del comma 1 dell'art. 50", ["50/1/b"]),
        ("comma 1, lettera b), dell\u2019art. 50", ["50/1/b"]),
        ("allegato I.1", ["all:I.1"]),
        ("all. i.3", ["all:I.3"]),
        ("allegato II.12, art. 2, comma 1", ["all:II.12/2/1"]),
        ("art. 2 dell'allegato I.1", ["all:I.1/2"]),
        ("l'art. 41, allegato I.7", ["41", "all:I.7"]),
        ("art. 50 del d.lgs. 36/2023", ["50"]),
        ("art. 50 del decreto legislativo 31 marzo 2023, n. 36", ["50"]),
    ],
)
def test_extracts_canonical_keys(text: str, expected: list[str]) -> None:
    assert extract_refs(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Qual è la soglia per l'affidamento diretto di servizi?",
        "art. 80 del d.lgs. 50/2016",
        "art. 1 della legge 241/1990",
        "art. 120 c.p.a.",
        "art. 1337 c.c.",
        "art. 3 del codice dell\u2019amministrazione digitale",
        "il comma 2 prevede un termine",
    ],
)
def test_ignores_text_without_refs_to_the_code(text: str) -> None:
    assert extract_refs(text) == []


def test_lettera_d_invito_is_not_a_lettera() -> None:
    assert extract_refs("la lettera d'invito del comma 1 dell'art. 50") == ["50/1"]


def test_refs_are_distinct_and_in_text_order() -> None:
    text = (
        "Confronta l'art. 50, comma 1, lett. b), del d.lgs. 50/2016 con l'art. 50, comma 1 "
        "del codice, l'art. 17 e di nuovo l'art. 50, comma 1."
    )
    assert extract_refs(text) == ["50/1", "17"]
