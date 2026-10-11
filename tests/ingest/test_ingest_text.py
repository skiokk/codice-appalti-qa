import hashlib

import pytest

from appalti_qa.ingest.akn import normalize_text, text_hash
from appalti_qa.ingest.text import (
    clean_heading,
    content_fingerprint,
    cut_notes,
    has_markers,
    is_note,
    is_repealed,
)


def test_normalize_text_drops_markers_and_blank_lines() -> None:
    raw = "  1. ((Le stazioni   appaltanti))  pubblicano: \n\n a)  ((uno));  \n"
    assert normalize_text(raw) == "1. Le stazioni appaltanti pubblicano:\na) uno;"


def test_text_hash_is_sha256_of_utf8() -> None:
    assert text_hash("attività") == hashlib.sha256("attività".encode()).hexdigest()


def test_has_markers() -> None:
    assert has_markers("testo ((inserito))")
    assert has_markers("fine inserimento))")
    assert not has_markers("testo (con parentesi)")


def test_fingerprint_ignores_layout_and_markers() -> None:
    flat = "2-bis. ((Le stazioni appaltanti)) indicano: a) uno; b) due."
    structured = "2 bis. Le stazioni appaltanti indicano:\na) uno;\nb) due."
    assert content_fingerprint(flat) == content_fingerprint(structured)
    assert content_fingerprint(flat) != content_fingerprint(flat.replace("due", "tre"))
    assert content_fingerprint(None) == content_fingerprint("")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\n  (( (Procedura di affidamento).))\n ", "Procedura di affidamento"),
        ("(Ulteriori disposizioni transitorie)", "Ulteriori disposizioni transitorie"),
        (" Principio del risultato. ", "Principio del risultato"),
        ("(Bandi) e (avvisi)", "(Bandi) e (avvisi)"),
        ("Autorita' nazionale anticorruzione (ANAC).", "Autorita' nazionale anticorruzione (ANAC)"),
        ("  ", None),
        (None, None),
    ],
)
def test_clean_heading(raw: str | None, expected: str | None) -> None:
    assert clean_heading(raw) == expected


@pytest.mark.parametrize(
    ("text", "repealed"),
    [
        ("3. ((COMMA ABROGATO DAL D.LGS. 31 DICEMBRE 2024, N. 209)).", True),
        ("ARTICOLO ABROGATO DAL D.LGS. 31 DICEMBRE 2024, N. 209", True),
        ("2-bis. COMMA SOPPRESSO DALLA L. 18 LUGLIO 2025, N. 105.", True),
        ("4. ((PERIODO SOPPRESSO DAL D.LGS. 31 DICEMBRE 2024, N. 209)). Le regioni", False),
        ("1. Il comma abrogato dal decreto resta applicabile.", False),
    ],
)
def test_is_repealed(text: str, repealed: bool) -> None:
    assert is_repealed(text) is repealed


def test_update_notes_are_cut() -> None:
    text = "8. Testo del comma. ------------ AGGIORNAMENTO (9) Il D.Lgs. 31 dicembre 2024 ..."
    assert cut_notes(text) == "8. Testo del comma."
    assert is_note("--------------\nAGGIORNAMENTO (14)\nLa L. 4 aprile 2025, n. 42, ...")
    assert not is_note("1. Testo con un trattino - e basta.")
