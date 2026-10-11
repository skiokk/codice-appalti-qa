"""Passage splitting, max-sim aggregation and hub scores (no database)."""

import re
from itertools import pairwise

import numpy as np
import pytest

from appalti_qa.retrieval import passage_text
from appalti_qa.retrieval.passages import (
    ELLIPSIS,
    INTRO_MAX_CHARS,
    SPLIT_MIN_CHARS,
    WINDOW_CHARS,
    WINDOW_OVERLAP,
    clip,
    hub_scores,
    max_sim,
    split_passages,
    windows,
)

HEADING = "Procedure per l'affidamento"
INTRO = "1. Le stazioni appaltanti procedono all'affidamento con le seguenti modalita':"
LETTERE = [
    "a) affidamento diretto per lavori di importo inferiore a 150.000 euro;",
    "b) affidamento diretto dei servizi e forniture di importo inferiore a 140.000 euro;",
    "c) procedura negoziata senza bando per i lavori fino a 1 milione di euro.",
]
COMMA = "\n".join([INTRO, *LETTERE])


def _sentences(count: int, length: int = 90) -> list[str]:
    return [f"Frase {i:03d} " + "x" * (length - 11) + "." for i in range(count)]


def test_short_unit_without_lettere_is_one_passage() -> None:
    text = "2. Il responsabile unico del progetto e' nominato tra i dipendenti."
    assert split_passages(HEADING, text) == [passage_text(HEADING, text)]


def test_comma_with_lettere_gives_intro_and_one_passage_per_lettera() -> None:
    passages = split_passages(HEADING, COMMA)
    assert passages[0] == passage_text(HEADING, COMMA)  # the whole unit comes first
    assert passages[1] == f"{HEADING}\n{INTRO}"  # the intro alone
    assert passages[2:] == [f"{HEADING}\n{INTRO}\n{lettera}" for lettera in LETTERE]


def test_lettera_keeps_its_nested_lines_and_suffixed_labels() -> None:
    text = "\n".join(
        [INTRO, "a) prima lettera:", "1) primo punto;", "2) secondo punto;", "a-bis) inserita."]
    )
    passages = split_passages(None, text)
    assert passages[2] == f"{INTRO}\na) prima lettera:\n1) primo punto;\n2) secondo punto;"
    assert passages[3] == f"{INTRO}\na-bis) inserita."


def test_lettere_are_grouped_under_their_own_paragraph() -> None:
    text = "\n".join(
        ["1. Primo comma:", "a) uno;", "b) due.", "2. Secondo comma:", "a) tre;", "b) quattro."]
    )
    passages = split_passages(None, text)
    assert "1. Primo comma:\nb) due." in passages
    assert "2. Secondo comma:\na) tre;" in passages
    assert not any("Primo" in p and "tre;" in p for p in passages[1:])


def test_a_long_intro_is_clipped_in_the_lettera_passages() -> None:
    intro = "1. " + " ".join(["parola"] * 110) + ":"  # 774 characters: one window
    text = "\n".join([intro, *LETTERE[:2]])
    passages = split_passages(None, text)
    assert intro in passages  # the intro alone keeps all of it (windowed if long)
    lettera = passages[-1]
    assert lettera.endswith(LETTERE[1])
    assert ELLIPSIS in lettera and len(lettera) <= INTRO_MAX_CHARS + len(ELLIPSIS) + 100


def test_long_text_is_windowed_with_the_heading() -> None:
    text = "\n".join(_sentences(40))
    assert len(text) > SPLIT_MIN_CHARS
    passages = split_passages("Tabelle", text)
    assert passages[0] == passage_text("Tabelle", text)
    assert len(passages) > 3
    assert all(p.startswith("Tabelle\n") for p in passages)
    assert all(len(p) <= len("Tabelle\n") + WINDOW_CHARS for p in passages[1:])


def test_windows_cover_every_sentence_with_bounded_overlap() -> None:
    sentences = _sentences(50)
    text = " ".join(sentences)
    parts = windows(text)
    assert all(len(part) <= WINDOW_CHARS for part in parts)
    assert all(any(s in part for part in parts) for s in sentences)
    for first, second in pairwise(parts):
        shared = [s for s in sentences if s in first and s in second]
        assert shared, "consecutive windows overlap"
        assert sum(len(s) + 1 for s in shared) <= WINDOW_OVERLAP + 1
    assert windows(text) == parts  # deterministic


def test_windows_cut_a_sentence_longer_than_the_window_at_spaces() -> None:
    text = " ".join(["parola"] * 600)  # 4,199 characters, no sentence break
    parts = windows(text)
    assert len(parts) >= 5
    assert all(len(part) <= WINDOW_CHARS for part in parts)
    assert all(re.fullmatch(r"parola( parola)*", part) for part in parts)


def test_windows_of_short_or_empty_text() -> None:
    assert windows("  breve testo. ") == ["breve testo."]
    assert windows("") == []


def test_clip() -> None:
    assert clip("corto", 10) == "corto"
    assert clip("una frase un po' lunga", 10) == "una frase" + ELLIPSIS


def test_split_is_deterministic_and_has_no_duplicates() -> None:
    text = "\n".join([INTRO, LETTERE[0], LETTERE[0]])
    passages = split_passages(HEADING, text)
    assert len(passages) == len(set(passages))
    assert split_passages(HEADING, text) == passages


def test_max_sim_keeps_the_best_passage_of_each_version() -> None:
    scored = [(1, 0.80), (2, 0.85), (1, 0.90), (3, 0.70), (2, 0.60)]
    best = max_sim(scored)
    assert best == {1: 0.90, 2: 0.85, 3: 0.70}
    assert list(best) == [1, 2, 3]
    assert max_sim([]) == {}


def test_max_sim_ties_keep_first_seen_order() -> None:
    assert list(max_sim([(5, 0.5), (4, 0.5)])) == [5, 4]


def test_hub_scores_average_the_nearest_probes() -> None:
    probes = np.eye(3, dtype=np.float32)
    hub = np.array([[1.0, 1.0, 1.0]], dtype=np.float32) / np.sqrt(3)  # close to every probe
    loner = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # close to one probe only
    scores = hub_scores(np.vstack([hub, loner]), probes, k=2)
    assert scores[0] == pytest.approx(1 / np.sqrt(3))
    assert scores[1] == pytest.approx(0.5)  # mean of 1 and 0
    assert scores[0] > scores[1]
    assert hub_scores(np.vstack([hub, loner]), probes, k=10)[1] == pytest.approx(1 / 3)
    assert len(hub_scores(np.zeros((0, 3), dtype=np.float32), probes)) == 0
