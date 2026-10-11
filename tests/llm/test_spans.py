from appalti_qa.llm.spans import best_sentence, first_sentence, sentences

TEXT = (
    "1. Le stazioni appaltanti, ai sensi dell'art. 50 del D.Lgs. 36/2023 e del D.L. n. 51, "
    "procedono all'affidamento:\n"
    "a) affidamento diretto per lavori di importo inferiore a 150.000 euro;\n"
    "b-bis) procedura negoziata per servizi. Restano ferme le altre disposizioni."
)


def test_sentences_skip_enumerators_and_abbreviations() -> None:
    assert sentences(TEXT) == [
        "Le stazioni appaltanti, ai sensi dell'art. 50 del D.Lgs. 36/2023 e del D.L. n. 51, "
        "procedono all'affidamento:",
        "affidamento diretto per lavori di importo inferiore a 150.000 euro;",
        "procedura negoziata per servizi.",
        "Restano ferme le altre disposizioni.",
    ]
    assert all(sentence in TEXT for sentence in sentences(TEXT))


def test_sentences_of_blank_text() -> None:
    assert sentences("  \n ") == []


def test_first_sentence_is_cut_at_a_word_boundary() -> None:
    sentence = first_sentence(TEXT, max_chars=40)
    assert sentence == "Le stazioni appaltanti, ai sensi"
    assert sentence in TEXT
    assert first_sentence("") == ""


def test_best_sentence_maximises_word_overlap() -> None:
    claim = "Per i lavori sotto i 150.000 euro si procede con affidamento diretto"
    assert best_sentence(claim, TEXT) == (
        "affidamento diretto per lavori di importo inferiore a 150.000 euro;"
    )


def test_best_sentence_defaults_to_the_first_sentence() -> None:
    assert best_sentence("", TEXT) == sentences(TEXT)[0]
