from appalti_qa.ingest.numbering import layout_lettere, split_numbered


def labels(text: str, *, restart: bool = False) -> list[str]:
    return [segment.label for segment in split_numbered(text, restart=restart)[1]]


def test_split_commi_after_heading() -> None:
    preamble, segments = split_numbered(
        "Principio del risultato. 1. Primo comma. 2. Secondo comma. 2-bis. Terzo comma."
    )
    assert preamble == "Principio del risultato. "
    assert [(s.label, s.body.strip()) for s in segments] == [
        ("1", "Primo comma."),
        ("2", "Secondo comma."),
        ("2-bis", "Terzo comma."),
    ]


def test_first_comma_may_follow_a_heading_without_punctuation() -> None:
    assert labels("Governance dei servizi 1. La Cabina di regia. 2. Altro.") == ["1", "2"]


def test_references_are_not_split() -> None:
    text = (
        "1. Si applica il comma 2. Le stazioni appaltanti, di cui all'allegato II.1. ricevono "
        "le offerte entro il 31 dicembre 2023. 2. Secondo comma, lettere c), d) ed e)."
    )
    _, segments = split_numbered(text)
    assert [s.label for s in segments] == ["1", "2"]
    assert "comma 2. Le stazioni" in segments[0].body


def test_numbering_must_continue() -> None:
    assert labels("1. Uno. 3. Non e' un comma. 2. Due.") == ["1", "2"]
    assert labels("Nessun comma numerato.") == []


def test_quoted_text_is_not_split() -> None:
    text = (
        "1. L'articolo 120 e' sostituito dal seguente: «Art. 120. 1. Gli atti. "
        "2. Per l'impugnazione.». 2. Entrata in vigore."
    )
    _, segments = split_numbered(text)
    assert [s.label for s in segments] == ["1", "2"]
    assert segments[1].body.strip() == "Entrata in vigore."


def test_suffixed_label_without_dot() -> None:
    assert labels("1. Il monitoraggio. 1-bis Con decreto del Ministro. 2. Altro.") == [
        "1",
        "1-bis",
        "2",
    ]


def test_inserted_comma_keeps_its_opening_marker() -> None:
    _, segments = split_numbered("1. Testo. ((2. Comma inserito.)) 3. Ultimo.")
    assert segments[1].label == "2"
    assert segments[1].body.startswith("((")


def test_misnumbered_comma_takes_the_expected_number() -> None:
    # Flat exports number some commi "1." by mistake (art. 62: "7. ... 1. ... 9.").
    _, segments = split_numbered("1. Uno. 2. Due. 1. Tre. 4. Quattro.")
    assert [s.label for s in segments] == ["1", "2", "3", "4"]
    assert segments[2].body.strip() == "Tre."


def test_only_the_last_copy_of_a_replaced_article_is_kept() -> None:
    old = "1. Le stazioni appaltanti qualificate sono iscritte nell'elenco. 2. Vecchio testo."
    new = "1. Le stazioni appaltanti qualificate sono iscritte nell'elenco ANAC. 2. Nuovo. 3. Fine."
    preamble, segments = split_numbered(f"Qualificazione. {old} {new}")
    assert preamble == "Qualificazione. "
    assert [s.label for s in segments] == ["1", "2", "3"]
    assert segments[1].body.strip() == "Nuovo."


def test_sub_items_after_a_colon_stay_in_their_comma() -> None:
    text = "1. Uno. 2. La soglia e': 1. decrementata; 2. incrementata. 3. Tre."
    _, segments = split_numbered(text)
    assert [s.label for s in segments] == ["1", "2", "3"]
    assert "1. decrementata; 2. incrementata." in segments[1].body


def test_restart_starts_new_blocks_after_part_titles() -> None:
    text = (
        "PARTE I 1. Nome. 2. Tipo. B - AVVISI (articolo 81) 1. Nome. 2. Codici CPV. "
        "SETTORI SPECIALI 1. Indirizzo."
    )
    assert labels(text, restart=True) == ["1", "2", "1", "2", "1"]
    assert labels(text) == ["1", "2"]


def test_layout_lettere() -> None:
    text = "1. Le stazioni appaltanti: a) uno; a-bis) uno bis; b) due, lettere c), d) ed e)."
    assert layout_lettere(text) == (
        "1. Le stazioni appaltanti:\na) uno;\na-bis) uno bis;\nb) due, lettere c), d) ed e)."
    )


def test_layout_lettere_follows_the_italian_alphabet() -> None:
    items = "; ".join(f"{letter}) voce" for letter in "abcdefghilmnopqrstuvz")
    assert layout_lettere(f"1. Elenco: {items}.").count("\n") == 21


def test_layout_lettere_restarts_after_a_colon() -> None:
    text = "1. Prima lista: a) uno; b) due. Seconda lista: a) tre; b) quattro."
    assert layout_lettere(text).count("\n") == 4


def test_layout_lettere_ignores_quotes_and_lone_letters() -> None:
    assert layout_lettere("1. Testo: «a) citato; b) citato». Fine.") == (
        "1. Testo: «a) citato; b) citato». Fine."
    )
    assert layout_lettere("1. Si veda la lettera b) del comma 2.") == (
        "1. Si veda la lettera b) del comma 2."
    )


def test_trailing_run_renumbered_from_one_continues_the_article() -> None:
    """Art. 34 in the flat exports: commi 5 and 6 are numbered "1." and "2."."""
    text = (
        "1. Le stazioni appaltanti possono chiedere un catalogo elettronico. "
        "2. I cataloghi sono predisposti dagli offerenti. 3. Quando i cataloghi sono accettati "
        "lo indicano nel bando. 4. Quando un accordo quadro e' concluso: a) invitano gli "
        "offerenti; b) comunicano agli offerenti. 1. In caso di riapertura del confronto, "
        "indicano la data della raccolta. 2. Possono aggiudicare appalti basati su un sistema "
        "dinamico di acquisizione."
    )
    assert labels(text) == ["1", "2", "3", "4"]  # the default keeps them inside comma 4
    _, segments = split_numbered(text, renumber_tail=True)
    assert [s.label for s in segments] == ["1", "2", "3", "4", "5", "6"]
    assert segments[4].body.strip().startswith("In caso di riapertura")
    assert segments[3].body.strip().endswith("b) comunicano agli offerenti.")


def test_trailing_run_needs_two_labels_and_no_gaps() -> None:
    lone = "1. Uno. 2. Due. 1. Ripete."
    assert labels(lone) == ["1", "2"]
    gapped = "1. Uno. 2. Due. 1. Ripete. 3. Salta."
    _, segments = split_numbered(gapped, renumber_tail=True)
    assert [s.label for s in segments] == ["1", "2", "3"]
    assert "1. Ripete." in segments[1].body
