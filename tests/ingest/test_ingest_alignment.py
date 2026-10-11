"""Comma numbering of flat articles decided against a structured export (arts. 34, 41, 62, 205).

The documents are minimal Akoma Ntoso exports built here: a flat one (one ``<p>`` per article,
as up to 2024) and a structured one (one ``<paragraph>`` per comma, as from 2025), with texts
shortened from the real articles.
"""

import logging
from datetime import date
from pathlib import Path

import pytest

from appalti_qa.ingest import ParsedSnapshot, build_versions, parse_snapshot
from appalti_qa.ingest.alignment import align_numbering

HEADER = """<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0"><act name="monovigente">
<meta><identification source="">
<FRBRWork>
  <FRBRthis value="/akn/it/act/decreto_legislativo/stato/2023-03-31/36/!main"/>
  <FRBRalias name="urn:nir" value="urn:nir:stato:decreto.legislativo:2023-03-31;36"/>
  <FRBRdate date="2023-03-31" name=""/>
</FRBRWork>
</identification>
<lifecycle source="">
  <eventRef date="2023-03-31" eId="eventRef_0" source="ro1"/>
  <eventRef date="2024-12-31" eId="eventRef_1" source="rp1"/>
</lifecycle>
<references source="">
  <original eId="ro1" href="/akn/it/act/DECRETO_LEGISLATIVO/stato/2023-03-31/36/!main"/>
  <passiveRef eId="rp1" href="/akn/it/act/DECRETO_LEGISLATIVO/stato/2024-12-31/209/!main"/>
</references>
</meta>
<body>{body}</body></act></akomaNtoso>
"""

ART_41 = {
    7: "Il progetto esecutivo sviluppa un livello di definizione degli elementi tale da "
    "individuarne compiutamente la funzione, i requisiti, la qualita' e il prezzo di elenco.",
    8: "Nel caso in cui motivate ragioni giustifichino l'affidamento disgiunto, il nuovo "
    "progettista accetta senza riserve l'attivita' progettuale svolta in precedenza.",
    9: "In caso di affidamento esterno di entrambi i livelli di progettazione, l'avvio della "
    "progettazione esecutiva e' condizionato alla determinazione delle stazioni appaltanti.",
    10: "Gli oneri della progettazione, delle indagini, delle ricerche e degli studi connessi "
    "gravano sulle disponibilita' finanziarie della stazione appaltante.",
    11: "Le spese strumentali, dovute anche a sopralluoghi, sono a carico delle risorse "
    "iscritte sui pertinenti capitoli dello stato di previsione del Ministero.",
    12: "La progettazione di servizi e forniture e' articolata in un unico livello ed e' "
    "predisposta dalle stazioni appaltanti mediante propri dipendenti in servizio.",
    13: "Per i contratti relativi a lavori, servizi e forniture, il costo del lavoro e' "
    "determinato annualmente, in apposite tabelle, dal Ministero del lavoro.",
}
ART_62 = {
    6: "Le stazioni appaltanti non qualificate procedono all'affidamento ricorrendo a una "
    "centrale di committenza qualificata o a una stazione appaltante qualificata.",
    7: "Le stazioni appaltanti qualificate possono effettuare ordini su strumenti di acquisto "
    "messi a disposizione dalle centrali di committenza qualificate.",
    8: "L'allegato II.4 puo' essere integrato con una disciplina specifica sul funzionamento e "
    "sugli ambiti di riferimento delle centrali di committenza.",
    9: "Il ricorso alla stazione appaltante qualificata o alla centrale di committenza "
    "qualificata e' formalizzato mediante un accordo ai sensi della legge n. 241 del 1990.",
    10: "Le stazioni appaltanti non qualificate, per gli acquisti di importo superiore alle "
    "soglie, ricorrono agli strumenti di acquisto messi a disposizione dalle centrali.",
}
# Art. 205 quotes a numbered text inside comma 5 in both shapes: the run is not a slip.
ART_205 = {
    4: "Il ricorso e' notificato all'amministrazione entro trenta giorni dalla comunicazione.",
    5: "All'articolo 120 del codice del processo amministrativo sono apportate le seguenti "
    "modificazioni. 1. Il comma 1 e' sostituito dal seguente. 2. Il comma 2 e' abrogato.",
}


def flat_article(number: str, heading: str, text: str) -> str:
    return (
        f'<article eId="art_{number}"><num>Art. {number}.</num><heading>{heading}</heading>'
        f"<paragraph><content><p>{text}</p></content></paragraph></article>"
    )


def structured_article(number: str, heading: str, commi: dict[int, str]) -> str:
    paragraphs = "".join(
        f'<paragraph eId="art_{number}__para_{n}"><num>{n}.</num><content><p>{text}</p>'
        "</content></paragraph>"
        for n, text in commi.items()
    )
    return (
        f'<article eId="art_{number}"><num>Articolo {number}. </num>'
        f"<heading>{heading}</heading>{paragraphs}</article>"
    )


def flat_text(commi: dict[int, str], labels: dict[int, int]) -> str:
    """Commi from 1, each numbered as in ``labels`` (the export's slips) or by its number."""
    filler = {n: f"Comma {n} dell'articolo, rimasto invariato." for n in range(1, min(commi))}
    return " ".join(f"{labels.get(n, n)}. {text}" for n, text in {**filler, **commi}.items())


def write(tmp_path: Path, day: str, articles: list[str]) -> Path:
    path = tmp_path / f"23G00044_{day}.xml"
    path.write_text(HEADER.format(body="".join(articles)), encoding="utf-8")
    return path


def full(commi: dict[int, str]) -> dict[int, str]:
    filler = {n: f"Comma {n} dell'articolo, rimasto invariato." for n in range(1, min(commi))}
    return {**filler, **commi}


@pytest.fixture
def snapshots(tmp_path: Path) -> list[ParsedSnapshot]:
    """Arts. 41 and 62 misnumbered in the flat export, numbered right in the structured one;
    art. 41, comma 13 also amended (marked) in the structured one."""
    amended_41 = {**ART_41, 13: ART_41[13].replace("il costo del", "il costo ((medio)) del")}
    flat = write(
        tmp_path,
        "20230401",
        [
            flat_article("41", "Livelli della progettazione", flat_text(ART_41, {9: 1, 10: 2})),
            flat_article("62", "Aggregazioni", flat_text(ART_62, {8: 1, 9: 2})),
            flat_article("205", "Ricorsi", flat_text(ART_205, {})),
        ],
    )
    structured = write(
        tmp_path,
        "20250315",
        [
            structured_article("41", "Livelli della progettazione", full(amended_41)),
            structured_article("62", "Aggregazioni", full(ART_62)),
            structured_article("205", "Ricorsi", full(ART_205)),
        ],
    )
    return [parse_snapshot(flat), parse_snapshot(structured)]


def keys(snapshot: ParsedSnapshot, article: str) -> list[str]:
    return [u.key for u in snapshot.units if u.article == article and u.allegato is None]


def test_flat_export_keeps_both_readings(snapshots: list[ParsedSnapshot]) -> None:
    flat = snapshots[0]
    assert keys(flat, "41")[-1] == "41/8"  # commi 9-13 merged into comma 8 as exported
    assert [u.key for u in flat.renumbered["41"]][-6:] == [f"41/{n}" for n in range(8, 14)]
    assert set(flat.renumbered) == {"41", "62", "205"}
    assert snapshots[1].renumbered == {}


def test_structured_numbering_is_adopted_when_the_texts_match(
    snapshots: list[ParsedSnapshot], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="appalti_qa.ingest.alignment"):
        aligned, decisions = align_numbering(snapshots)
    by_article = {d.article: d for d in decisions}
    assert by_article["41"].renumbered and by_article["62"].renumbered
    assert by_article["41"].keys == tuple(f"41/{n}" for n in range(9, 14))
    assert by_article["41"].reference == "23G00044_20250315.xml"
    assert by_article["41"].similarity >= 0.9
    assert keys(aligned[0], "41") == [f"41/{n}" for n in range(1, 14)]
    assert keys(aligned[0], "62") == [f"62/{n}" for n in range(1, 11)]
    units = {u.key: u for u in aligned[0].units}
    assert units["41/13"].text.startswith("13. Per i contratti relativi a lavori")
    assert units["41/8"].text.endswith("svolta in precedenza.")
    assert [u.ordinal for u in aligned[0].units] == list(range(1, len(aligned[0].units) + 1))
    assert aligned[0].renumbered == {}
    # Art. 205: the structured export has no comma 6, the run stays inside comma 5.
    assert not by_article["205"].renumbered
    assert keys(aligned[0], "205") == [f"205/{n}" for n in range(1, 6)]
    assert "art. 205 restarts its numbering" in caplog.text


def test_point_in_time_text_exists_from_the_first_snapshot(
    snapshots: list[ParsedSnapshot],
) -> None:
    corpus = build_versions(snapshots)
    versions = {
        key: [
            (v.valid_from, v.valid_to, v.change_kind) for v in corpus.versions if v.unit_key == key
        ]
        for key in ("41/9", "41/12", "41/13", "62/8", "62/10")
    }
    first, switch = date(2023, 4, 1), date(2025, 3, 15)
    assert versions["41/9"] == [(first, None, "original")]
    assert versions["41/12"] == [(first, None, "original")]
    assert versions["41/13"] == [(first, switch, "original"), (switch, None, "amendment")]
    assert versions["62/8"] == [(first, None, "original")]
    assert versions["62/10"] == [(first, None, "original")]
    assert [d.article for d in corpus.numbering if d.renumbered] == ["41", "62"]


def test_without_a_structured_export_the_numbering_is_kept(
    snapshots: list[ParsedSnapshot], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="appalti_qa.ingest.alignment"):
        aligned, decisions = align_numbering(snapshots[:1])
    assert not any(d.renumbered for d in decisions)
    assert all(d.reference is None for d in decisions)
    assert keys(aligned[0], "41")[-1] == "41/8"
    assert "no export has it structured" in caplog.text


def test_a_counterpart_with_other_text_does_not_confirm(tmp_path: Path) -> None:
    """The structured export has commi 9 and 10, but they say something else."""
    other = {**ART_41, 9: "Testo del tutto diverso sulle verifiche.", 10: "Altro testo ancora."}
    flat = write(tmp_path, "20230401", [flat_article("41", "L", flat_text(ART_41, {9: 1, 10: 2}))])
    structured = write(tmp_path, "20250315", [structured_article("41", "L", full(other))])
    aligned, (decision,) = align_numbering([parse_snapshot(flat), parse_snapshot(structured)])
    assert not decision.renumbered and decision.similarity < 0.5
    assert keys(aligned[0], "41")[-1] == "41/8"


def test_trailing_run_of_art_34(tmp_path: Path) -> None:
    commi = {
        4: "Quando un accordo quadro e' concluso con piu' operatori, le stazioni appaltanti "
        "invitano gli offerenti a ripresentare i cataloghi.",
        5: "In caso di riapertura del confronto competitivo le stazioni appaltanti indicano la "
        "data e l'ora della raccolta delle informazioni.",
        6: "Le stazioni appaltanti possono aggiudicare appalti basati su un sistema dinamico di "
        "acquisizione imponendo che le offerte siano presentate sotto forma di catalogo.",
    }
    flat = write(
        tmp_path, "20240302", [flat_article("34", "Cataloghi", flat_text(commi, {5: 1, 6: 2}))]
    )
    structured = write(tmp_path, "20250315", [structured_article("34", "Cataloghi", full(commi))])
    aligned, (decision,) = align_numbering([parse_snapshot(flat), parse_snapshot(structured)])
    assert decision.renumbered and decision.keys == ("34/5", "34/6")
    assert keys(aligned[0], "34") == [f"34/{n}" for n in range(1, 7)]
