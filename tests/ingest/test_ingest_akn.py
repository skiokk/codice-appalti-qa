import hashlib
from datetime import date
from pathlib import Path

import pytest

from appalti_qa.ingest import ParsedSnapshot, parse_snapshot
from appalti_qa.ingest.annexes import doc_identity, slugify
from appalti_qa.ingest.text import content_fingerprint
from appalti_qa.keys import is_valid_key

CODE_KEYS_2025 = [
    *(f"1/{n}" for n in range(1, 5)),
    "6/1",
    "50/1",
    "50/2",
    "50/2-bis",
    *(f"50/{n}" for n in range(3, 10)),
    "109",
    *(f"225-bis/{n}" for n in range(1, 6)),
]
ANNEX_KEYS_2025 = [
    *(f"all:I.3/p{n}" for n in range(1, 6)),
    "all:I.4/1/1",
    "all:I.4/1/2",
    "all:I.4/tabella-a",
    *(f"all:II.2-bis/1/{n}" for n in range(1, 5)),
]


def units_by_key(snapshot: ParsedSnapshot) -> dict:
    return {unit.key: unit for unit in snapshot.units}


@pytest.fixture(scope="module")
def structured(fixture_snapshots: list[ParsedSnapshot]) -> ParsedSnapshot:
    return fixture_snapshots[2]


def test_snapshot_metadata(structured: ParsedSnapshot, fixture_paths: list[Path]) -> None:
    assert structured.vigenza_date == date(2025, 4, 1)
    assert structured.expression_date == date(2024, 12, 31)
    assert structured.source_file == "23G00044_20250401.xml"
    assert structured.sha256 == hashlib.sha256(fixture_paths[2].read_bytes()).hexdigest()
    act = structured.act
    assert (act.code, act.urn, act.short_title, act.issued_on) == (
        "23G00044",
        "urn:nir:stato:decreto.legislativo:2023-03-31;36",
        "D.Lgs. 36/2023",
        date(2023, 3, 31),
    )
    assert act.title.startswith("Codice dei contratti pubblici in attuazione")
    assert "23G00044" not in act.title


def test_events_from_lifecycle(structured: ParsedSnapshot) -> None:
    events = structured.events
    assert len(events) == 12
    assert events[0].kind == "original"
    assert events[0].label == "D.Lgs. 31 marzo 2023, n. 36"
    assert all(event.kind == "amendment" for event in events[1:])
    assert [event.date for event in events] == sorted(event.date for event in events)
    labels = {event.label for event in events}
    assert {"D.L. 10 maggio 2023, n. 51", "D.Lgs. 31 dicembre 2024, n. 209"} <= labels
    assert events[-1].label == "D.L. 14 marzo 2025, n. 25"
    assert events[-1].amending_urn == "/akn/it/act/DECRETO-LEGGE/stato/2025-03-14/25/!main"


def test_keys_and_document_order(structured: ParsedSnapshot) -> None:
    assert [unit.key for unit in structured.units] == CODE_KEYS_2025 + ANNEX_KEYS_2025
    assert [unit.ordinal for unit in structured.units] == list(range(1, len(structured.units) + 1))
    assert all(is_valid_key(unit.key) for unit in structured.units)


def test_comma_text_layout(structured: ParsedSnapshot) -> None:
    lines = units_by_key(structured)["50/1"].text.split("\n")
    assert lines[0].startswith("1. Salvo quanto previsto dagli articoli 62 e 63")
    assert [line[:3] for line in lines[1:]] == ["a) ", "b) ", "c) ", "d) ", "e) "]
    assert lines[1].startswith("a) affidamento diretto per lavori di importo inferiore")


def test_markers_are_stripped_and_recorded(structured: ParsedSnapshot) -> None:
    units = units_by_key(structured)
    assert not any("((" in unit.text or "))" in unit.text for unit in structured.units)
    inserted = units["50/2-bis"]
    assert inserted.amended
    assert inserted.text.startswith("2-bis. Le stazioni appaltanti pubblicano sul proprio sito")
    assert not units["50/1"].amended


def test_repealed_units(structured: ParsedSnapshot) -> None:
    units = units_by_key(structured)
    assert units["50/3"].repealed
    assert units["50/3"].text == "3. COMMA ABROGATO DAL D.LGS. 31 DICEMBRE 2024, N. 209."
    article = units["109"]
    assert (article.kind, article.article, article.comma, article.repealed) == (
        "article",
        "109",
        None,
        True,
    )
    assert not units["50/4"].repealed


def test_headings(fixture_snapshots: list[ParsedSnapshot]) -> None:
    for snapshot in fixture_snapshots:
        units = units_by_key(snapshot)
        assert units["1/1"].heading == "Principio del risultato"  # flat: heading inside the text
        assert units["6/1"].heading == (
            "Principi di solidarieta' e di sussidiarieta' orizzontale. "
            "Rapporti con gli enti del Terzo settore"
        )
        assert units["50/1"].heading == "Procedure per l'affidamento"
        assert units["all:I.3/p1"].heading == "Termini delle procedure di appalto e di concessione"
        # Allegato I.4, art. 1 has no heading: the allegato title stands in.
        assert units["all:I.4/1/1"].heading == (
            "Imposta di bollo relativa alla stipulazione del contratto"
        )
    units = units_by_key(fixture_snapshots[2])
    assert units["225-bis/1"].heading == "Ulteriori disposizioni transitorie"
    assert units["all:II.2-bis/1/1"].heading == "Ambito di applicazione"


def test_annex_units(structured: ParsedSnapshot) -> None:
    units = units_by_key(structured)
    block = units["all:I.3/p1"]
    assert (block.kind, block.allegato, block.article, block.comma) == (
        "allegato_block",
        "I.3",
        None,
        None,
    )
    assert block.text.startswith("(Articolo 17, commi 3 e 3-bis)\n1. A norma dell'articolo 17")
    assert "\na) procedura aperta: nove mesi;\n" in block.text
    comma = units["all:II.2-bis/1/2"]
    assert (comma.allegato, comma.article, comma.comma) == ("II.2-bis", "1", "2")
    table = units["all:I.4/tabella-a"]
    assert table.heading == "Tabella A"
    assert "≥ 40.000 < 150.000 | 40" in table.text.split("\n")
    assert not any("INDICE ALLEGATI" in unit.text for unit in structured.units)


def test_flat_and_structured_exports_agree(fixture_snapshots: list[ParsedSnapshot]) -> None:
    flat, structured = units_by_key(fixture_snapshots[1]), units_by_key(fixture_snapshots[2])
    assert flat.keys() == structured.keys()
    for key, unit in structured.items():
        assert content_fingerprint(flat[key].text) == content_fingerprint(unit.text), key
        assert flat[key].heading == unit.heading, key
    assert flat["50/1"].text == structured["50/1"].text


def test_flat_export_before_the_correttivo(fixture_snapshots: list[ParsedSnapshot]) -> None:
    units = units_by_key(fixture_snapshots[0])
    assert [key for key in units if key.startswith("109")] == ["109/1", "109/2", "109/3"]
    assert "50/2-bis" not in units
    assert not units["50/3"].repealed
    assert "all:II.2-bis/1/1" not in units


@pytest.mark.parametrize(
    ("name", "identity"),
    [
        ("Allegato I.1-art. 2", ("I.1", "art. 2")),
        ("Allegati - Allegato I.01 art. 1", ("I.01", "art. 1")),
        ("Allegato I.8 - art. 1", ("I.8", "art. 1")),
        ("Allegato I.7-art. 4 bis", ("I.7", "art. 4 bis")),
        ("Allegato I.3", ("I.3", "")),
        ("Allegato I.4-Tabella A", ("I.4", "Tabella A")),
        ("Allegato II.2-Allegato II.2 bis Tabella A", ("II.2-bis", "Tabella A")),
        ("Allegato II.6-Allegato II.6 bis art. 3", ("II.6-bis", "art. 3")),
        ("Allegati - Indice allegati", None),
    ],
)
def test_doc_identity(name: str, identity: tuple[str, str] | None) -> None:
    assert doc_identity(name) == identity


def test_slugify() -> None:
    assert slugify("Tabella A") == "tabella-a"
    assert slugify("Schede") == "schede"
    assert slugify("Tabella 1 - Opere") == "tabella-1-opere"


def test_invalid_files(tmp_path: Path) -> None:
    not_xml = tmp_path / "23G00044_20250101.xml"
    not_xml.write_text("<html><body>Errore</body>")
    with pytest.raises(ValueError, match="not well-formed"):
        parse_snapshot(not_xml)
    not_akn = tmp_path / "23G00044_20250102.xml"
    not_akn.write_text("<?xml version='1.0'?><root/>")
    with pytest.raises(ValueError, match="Akoma Ntoso"):
        parse_snapshot(not_akn)
    badly_named = tmp_path / "codice.xml"
    badly_named.write_bytes((Path(__file__).parent / "fixtures/23G00044_20250401.xml").read_bytes())
    with pytest.raises(ValueError, match="file name"):
        parse_snapshot(badly_named)
