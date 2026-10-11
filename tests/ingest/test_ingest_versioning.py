from dataclasses import replace
from datetime import date
from itertools import pairwise

import pytest

from appalti_qa.ingest import ParsedSnapshot, ParsedUnit, VersionedCorpus, build_versions
from appalti_qa.ingest.akn import ActInfo
from appalti_qa.ingest.text import text_hash
from appalti_qa.ingest.units import make_unit
from appalti_qa.ingest.versioning import effective_date
from appalti_qa.models import Event

ACT = ActInfo(
    code="23G00044",
    urn="urn:nir:stato:decreto.legislativo:2023-03-31;36",
    title="Codice dei contratti pubblici",
    short_title="D.Lgs. 36/2023",
    issued_on=date(2023, 3, 31),
)
ORIGINAL = Event(
    date=date(2023, 3, 31),
    kind="original",
    amending_urn="/akn/it/act/DECRETO_LEGISLATIVO/stato/2023-03-31/36/!main",
    label="D.Lgs. 31 marzo 2023, n. 36",
)


def amendment(day: date, number: int) -> Event:
    return Event(
        date=day,
        amending_urn=f"/akn/it/act/DECRETO-LEGGE/stato/{day.isoformat()}/{number}/!main",
        label=f"D.L. n. {number}",
    )


def comma(key: str, text: str, heading: str | None = "Rubrica") -> ParsedUnit:
    article, number = key.split("/")
    return make_unit(key, "comma", text, article=article, comma=number, heading=heading)


def snapshot(
    vigenza: date,
    units: list[ParsedUnit],
    *,
    expression: date | None = None,
    events: list[Event] | None = None,
    sha256: str | None = None,
) -> ParsedSnapshot:
    return ParsedSnapshot(
        vigenza_date=vigenza,
        expression_date=expression,
        source_file=f"23G00044_{vigenza:%Y%m%d}.xml",
        sha256=sha256 or text_hash(f"{vigenza} {[u.text for u in units]}"),
        act=ACT,
        events=events if events is not None else [ORIGINAL],
        units=[replace(unit, ordinal=index) for index, unit in enumerate(units, start=1)],
    )


def versions_of(corpus: VersionedCorpus, key: str) -> list[tuple]:
    return [
        (v.valid_from, v.valid_to, v.changed_by.label if v.changed_by else None, v.repealed)
        for v in corpus.versions
        if v.unit_key == key
    ]


D1, D2, D3, D4 = date(2024, 1, 10), date(2024, 3, 1), date(2024, 6, 1), date(2024, 9, 1)


@pytest.mark.parametrize(
    ("expression", "previous", "expected"),
    [
        (None, D1, D3),
        (D3, D1, D3),  # expression equal to vigenza
        (D2, D1, D2),  # strictly between: the change applies from the expression date
        (D1, D1, D3),  # not after the previous vigenza
        (date(2023, 12, 1), D1, D3),
        (D2, None, D2),  # first snapshot
        (date(2024, 7, 1), D1, D3),  # after the vigenza
    ],
)
def test_effective_date(expression: date | None, previous: date | None, expected: date) -> None:
    assert effective_date(snapshot(D3, [], expression=expression), previous) == expected


def test_changes_open_and_close_versions() -> None:
    first = snapshot(D1, [comma("1/1", "1. Testo originale."), comma("1/2", "2. Secondo.")])
    second = snapshot(
        D2,
        [comma("1/1", "1. Testo ((modificato))."), comma("1/2", "2. Secondo.")],
        events=[ORIGINAL, amendment(date(2024, 2, 20), 5)],
    )
    corpus = build_versions([second, first])
    assert versions_of(corpus, "1/1") == [
        (D1, D2, None, False),
        (D2, None, "D.L. n. 5", False),
    ]
    assert versions_of(corpus, "1/2") == [(D1, None, None, False)]
    texts = [v.text for v in corpus.versions if v.unit_key == "1/1"]
    assert texts == ["1. Testo originale.", "1. Testo modificato."]
    assert all(v.text_hash == text_hash(v.text) for v in corpus.versions)


def test_layout_changes_do_not_create_versions() -> None:
    flat = comma("1/1", "1. Le stazioni appaltanti: a) uno; b) due.")
    structured = comma("1/1", "1. Le stazioni appaltanti:\na) uno;\nb) due.")
    corpus = build_versions([snapshot(D1, [flat]), snapshot(D2, [structured])])
    assert versions_of(corpus, "1/1") == [(D1, None, None, False)]
    assert corpus.versions[0].text == structured.text  # latest rendering
    assert corpus.versions[0].text_hash == text_hash(structured.text)


def test_heading_change_creates_a_version() -> None:
    before = comma("1/1", "1. Testo.", heading="Vecchia rubrica")
    after = comma("1/1", "1. Testo.", heading="Nuova rubrica")
    corpus = build_versions([snapshot(D1, [before]), snapshot(D2, [after])])
    assert [v.heading for v in corpus.versions] == ["Vecchia rubrica", "Nuova rubrica"]


def test_missing_units_are_closed_and_may_reappear() -> None:
    unit = comma("1/2", "2. Comma.")
    corpus = build_versions(
        [
            snapshot(D1, [comma("1/1", "1. Uno."), unit]),
            snapshot(D2, [comma("1/1", "1. Uno.")]),
            snapshot(D3, [comma("1/1", "1. Uno."), unit]),
        ]
    )
    assert [(v[0], v[1]) for v in versions_of(corpus, "1/2")] == [(D1, D2), (D3, None)]


def test_identical_files_are_skipped_and_bound_the_expression_date() -> None:
    units = [comma("1/1", "1. Uno.")]
    original = snapshot(D1, units, sha256="a" * 64)
    duplicate = snapshot(D2, units, sha256="a" * 64)
    # The text changed after D2 (the duplicate still had the old one), so an expression date
    # before D2 cannot be the effective date.
    changed = snapshot(D3, [comma("1/1", "1. Due.")], expression=date(2024, 2, 1))
    corpus = build_versions([changed, duplicate, original])
    assert [(s.vigenza_date, effective) for s, effective in corpus.snapshots] == [
        (D1, D1),
        (D3, D3),
    ]
    later = snapshot(D4, [comma("1/1", "1. Tre.")], expression=date(2024, 8, 1))
    corpus = build_versions([original, duplicate, later])
    assert corpus.snapshots[-1][1] == date(2024, 8, 1)


def test_changed_by_uses_the_lifecycle_of_the_introducing_snapshot() -> None:
    older, newer = amendment(date(2024, 2, 1), 7), amendment(D2, 9)
    first = snapshot(D1, [comma("1/1", "1. Uno.")])
    # The file requested on D2 does not include the act published that day yet.
    second = snapshot(D2, [comma("1/1", "1. Due.")], events=[ORIGINAL, older])
    third = snapshot(D3, [comma("1/1", "1. Tre.")], events=[ORIGINAL, older, newer])
    corpus = build_versions([first, second, third])
    assert [v[2] for v in versions_of(corpus, "1/1")] == [None, "D.L. n. 7", "D.L. n. 9"]
    assert [event.label for event in corpus.events] == [
        "D.Lgs. 31 marzo 2023, n. 36",
        "D.L. n. 7",
        "D.L. n. 9",
    ]


def test_changed_by_prefers_the_event_on_the_effective_date() -> None:
    same_day, before = amendment(D2, 3), amendment(date(2024, 2, 1), 2)
    corpus = build_versions(
        [
            snapshot(D1, [comma("1/1", "1. Uno.")]),
            snapshot(D2, [comma("1/1", "1. Due.")], events=[ORIGINAL, same_day, before]),
        ]
    )
    assert versions_of(corpus, "1/1")[-1][2] == "D.L. n. 3"


def test_removed_keys_keep_their_place_in_document_order() -> None:
    corpus = build_versions(
        [
            snapshot(D1, [comma("1/1", "1. A."), comma("2/1", "1. B."), comma("3/1", "1. C.")]),
            snapshot(D2, [comma("1/1", "1. A."), comma("3/1", "1. C."), comma("4/1", "1. D.")]),
        ]
    )
    assert [(u.key, u.ordinal) for u in corpus.units] == [
        ("1/1", 1),
        ("2/1", 2),
        ("3/1", 3),
        ("4/1", 4),
    ]
    assert [v.unit_key for v in corpus.versions] == ["1/1", "2/1", "3/1", "4/1"]


def test_empty_input() -> None:
    with pytest.raises(ValueError):
        build_versions([])


def test_fixture_corpus(fixture_corpus: VersionedCorpus) -> None:
    effective = [(s.vigenza_date, day) for s, day in fixture_corpus.snapshots]
    assert effective == [
        (date(2024, 3, 2), date(2024, 3, 2)),
        (date(2024, 12, 31), date(2024, 12, 31)),
        (date(2025, 4, 1), date(2025, 4, 1)),
    ]
    correttivo = "D.Lgs. 31 dicembre 2024, n. 209"
    # Unchanged across the flat -> structured switch of 2025: one version.
    assert versions_of(fixture_corpus, "50/1") == [(date(2024, 3, 2), None, None, False)]
    assert versions_of(fixture_corpus, "50/3") == [
        (date(2024, 3, 2), date(2024, 12, 31), None, False),
        (date(2024, 12, 31), None, correttivo, True),
    ]
    assert versions_of(fixture_corpus, "109/1") == [
        (date(2024, 3, 2), date(2024, 12, 31), None, False)
    ]
    assert versions_of(fixture_corpus, "109") == [(date(2024, 12, 31), None, correttivo, True)]
    assert versions_of(fixture_corpus, "225-bis/1") == [
        (date(2024, 12, 31), None, correttivo, False)
    ]
    assert len(versions_of(fixture_corpus, "all:I.3/p1")) == 2
    assert len(fixture_corpus.events) == 12
    assert fixture_corpus.act.short_title == "D.Lgs. 36/2023"


def test_fixture_versions_do_not_overlap(fixture_corpus: VersionedCorpus) -> None:
    by_key: dict[str, list] = {}
    for version in fixture_corpus.versions:
        by_key.setdefault(version.unit_key, []).append(version)
    assert by_key.keys() == {unit.key for unit in fixture_corpus.units}
    for versions in by_key.values():
        for earlier, later in pairwise(versions):
            assert earlier.valid_to is not None and earlier.valid_to <= later.valid_from
        assert all(v.valid_to is None or v.valid_to > v.valid_from for v in versions)
