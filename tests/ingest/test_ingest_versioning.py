import logging
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
    ("expression", "previous", "logged"),
    [
        (None, D1, None),
        (D3, D1, None),  # expression equal to the file date
        (D2, D1, "may apply earlier"),  # between: the file was not refined
        (D2, None, "may apply earlier"),
        (D1, D1, "stale expression date"),  # not after the previous snapshot
        (date(2023, 12, 1), D1, "stale expression date"),
        (date(2024, 7, 1), D1, "after the file date"),
    ],
)
def test_effective_date_is_the_file_date(
    expression: date | None, previous: date | None, logged: str | None, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="appalti_qa.ingest.versioning"):
        assert effective_date(snapshot(D3, [], expression=expression), previous) == D3
    if logged is None:
        assert caplog.records == []
    else:
        assert logged in caplog.text


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


def test_heading_change_alone_updates_the_heading() -> None:
    before = comma("1/1", "1. Testo.", heading="Vecchia rubrica")
    after = comma("1/1", "1. Testo.", heading="Nuova rubrica")
    corpus = build_versions([snapshot(D1, [before]), snapshot(D2, [after])])
    assert versions_of(corpus, "1/1") == [(D1, None, None, False)]
    assert corpus.versions[0].heading == "Nuova rubrica"
    assert [s.vigenza_date for s, _ in corpus.snapshots] == [D1, D2]  # still a distinct text


def test_quotes_and_accents_are_layout() -> None:
    before = comma("1/1", "1. L'art. «uno» e` vigente.")
    after = comma("1/1", '1. L\u2019art. "uno" e\' vigente.')
    corpus = build_versions([snapshot(D1, [before]), snapshot(D2, [after])])
    assert versions_of(corpus, "1/1") == [(D1, None, None, False)]


def test_punctuation_only_changes_are_editorial() -> None:
    before = comma("1/1", "1. Le stazioni appaltanti, ove possibile, provvedono")
    after = comma("1/1", "1. Le stazioni appaltanti ove possibile provvedono;")
    amended = comma("1/1", "1. Le stazioni appaltanti provvedono;")
    corpus = build_versions(
        [
            snapshot(D1, [before]),
            snapshot(D2, [after], events=[ORIGINAL, amendment(date(2024, 2, 1), 4)]),
            snapshot(D3, [amended], events=[ORIGINAL, amendment(D3, 6)]),
        ]
    )
    assert [(v.valid_from, v.change_kind) for v in corpus.versions] == [
        (D1, "original"),
        (D2, "editorial"),
        (D3, "amendment"),
    ]
    assert [v[2] for v in versions_of(corpus, "1/1")] == [None, None, "D.L. n. 6"]


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


def test_commi_split_differently_are_editorial() -> None:
    """Art. 62 in the flat exports: the first sentence of comma 15 ends up in comma 14."""
    flat = [
        comma(
            "62/14", "14. Le stazioni operano insieme. 15. Se la procedura e' congiunta in parte"
        ),
        comma("62/15", "rispondono per quella parte. Le stazioni scelgono il risultato."),
    ]
    structured = [
        comma("62/14", "14. Le stazioni operano insieme."),
        comma(
            "62/15",
            "15. Se la procedura e' congiunta in parte rispondono per quella parte.\n"
            "Le stazioni scelgono il risultato.",
        ),
        comma("62/16", "16. Comma nuovo."),
    ]
    amended = [*structured[:2], comma("62/16", "16. Comma nuovo e modificato.")]
    corpus = build_versions(
        [
            snapshot(D1, flat),
            snapshot(D2, structured[:2], events=[ORIGINAL, amendment(D2, 25)]),
            snapshot(D3, structured, events=[ORIGINAL, amendment(D2, 25)]),
            snapshot(D4, amended, events=[ORIGINAL, amendment(D4, 40)]),
        ]
    )
    kinds = [(v.unit_key, v.valid_from, v.change_kind) for v in corpus.versions]
    assert kinds == [
        ("62/14", D1, "original"),
        ("62/14", D2, "editorial"),
        ("62/15", D1, "original"),
        ("62/15", D2, "editorial"),
        ("62/16", D3, "amendment"),  # new words: a real insertion
        ("62/16", D4, "amendment"),
    ]
    assert [v.changed_by is None for v in corpus.versions] == [True] * 4 + [False] * 2


def test_files_with_the_same_text_are_skipped() -> None:
    """Exports that differ only in metadata or layout add no snapshot and no version."""
    original = snapshot(D1, [comma("1/1", "1. Uno: a) x; b) y.")], sha256="a" * 64)
    restamped = snapshot(
        D2,
        [comma("1/1", "1. Uno:\na) x;\nb) y.")],
        sha256="b" * 64,
        events=[ORIGINAL, amendment(date(2024, 2, 1), 2)],
    )
    changed = snapshot(D3, [comma("1/1", "1. Due.")], events=[ORIGINAL, amendment(D3, 3)])
    corpus = build_versions([changed, restamped, original])
    assert [(s.vigenza_date, effective) for s, effective in corpus.snapshots] == [
        (D1, D1),
        (D3, D3),
    ]
    assert versions_of(corpus, "1/1") == [(D1, D3, None, False), (D3, None, "D.L. n. 3", False)]
    assert corpus.versions[0].text == "1. Uno:\na) x;\nb) y."  # latest rendering kept
    assert len(corpus.events) == 3  # events of skipped files are kept


def test_distinct_exports_sharing_an_expression_date(caplog) -> None:
    """Normattiva left 2023-07-06 as the expression date of both 20230919 and 20230929."""
    stale = date(2023, 7, 6)
    l87 = amendment(date(2023, 7, 5), 87)
    dl124, dl132 = amendment(date(2023, 9, 19), 124), amendment(date(2023, 9, 29), 132)
    first = snapshot(
        date(2023, 7, 6),
        [comma("1/1", "1. Uno."), comma("1/2", "2. Due.")],
        expression=stale,
        events=[ORIGINAL, l87],
    )
    second = snapshot(
        date(2023, 9, 20),
        [comma("1/1", "1. Uno modificato."), comma("1/2", "2. Due.")],
        expression=stale,
        events=[ORIGINAL, l87, dl124],
    )
    third = snapshot(
        date(2023, 9, 30),
        [comma("1/1", "1. Uno modificato."), comma("1/2", "2. Due modificato.")],
        expression=stale,
        events=[ORIGINAL, l87, dl124, dl132],
    )
    with caplog.at_level(logging.INFO, logger="appalti_qa.ingest.versioning"):
        corpus = build_versions([first, second, third])
    assert [effective for _, effective in corpus.snapshots] == [
        date(2023, 7, 6),
        date(2023, 9, 20),
        date(2023, 9, 30),
    ]
    assert versions_of(corpus, "1/1")[-1] == (date(2023, 9, 20), None, "D.L. n. 124", False)
    assert versions_of(corpus, "1/2")[-1] == (date(2023, 9, 30), None, "D.L. n. 132", False)
    assert caplog.text.count("stale expression date 2023-07-06") == 2


def test_changed_by_is_the_latest_act_on_or_before_valid_from(caplog) -> None:
    published, effective = date(2026, 3, 23), date(2026, 4, 7)  # L. 34/2026, vacatio legis
    law = amendment(published, 34)
    corpus = build_versions(
        [
            snapshot(D1, [comma("1/1", "1. Uno.")]),
            snapshot(effective, [comma("1/1", "1. Due.")], events=[ORIGINAL, law]),
        ]
    )
    assert versions_of(corpus, "1/1")[-1] == (effective, None, "D.L. n. 34", False)

    later = snapshot(date(2026, 5, 1), [comma("1/1", "1. Tre.")], events=[ORIGINAL, law])
    with caplog.at_level(logging.WARNING, logger="appalti_qa.ingest.versioning"):
        corpus = build_versions([*(s for s, _ in corpus.snapshots), later])
    assert versions_of(corpus, "1/1")[-1][2] == "D.L. n. 34"
    assert "no amending act published between 2026-04-07 and 2026-05-01" in caplog.text


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
    # The structured 20250401 export says what the flat 20241231 one says: not a new snapshot.
    assert effective == [
        (date(2024, 3, 2), date(2024, 3, 2)),
        (date(2024, 12, 31), date(2024, 12, 31)),
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
    kinds = {v.change_kind for v in fixture_corpus.versions}
    assert kinds == {"original", "amendment"}
    assert all(
        (v.change_kind == "amendment") == (v.changed_by is not None)
        for v in fixture_corpus.versions
    )


def test_fixture_versions_do_not_overlap(fixture_corpus: VersionedCorpus) -> None:
    by_key: dict[str, list] = {}
    for version in fixture_corpus.versions:
        by_key.setdefault(version.unit_key, []).append(version)
    assert by_key.keys() == {unit.key for unit in fixture_corpus.units}
    for versions in by_key.values():
        for earlier, later in pairwise(versions):
            assert earlier.valid_to is not None and earlier.valid_to <= later.valid_from
        assert all(v.valid_to is None or v.valid_to > v.valid_from for v in versions)
