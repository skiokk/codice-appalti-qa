from datetime import date
from pathlib import Path

import psycopg
import pytest

from appalti_qa.evaluation.dataset import (
    DatasetError,
    EvalItem,
    load_dataset,
    referenced_texts,
    select_items,
    validate_dataset,
)


def _item(**overrides: object) -> EvalItem:
    fields = {
        "id": "Q100",
        "question": "Domanda?",
        "kind": "answerable",
        "category": "procedure",
        "vigenza": "2025-01-01",
        "expected_answer": "Risposta.",
        "expected_refs": ["50/1/b"],
        "source_excerpt": "affidamento diretto dei servizi e forniture",
        "difficulty": "easy",
    }
    return EvalItem.model_validate({**fields, **overrides})


def test_load_dataset(items: list[EvalItem]) -> None:
    assert [item.id for item in items] == ["Q001", "Q002", "Q003", "Q004"]
    first = items[0]
    assert first.vigenza == date(2025, 1, 1)
    assert first.expected_refs == ["50/1/b"]
    assert first.verified and items[1].verified and not items[2].verified
    assert validate_dataset(items) == []


def test_load_dataset_reports_every_invalid_entry(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "- id: Q1\n  question: x\n  kind: answerable\n  category: c\n  vigenza: '2025-01-01'\n"
        "  expected_answer: a\n  difficulty: easy\n"
        "- id: Q002\n  question: x\n  kind: maybe\n  category: c\n  vigenza: 'not a date'\n"
        "  expected_answer: a\n  difficulty: easy\n  extra: 1\n"
        "- just a string\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as error:
        load_dataset(path)
    problems = "\n".join(error.value.problems)
    assert "Q1 (entry 1): id:" in problems
    assert "Q002 (entry 2): kind:" in problems
    assert "Q002 (entry 2): vigenza:" in problems
    assert "Q002 (entry 2): extra:" in problems
    assert "entry 3: expected a mapping" in problems


@pytest.mark.parametrize(
    ("content", "message"),
    [("id: Q001\n", "expected a list"), ("- [unclosed\n", "invalid YAML")],
)
def test_load_dataset_rejects_bad_documents(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="file not found"):
        load_dataset(tmp_path / "missing.yaml")


def test_none_fields_become_empty() -> None:
    assert _item(notes=None, source_excerpt=None).notes == ""


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_refs": ["art50"]}, "invalid key 'art50'"),
        ({"expected_refs": ["50/1", "50/1"]}, "duplicate expected_refs"),
        ({"kind": "unanswerable"}, "must have no expected_refs"),
        ({"expected_refs": []}, "need at least one expected_ref"),
        ({"kind": "multi"}, "at least two expected_refs"),
        ({"source_excerpt": "x" * 301}, "301 characters"),
        ({"source_excerpt": ""}, "source_excerpt is empty"),
        ({"expected_answer": " "}, "expected_answer is empty"),
    ],
)
def test_static_checks(overrides: dict, message: str) -> None:
    problems = validate_dataset([_item(**overrides)])
    assert any(message in problem for problem in problems), problems


def test_duplicate_ids() -> None:
    assert validate_dataset([_item(), _item()]) == ["Q100: duplicate id"]


def test_select_items(items: list[EvalItem]) -> None:
    assert [i.id for i in select_items(items)] == ["Q001", "Q002"]
    assert [i.id for i in select_items(items, include_drafts=True, limit=3)] == [
        "Q001",
        "Q002",
        "Q003",
    ]
    assert [i.id for i in select_items(items, include_drafts=True, ids=["Q004"])] == ["Q004"]
    dropped = [items[0].model_copy(update={"status": "scartata"}), *items[1:]]
    assert "Q001" not in [i.id for i in select_items(dropped, include_drafts=True)]


@pytest.mark.db
def test_corpus_checks(seeded_conn: psycopg.Connection) -> None:
    valid = [
        _item(
            id="Q101",
            source_excerpt="ivi compresi i servizi di ingegneria e architettura",
        ),
        _item(
            id="Q102",
            kind="multi",
            expected_refs=["1/1", "all:I.3"],
            source_excerpt="procedura aperta: nove mesi",
        ),
    ]
    assert validate_dataset(valid, conn=seeded_conn) == []

    invalid = [
        # the amended wording is not in force yet in June 2024
        _item(
            id="Q103",
            vigenza="2024-06-01",
            source_excerpt="ivi compresi i servizi di ingegneria e architettura",
        ),
        _item(id="Q104", vigenza="2024-06-01", expected_refs=["225-bis"], source_excerpt="x"),
        _item(id="Q105", expected_refs=["99/1"], source_excerpt="x"),
    ]
    problems = validate_dataset(invalid, conn=seeded_conn)
    assert problems == [
        "Q103: source_excerpt not found in the text of ['50/1/b'] in force on 2024-06-01",
        "Q104: 225-bis matches no unit in force on 2024-06-01",
        "Q105: 99/1 matches no unit in force on 2025-01-01",
    ]


@pytest.mark.db
def test_excerpt_match_ignores_whitespace(seeded_conn: psycopg.Connection) -> None:
    item = _item(source_excerpt="sotto soglia:\n   a)   affidamento diretto per lavori")
    assert validate_dataset([item], conn=seeded_conn) == []


@pytest.mark.db
def test_referenced_texts(seeded_conn: psycopg.Connection) -> None:
    texts = referenced_texts([_item(expected_refs=["1/1", "50/1/b"])], conn=seeded_conn)
    text = texts["Q100"]
    assert text.startswith("[Art. 1, comma 1]\n1. Le stazioni appaltanti")
    assert "[Art. 50, comma 1]\n1. Le stazioni appaltanti procedono" in text
    assert "ingegneria e architettura" in text  # the version in force on 2025-01-01
