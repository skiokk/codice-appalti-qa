import difflib
import logging
from pathlib import Path

import pytest
import yaml

from appalti_qa.evaluation.dataset import EvalItem, load_dataset
from appalti_qa.evaluation.yaml_writer import patch_dataset_text, write_dataset, yaml_value


def _diff(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


def test_unchanged_items_keep_the_file_identical(dataset_path: Path, items: list[EvalItem]) -> None:
    text = dataset_path.read_text(encoding="utf-8")
    assert patch_dataset_text(text, items) == text


def test_only_changed_lines_are_rewritten(dataset_path: Path, items: list[EvalItem]) -> None:
    before = dataset_path.read_text(encoding="utf-8")
    items[1] = items[1].model_copy(
        update={
            "status": "verificata",
            "notes": 'Rivista: "ok"\nseconda riga',
            "expected_refs": ["50/1/b", "50/1/a"],
        }
    )
    write_dataset(dataset_path, items)
    after = dataset_path.read_text(encoding="utf-8")
    diff = _diff(before, after)
    assert [line for line in diff if line.startswith("-")] == [
        '-  expected_refs: ["50/1/b"]',
        "-  status: corretta",
        '-  notes: "Changed by D.Lgs. 209/2024."',
    ]
    assert [line for line in diff if line.startswith("+")] == [
        '+  expected_refs: ["50/1/b", "50/1/a"]',
        "+  status: verificata",
        '+  notes: "Rivista: \\"ok\\"\\nseconda riga"',
    ]
    assert load_dataset(dataset_path) == items
    assert "# --- temporal questions ---" in after


def test_missing_field_is_added(tmp_path: Path, items: list[EvalItem]) -> None:
    path = tmp_path / "dataset.yaml"
    text = '# header\n\n- id: Q003\n  question: "x"\n  kind: unanswerable\n  category: rup\n'
    text += '  vigenza: "2025-01-01"\n  expected_answer: "a"\n  difficulty: easy\n\n# trailing\n'
    path.write_text(text, encoding="utf-8")
    (item,) = load_dataset(path)
    write_dataset(path, [item.model_copy(update={"status": "verificata", "notes": "ok"})])
    after = path.read_text(encoding="utf-8")
    assert after.endswith('  difficulty: easy\n  status: verificata\n  notes: "ok"\n\n# trailing\n')


def test_unknown_layout_is_regenerated(
    tmp_path: Path, items: list[EvalItem], caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "dataset.yaml"
    flow = yaml.safe_dump([item.model_dump(mode="json") for item in items], default_flow_style=True)
    path.write_text("# kept header\n" + flow, encoding="utf-8")
    items[0] = items[0].model_copy(update={"status": "scartata"})
    with caplog.at_level(logging.WARNING):
        write_dataset(path, items)
    assert "layout not recognised" in caplog.text
    after = path.read_text(encoding="utf-8")
    assert after.startswith("# kept header\n\n- id: Q001\n  question: ")
    assert load_dataset(path) == items


def test_yaml_values() -> None:
    assert yaml_value("status", "bozza") == "bozza"
    assert yaml_value("category", "null") == '"null"'  # would load as None unquoted
    assert yaml_value("question", "Perché?") == '"Perché?"'
    assert yaml_value("expected_refs", ["all:I.3/p1"]) == '["all:I.3/p1"]'
