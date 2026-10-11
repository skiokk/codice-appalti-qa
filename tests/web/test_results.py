import json
from pathlib import Path

import pytest

from appalti_qa.web.results import format_metric, load_results, metric_label, results_view


def test_missing_or_invalid_file(tmp_path: Path) -> None:
    assert load_results(tmp_path / "missing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("[1, 2]", encoding="utf-8")
    assert load_results(broken) is None


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("recall@5", 0.725, "72,5%"),
        ("mrr", 0.61234, "0,612"),
        ("latency_p95_ms", 2500, "2.500 ms"),
        ("input_tokens", 1234.4, "1.234"),
        ("correctness", None, "n/d"),
        ("abstained", True, "sì"),
        ("n", 12, "12"),
        ("article_mrr", 0.5, "0,500"),
        ("cost_usd", 0.01234, "0,0123 USD"),
        ("judge_output_tokens", 1500, "1.500"),
        ("errors", 0, "0"),
        ("cost_unknown", 3, "3"),
    ],
)
def test_format_metric(name: str, value: object, expected: str) -> None:
    assert format_metric(name, value) == expected


def test_metric_labels() -> None:
    assert metric_label("recall@10") == "Recall@10"
    assert metric_label("hit@1") == "Hit@1"
    assert metric_label("article_recall@5") == "Recall@5 per articolo"
    assert metric_label("citation_precision") == "Precisione delle citazioni"
    assert metric_label("cost_unknown") == "Risposte senza prezzo noto"
    assert metric_label("judge_cost_unknown") == "Valutazioni senza prezzo noto"
    assert metric_label("new_metric") == "New metric"


def test_partial_results_render() -> None:
    view = results_view({"run_id": "r1", "answer": {"overall": {"correctness": 0.5}}})
    assert view.run_id == "r1"
    assert view.dataset_size == "n/d"
    assert [section.title for section in view.sections] == ["Risposte"]
    assert view.sections[0].overall == [("Correttezza", "50,0%")]
    assert view.sections[0].breakdowns == []
    assert view.items == []


def test_breakdown_tables(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "overall": {"recall@1": 0.5, "mrr": 0.5},
                    "by_kind": {"answerable": {"recall@1": 1.0}, "multi": {"mrr": 0.25}},
                },
                "items": [{"id": "Q002", "retrieved": ["1", "2", "3", "4", "5", "6"]}],
            }
        ),
        encoding="utf-8",
    )
    view = load_results(path)
    assert view is not None
    [table] = view.sections[0].breakdowns
    assert table.columns == ["MRR"]  # only the headline metrics of each group
    assert table.rows == [("Con risposta nel testo", ["n/d"]), ("Con più riferimenti", ["0,250"])]
    assert view.items[0].retrieved == "1, 2, 3, 4, 5"
    assert view.items[0].cited == "n/d"  # not recorded, as opposed to an empty list


def test_retrieval_run_compares_modes() -> None:
    view = results_view(
        {
            "config": {"retrieval_mode": "hybrid", "modes": ["lexical", "hybrid"]},
            "retrieval": {"overall": {"recall@5": 0.75, "mrr": 0.5}},
            "retrieval_by_mode": {
                "lexical": {"overall": {"recall@5": 0.5, "mrr": 0.25}},
                "hybrid": {"overall": {"recall@5": 0.75, "mrr": 0.5}},
            },
            "items": [
                {
                    "id": "Q001",
                    "expected_refs": ["50/1/b"],
                    "by_mode": {"lexical": {"retrieved": ["1/1"]}, "hybrid": {"retrieved": []}},
                }
            ],
        }
    )
    [section] = view.sections
    table = section.breakdowns[0]
    assert table.group_header == "Ricerca"
    assert table.rows == [("Lessicale", ["50,0%", "0,250"]), ("Ibrida", ["75,0%", "0,500"])]
    assert view.items[0].retrieved == "nessuno"  # the primary (hybrid) ranking


def test_config_values() -> None:
    view = results_view(
        {
            "dataset": {"size": 120, "verified": 80, "evaluated": 80},
            "config": {
                "provider": None,
                "modes": ["lexical", "hybrid", "rerank"],
                "k_values": [1, 5],
                "rerank_model": "claude-haiku-5-5",
            },
        }
    )
    assert view.evaluated == "80"
    assert view.config == [
        ("Fornitore del modello", "n/d"),
        ("Ricerche confrontate", "lessicale, ibrida, ibrida riordinata da un modello"),
        ("Valori di k", "1, 5"),
        ("Modello di riordino delle fonti", "claude-haiku-5-5"),
    ]


def test_answer_run_shows_published_retrieval_comparison(tmp_path: Path) -> None:
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "run_id": "answers1",
                "retrieval": {"overall": {"recall@5": 0.75}},
                "answer": {"overall": {"correctness": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "latest_retrieval.json").write_text(
        json.dumps(
            {
                "run_id": "retrieval1",
                "retrieval_by_mode": {
                    "lexical": {"overall": {"recall@5": 0.5}},
                    "hybrid": {"overall": {"recall@5": 0.75}},
                },
            }
        ),
        encoding="utf-8",
    )
    view = load_results(tmp_path / "latest.json")
    assert view is not None
    assert view.run_id == "answers1"
    table = view.sections[0].breakdowns[0]
    assert table.caption == "Recupero delle fonti: per tipo di ricerca (esecuzione retrieval1)"
    assert table.rows == [("Lessicale", ["50,0%"]), ("Ibrida", ["75,0%"])]
    assert [section.title for section in view.sections] == ["Recupero delle fonti", "Risposte"]


def test_retrieval_comparison_without_retrieval_section() -> None:
    view = results_view(
        {"answer": {"overall": {"correctness": 0.5}}},
        {"run_id": "r2", "retrieval_by_mode": {"hybrid": {"overall": {"mrr": 0.5}}}},
    )
    assert [section.title for section in view.sections] == ["Recupero delle fonti", "Risposte"]
    assert view.sections[0].overall == []
    assert view.sections[0].breakdowns[0].rows == [("Ibrida", ["0,500"])]
