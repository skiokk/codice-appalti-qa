"""View of the latest evaluation run (`eval/results/latest.json`) for the "Valutazione" page.

The file is produced by `appalti_qa.evaluation`; this module only reads it and tolerates missing
or unexpected fields, so that a partial results file still renders. When the latest run is an
answer run, which does not compare retrieval modes, the comparison is taken from the latest
published retrieval run (`latest_retrieval.json` next to it).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from appalti_qa.web.formatting import format_decimal, format_int, format_percent

logger = logging.getLogger(__name__)

RESULTS_PATH = Path("eval/results/latest.json")
RETRIEVAL_RESULTS_NAME = "latest_retrieval.json"
MAX_ITEMS = 500

_METRIC_LABELS = {
    "n": "Domande",
    "mrr": "MRR",
    "correctness": "Correttezza",
    "citation_precision": "Precisione delle citazioni",
    "citation_recall": "Copertura delle citazioni",
    "abstention_accuracy": "Accuratezza dell'astensione",
    "abstention_rate": "Quota di astensioni",
    "latency_p50_ms": "Latenza mediana",
    "latency_p95_ms": "Latenza al 95° percentile",
    "input_tokens": "Token in ingresso",
    "output_tokens": "Token in uscita",
    "cost_usd": "Costo stimato",
    "cost_unknown": "Risposte senza prezzo noto",
    "judge_input_tokens": "Token in ingresso del giudice",
    "judge_output_tokens": "Token in uscita del giudice",
    "judge_cost_usd": "Costo stimato del giudice",
    "judge_cost_unknown": "Valutazioni senza prezzo noto",
    "errors": "Errori",
    "judge_errors": "Errori del giudice",
}
_AT_K = re.compile(r"(?P<article>article_)?(?P<name>recall|hit)@(?P<k>\d+)")
_COUNTS = {
    "n",
    "count",
    "size",
    "verified",
    "evaluated",
    "errors",
    "judge_errors",
    "cost_unknown",
    "judge_cost_unknown",
}
# Columns of the per-kind and per-topic tables (the overall table lists every metric).
_BREAKDOWN_METRICS = (
    "n",
    "recall@5",
    "hit@5",
    "mrr",
    "correctness",
    "citation_precision",
    "citation_recall",
    "abstention_accuracy",
)

_KIND_LABELS = {
    "answerable": "Con risposta nel testo",
    "temporal": "Dipendenti dalla data",
    "unanswerable": "Senza risposta nel testo",
    "multi": "Con più riferimenti",
}
_CONFIG_LABELS = {
    "provider": "Fornitore del modello",
    "model": "Modello linguistico",
    "effort": "Livello di ragionamento",
    "retrieval_mode": "Ricerca",
    "modes": "Ricerche confrontate",
    "k": "Fonti recuperate per domanda",
    "k_values": "Valori di k",
    "embedding_model": "Modello di embedding",
    "judge": "Giudice della correttezza",
    "judge_model": "Modello del giudice",
}
_MODE_LABELS = {"lexical": "lessicale", "semantic": "semantica", "hybrid": "ibrida"}


@dataclass(frozen=True, slots=True)
class MetricTable:
    caption: str
    group_header: str
    columns: list[str]
    rows: list[tuple[str, list[str]]]


@dataclass(frozen=True, slots=True)
class MetricSection:
    title: str
    overall: list[tuple[str, str]]
    breakdowns: list[MetricTable]


@dataclass(frozen=True, slots=True)
class ItemRow:
    id: str
    expected: str
    score: str
    abstained: str
    cited: str
    retrieved: str
    latency: str


@dataclass(frozen=True, slots=True)
class ResultsView:
    run_id: str
    created_at: str
    dataset_size: str
    verified: str
    evaluated: str
    config: list[tuple[str, str]]
    sections: list[MetricSection]
    items: list[ItemRow]


def metric_label(name: str) -> str:
    if name in _METRIC_LABELS:
        return _METRIC_LABELS[name]
    if name == "article_mrr":
        return "MRR per articolo"
    match = _AT_K.fullmatch(name)
    if match:
        label = f"{match['name'].capitalize()}@{match['k']}"
        return f"{label} per articolo" if match["article"] else label
    return name.replace("_", " ").capitalize()


def format_metric(name: str, value: object) -> str:
    """Ratios as percentages, MRR as a decimal, latencies in ms, costs in USD, counts as is."""
    if value is None:
        return "n/d"
    if isinstance(value, bool):
        return "sì" if value else "no"
    if not isinstance(value, int | float):
        return str(value)
    if name.endswith("_ms"):
        return f"{format_int(value)} ms"
    if name.endswith("_usd"):
        return f"{format_decimal(value, 2 if value >= 1 else 4)} USD"
    if name in _COUNTS or name.endswith("_tokens") or isinstance(value, int):
        return format_int(value)
    if name.endswith("mrr"):
        return format_decimal(value, 3)
    if 0 <= value <= 1:
        return format_percent(value)
    return format_decimal(value, 2)


def group_label(name: str) -> str:
    label = _KIND_LABELS.get(name) or _MODE_LABELS.get(name)
    return label.capitalize() if label else name.replace("_", " ").capitalize()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _breakdown(caption: str, header: str, groups: Mapping[str, Any]) -> MetricTable | None:
    rows = {name: _mapping(metrics) for name, metrics in groups.items()}
    present = list(dict.fromkeys(name for metrics in rows.values() for name in metrics))
    names = [name for name in _BREAKDOWN_METRICS if name in present] or present
    if not names:
        return None
    return MetricTable(
        caption=caption,
        group_header=header,
        columns=[metric_label(name) for name in names],
        rows=[
            (group_label(group), [format_metric(n, metrics.get(n)) for n in names])
            for group, metrics in rows.items()
        ],
    )


def _section(title: str, raw: Mapping[str, Any]) -> MetricSection | None:
    metrics = _mapping(raw.get("overall"))
    overall = [(metric_label(name), format_metric(name, value)) for name, value in metrics.items()]
    if not overall:
        return None
    tables = [
        _breakdown(f"{title}: per tipo di domanda", "Tipo", _mapping(raw.get("by_kind"))),
        _breakdown(f"{title}: per argomento", "Argomento", _mapping(raw.get("by_category"))),
    ]
    return MetricSection(title, overall, [table for table in tables if table])


def _config_value(name: str, value: object) -> str:
    if value is None:
        return "n/d"
    if isinstance(value, list):
        return ", ".join(_config_value(name, item) for item in value)
    if name in {"retrieval_mode", "modes"} and isinstance(value, str):
        return _MODE_LABELS.get(value, value)
    return str(value)


def _created_at(value: object) -> str:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value or "n/d")
    return f"{moment:%d/%m/%Y %H:%M} UTC"


def _keys(value: object, limit: int | None = None) -> str:
    if not isinstance(value, list):
        return "n/d"
    return ", ".join(str(key) for key in value[:limit]) or "nessuno"


def _item(raw: Mapping[str, Any], mode: object) -> ItemRow:
    # Retrieval-only runs keep the ranking per mode; show the primary mode's.
    retrieved = raw.get(
        "retrieved", _mapping(_mapping(raw.get("by_mode")).get(mode)).get("retrieved")
    )
    return ItemRow(
        id=str(raw.get("id", "")),
        expected=_keys(raw.get("expected_refs")),
        score=format_metric("score", raw.get("score")),
        abstained=format_metric("abstained", raw.get("abstained")),
        cited=_keys(raw.get("cited")),
        retrieved=_keys(retrieved, limit=5),
        latency=format_metric("latency_ms", raw.get("latency_ms")),
    )


def _by_mode(raw: Mapping[str, Any], run_id: object = None) -> MetricTable | None:
    """Overall retrieval metrics of each mode compared in a retrieval-only run."""
    modes = {mode: _mapping(metrics).get("overall") for mode, metrics in raw.items()}
    caption = "Recupero delle fonti: per tipo di ricerca"
    if run_id:
        caption += f" (esecuzione {run_id})"
    return _breakdown(caption, "Ricerca", modes)


def results_view(
    raw: Mapping[str, Any], retrieval_run: Mapping[str, Any] | None = None
) -> ResultsView:
    """The page data of a run; `retrieval_run` supplies the mode comparison it lacks."""
    dataset = _mapping(raw.get("dataset"))
    sections = [
        _section(title, _mapping(raw.get(key)))
        for key, title in (("retrieval", "Recupero delle fonti"), ("answer", "Risposte"))
    ]
    by_mode = _by_mode(_mapping(raw.get("retrieval_by_mode")))
    if by_mode is None and retrieval_run is not None:
        by_mode = _by_mode(
            _mapping(retrieval_run.get("retrieval_by_mode")), retrieval_run.get("run_id", "n/d")
        )
    if by_mode:
        if sections[0] is None:
            sections[0] = MetricSection("Recupero delle fonti", [], [])
        sections[0].breakdowns.insert(0, by_mode)
    items = raw.get("items") if isinstance(raw.get("items"), list) else []
    mode = _mapping(raw.get("config")).get("retrieval_mode")
    return ResultsView(
        run_id=str(raw.get("run_id", "n/d")),
        created_at=_created_at(raw.get("created_at")),
        dataset_size=format_metric("size", dataset.get("size")),
        verified=format_metric("verified", dataset.get("verified")),
        evaluated=format_metric("evaluated", dataset.get("evaluated", dataset.get("size"))),
        config=[
            (_CONFIG_LABELS.get(name, name), _config_value(name, value))
            for name, value in _mapping(raw.get("config")).items()
        ],
        sections=[section for section in sections if section],
        items=[_item(_mapping(item), mode) for item in items[:MAX_ITEMS]],
    )


def _read(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read evaluation results %s: %s", path, exc)
        return None
    if not isinstance(raw, Mapping):
        logger.warning("Unexpected evaluation results format in %s", path)
        return None
    return raw


def load_results(path: Path = RESULTS_PATH) -> ResultsView | None:
    """The results view, or None when the file is missing or unreadable."""
    raw = _read(path)
    if raw is None:
        return None
    retrieval_run = None
    if not raw.get("retrieval_by_mode"):
        retrieval_run = _read(path.with_name(RETRIEVAL_RESULTS_NAME))
    return results_view(raw, retrieval_run)
