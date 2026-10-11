"""Evaluation: dataset checks, retrieval and answer metrics, judges, reports, xlsx review."""

from appalti_qa.evaluation.cli import cli_main
from appalti_qa.evaluation.dataset import EvalItem, load_dataset, validate_dataset
from appalti_qa.evaluation.judge import Judge, JudgeVerdict, get_judge
from appalti_qa.evaluation.metrics import answer_metrics, retrieval_metrics
from appalti_qa.evaluation.report import render_report, write_results
from appalti_qa.evaluation.xlsx import export_xlsx, import_xlsx

__all__ = [
    "EvalItem",
    "Judge",
    "JudgeVerdict",
    "answer_metrics",
    "cli_main",
    "export_xlsx",
    "get_judge",
    "import_xlsx",
    "load_dataset",
    "render_report",
    "retrieval_metrics",
    "validate_dataset",
    "write_results",
]
