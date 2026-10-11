"""Web interface (FastAPI, Jinja2, htmx, Bootstrap Italia) and JSON API."""

from appalti_qa.web.app import AnswerFn, cli_serve, create_app

__all__ = ["AnswerFn", "cli_serve", "create_app"]
