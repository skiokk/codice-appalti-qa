"""Jinja2 environment: autoescaping everywhere, Italian filters and pinned third-party assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jinja2
from fastapi.templating import Jinja2Templates

from appalti_qa import __version__
from appalti_qa.keys import key_label
from appalti_qa.web import formatting
from appalti_qa.web.views import norma_path

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

CDN = "https://cdn.jsdelivr.net/npm"
BOOTSTRAP_ITALIA_VERSION = "2.18.3"
HTMX_VERSION = "2.0.11"


@dataclass(frozen=True, slots=True)
class Asset:
    """A CDN file pinned to an exact version, with its Subresource Integrity hash."""

    url: str
    integrity: str


# Hashes computed from the files served at these URLs (sha384, base64).
ASSETS = {
    "bootstrap_italia_css": Asset(
        f"{CDN}/bootstrap-italia@{BOOTSTRAP_ITALIA_VERSION}/dist/css/bootstrap-italia.min.css",
        "sha384-X1VhbjQ27qjkEci0FoXB0x50918IZwgk8UPm4vY+brDtJfaP6MUxrF3DcSlQwxBr",
    ),
    "htmx": Asset(
        f"{CDN}/htmx.org@{HTMX_VERSION}/dist/htmx.min.js",
        "sha384-2OatzQy1H+Zd/IIrjr1TcuDGqLXeHhbooAyJY1KdQMKnr4LZ22k31GBLdYKHmVjg",
    ),
}


def static_url(path: str) -> str:
    """URL of a bundled static file, versioned so that releases bust browser caches."""
    return f"/static/{path}?v={__version__}"


def build_templates() -> Jinja2Templates:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=jinja2.StrictUndefined,
    )
    env.filters.update(
        data=formatting.format_date,
        data_estesa=formatting.format_date_long,
        intero=formatting.format_int,
        secondi=formatting.format_seconds,
        etichetta=key_label,
    )
    env.globals.update(
        assets=ASSETS,
        static=static_url,
        validity=formatting.validity,
        norma_path=norma_path,
        version=__version__,
    )
    return Jinja2Templates(env=env)
