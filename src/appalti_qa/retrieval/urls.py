"""Links to the official consolidated text on Normattiva."""

from __future__ import annotations

from datetime import date

from appalti_qa.keys import parse_key

NORMATTIVA_ACT_URL = (
    "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36"
)


def normattiva_url(unit_key: str, at_date: date) -> str:
    """Normattiva page showing the text in force at `at_date`.

    Code units link to their article (`~art225bis`); allegati have no stable per-article anchor,
    so their units link to the act as a whole. Raises ValueError on invalid keys.
    """
    parsed = parse_key(unit_key)
    anchor = "" if parsed.annex else f"~art{(parsed.article or '').replace('-', '')}"
    return f"{NORMATTIVA_ACT_URL}{anchor}!vig={at_date.isoformat()}"
