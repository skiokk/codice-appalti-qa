"""Human-readable Italian labels for acts referenced by Akoma Ntoso hrefs.

``/akn/it/act/DECRETO-LEGGE/stato/2023-05-10/51/!main`` -> ``"D.L. 10 maggio 2023, n. 51"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)

# Act types as they appear in Normattiva hrefs (dashes and case normalised), with their
# customary Italian abbreviation.
_ABBREVIATIONS = {
    "DECRETO_LEGGE": "D.L.",
    "DECRETO_LEGISLATIVO": "D.Lgs.",
    "LEGGE": "L.",
    "LEGGE_COSTITUZIONALE": "L. cost.",
    "DECRETO_DEL_PRESIDENTE_DELLA_REPUBBLICA": "D.P.R.",
    "DECRETO_DEL_PRESIDENTE_DEL_CONSIGLIO_DEI_MINISTRI": "D.P.C.M.",
    "DECRETO_MINISTERIALE": "D.M.",
    "DECRETO": "Decreto",
    "CIRCOLARE": "Circolare",
    "COMUNICATO": "Comunicato",
    "REGOLAMENTO": "Regolamento",
    "DELIBERA": "Delibera",
    "SENTENZA": "Sentenza",
}

_HREF_RE = re.compile(
    r"/akn/[a-z]{2}/act/(?P<type>[^/]+)/(?P<author>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2})"
    r"/(?P<number>[^/!]+)"
)


@dataclass(frozen=True, slots=True)
class ActRef:
    """The parts of an AKN act reference."""

    type: str  # normalised, e.g. "DECRETO_LEGGE"
    author: str  # "stato" or the issuing body, e.g. "MINISTERO_DELLE_INFRASTRUTTURE_..."
    date: date
    number: str

    @property
    def abbreviation(self) -> str:
        return _ABBREVIATIONS.get(self.type) or self.type.replace("_", " ").capitalize()

    @property
    def issuer(self) -> str | None:
        """The issuing body for acts not issued by the State, e.g. a ministry."""
        if self.author.lower() == "stato":
            return None
        return self.author.replace("_", " ").replace("-", " ").capitalize()


def parse_act_href(href: str) -> ActRef | None:
    """Parse an AKN act href; None when it does not follow the /akn/it/act/... pattern."""
    match = _HREF_RE.search(href)
    if not match:
        return None
    act_type = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", match["type"])  # decretoLegge -> decreto_Legge
    return ActRef(
        type=act_type.replace("-", "_").upper(),
        author=match["author"],
        date=date.fromisoformat(match["date"]),
        number=match["number"],
    )


def italian_date(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]} {day.year}"


def act_label(href: str) -> str:
    """'D.Lgs. 31 dicembre 2024, n. 209'; the href itself when it cannot be parsed."""
    ref = parse_act_href(href)
    if ref is None:
        return href
    parts = [ref.abbreviation]
    if ref.issuer:
        parts.append(ref.issuer)
    parts.append(italian_date(ref.date))
    return f"{' '.join(parts)}, n. {ref.number}"


def short_label(href: str) -> str | None:
    """'D.Lgs. 36/2023' for the act's own FRBR href."""
    ref = parse_act_href(href)
    return f"{ref.abbreviation} {ref.number}/{ref.date.year}" if ref else None
