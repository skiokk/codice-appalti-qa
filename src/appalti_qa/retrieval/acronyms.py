"""Acronyms of procurement practice, spelled out as the code writes them.

People ask about the "DEC" or the "DGUE", but the code never uses those acronyms: it says
"direttore dell'esecuzione del contratto" and "documento di gara unico europeo". A question that
only has the acronym then matches nothing lexically and little semantically. `expand_acronyms`
adds the code's wording after the first occurrence of each acronym; search uses the expanded
text, while the model still reads the question as written.

Matching is case-sensitive and on whole words, so "DEC" does not match "dec" or "DECRETO".
"""

from __future__ import annotations

import re

# acronym -> the code's own wording (with its apostrophe accents, as in the Normattiva text)
ACRONYMS: dict[str, str] = {
    "ATI": "raggruppamento temporaneo",
    "BDNCP": "banca dati nazionale dei contratti pubblici",
    "CAM": "criteri ambientali minimi",
    "CCNL": "contratto collettivo nazionale di lavoro",
    "CCT": "collegio consultivo tecnico",
    "CIG": "codice identificativo di gara",
    "CSE": "coordinatore per la sicurezza in fase di esecuzione",
    "DEC": "direttore dell'esecuzione del contratto",
    "DGUE": "documento di gara unico europeo",
    "DURC": "documento unico di regolarita' contributiva",
    "FVOE": "fascicolo virtuale dell'operatore economico",
    "MEPA": "mercato elettronico della pubblica amministrazione",
    "MePA": "mercato elettronico della pubblica amministrazione",
    "Mepa": "mercato elettronico della pubblica amministrazione",
    "OE": "operatore economico",
    "OEPV": "offerta economicamente piu' vantaggiosa",
    "PEF": "piano economico-finanziario",
    "PFTE": "progetto di fattibilita' tecnico-economica",
    "PPP": "partenariato pubblico-privato",
    "RTI": "raggruppamento temporaneo",
    "SA": "stazione appaltante",
}
_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, ACRONYMS), key=len, reverse=True)) + r")\b"
)


def expand_acronyms(text: str) -> str:
    """`text` with "ACRONYM (expansion)" at the first occurrence of each known acronym.

    An acronym whose expansion already appears in the text is left alone.
    """
    lowered = text.lower()
    done: set[str] = set()

    def expand(match: re.Match[str]) -> str:
        expansion = ACRONYMS[match.group(1)]
        if expansion in done or expansion.lower() in lowered:
            return match.group(0)
        done.add(expansion)
        return f"{match.group(0)} ({expansion})"

    return _PATTERN.sub(expand, text)
