"""View models that turn an `Answer` into what the result template shows.

Model output is untrusted: every piece of it is escaped here or by Jinja's autoescaping. The only
markup added is the link of each valid "[n]" marker to its source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import quote

from markupsafe import Markup, escape

from appalti_qa.models import Answer, Hit

_MARKERS = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_MARKER_SEPARATOR = re.compile(r"\s*[,;]\s*")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MARKER_LINK = Markup(
    '<a class="rif-fonte" href="#fonte-{n}"><span class="visually-hidden">fonte </span>[{n}]</a>'
)


def norma_path(unit_key: str, day: date | None = None) -> str:
    """Link to the unit's page (text in force at `day`, default today, and version history)."""
    path = f"/norma/{quote(unit_key, safe='/:')}"
    return f"{path}?data={day.isoformat()}" if day else path


def link_markers(text: str, source_count: int) -> Markup:
    """Escape `text` and turn "[n]" / "[n, m]" markers that point at a source into links."""
    parts: list[Markup] = []
    position = 0
    for match in _MARKERS.finditer(text):
        numbers = [int(n) for n in _MARKER_SEPARATOR.split(match.group(1))]
        if not all(1 <= n <= source_count for n in numbers):
            continue
        parts.append(escape(text[position : match.start()]))
        parts.extend(_MARKER_LINK.format(n=n) for n in numbers)
        position = match.end()
    parts.append(escape(text[position:]))
    return Markup("").join(parts)


def paragraphs(text: str, source_count: int) -> list[Markup]:
    """The answer split on blank lines, each paragraph escaped and with linked markers."""
    blocks = (block.strip() for block in _PARAGRAPH_BREAK.split(text))
    return [link_markers(block, source_count) for block in blocks if block]


@dataclass(frozen=True, slots=True)
class SourceView:
    number: int  # 1-based, as in the "[n]" markers
    hit: Hit
    passages: list[str]  # exact cited spans, in citation order


@dataclass(frozen=True, slots=True)
class AnswerView:
    answer: Answer
    paragraphs: list[Markup]
    cited: list[SourceView]
    others: list[SourceView]  # retrieved but not cited

    @property
    def demo(self) -> bool:
        return self.answer.provider == "fake"


def answer_view(answer: Answer) -> AnswerView:
    passages: dict[int, list[str]] = {}
    for citation in answer.citations:
        quotes = passages.setdefault(citation.source_index, [])
        text = citation.cited_text.strip()
        if text and text not in quotes:
            quotes.append(text)
    sources = [
        SourceView(index + 1, hit, passages.get(index, []))
        for index, hit in enumerate(answer.sources)
    ]
    return AnswerView(
        answer=answer,
        paragraphs=[] if answer.abstained else paragraphs(answer.text, len(answer.sources)),
        cited=[source for source in sources if source.number - 1 in passages],
        others=[source for source in sources if source.number - 1 not in passages],
    )
