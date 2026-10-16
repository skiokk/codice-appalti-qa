"""View models that turn an `Answer` into what the result template shows.

Model output is untrusted: every piece of it is escaped here or by Jinja's autoescaping. The only
markup added is the link of each valid "[n]" marker to its source and the few Markdown forms the
model writes (paragraphs, bullet and numbered lists, "**bold**", "### headings"), rebuilt as tags
around text that is already escaped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from operator import itemgetter
from urllib.parse import quote

from markupsafe import Markup, escape

from appalti_qa.models import Answer, Hit

_MARKERS = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_MARKER_SEPARATOR = re.compile(r"\s*[,;]\s*")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*\d{1,2}[.)]\s+(.*)$")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
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


def _inline(text: str, source_count: int) -> Markup:
    """Escaped text with linked markers and "**bold**" spans as <strong>."""
    html = str(link_markers(text, source_count))  # escaped: "**" is the only markup left
    return Markup(_BOLD.sub(r"<strong>\1</strong>", html))


def _line_kind(line: str) -> tuple[str, str]:
    """("ul" | "ol" | "h" | "p", text without its Markdown prefix)."""
    for kind, pattern in (("ul", _BULLET), ("ol", _NUMBERED), ("h", _HEADING)):
        match = pattern.match(line)
        if match:
            return kind, match.group(1)
    return "p", line.strip()


def answer_blocks(text: str, source_count: int) -> list[Markup]:
    """The answer as HTML blocks: paragraphs (split on blank lines), lists and headings.

    Consecutive "- item" or "1. item" lines become a list; a "### heading" line becomes a
    paragraph in bold; consecutive other lines stay one paragraph, one line each.
    """
    blocks: list[Markup] = []
    for chunk in _PARAGRAPH_BREAK.split(text):
        lines = [_line_kind(line) for line in chunk.splitlines() if line.strip()]
        for kind, group in groupby(lines, key=itemgetter(0)):
            parts = [_inline(content, source_count) for _, content in group]
            if kind in ("ul", "ol"):
                items = Markup("").join(Markup("<li>{}</li>").format(part) for part in parts)
                blocks.append(Markup(f"<{kind}>{{}}</{kind}>").format(items))
            elif kind == "h":
                blocks.extend(Markup("<p><strong>{}</strong></p>").format(part) for part in parts)
            else:
                blocks.append(Markup("<p>{}</p>").format(Markup("<br>").join(parts)))
    return blocks


@dataclass(frozen=True, slots=True)
class SourceView:
    number: int  # 1-based, as in the "[n]" markers
    hit: Hit
    passages: list[str]  # exact cited spans, in citation order


@dataclass(frozen=True, slots=True)
class AnswerView:
    answer: Answer
    blocks: list[Markup]  # HTML of the answer text (see `answer_blocks`)
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
        blocks=[] if answer.abstained else answer_blocks(answer.text, len(answer.sources)),
        cited=[source for source in sources if source.number - 1 in passages],
        others=[source for source in sources if source.number - 1 not in passages],
    )
