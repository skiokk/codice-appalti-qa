"""Pydantic models shared across ingestion, retrieval, generation, web and evaluation."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from appalti_qa.keys import key_label

UnitKind = Literal["article", "comma", "allegato_block"]
# original: first snapshot; amendment: the words changed where Normattiva marks an amendment;
# editorial: the words, punctuation or split into commi changed outside any amendment marker
# (no act credited); heading: only the article heading changed, the text is the previous
# row's (an act is credited only when the new heading is marked as amended).
ChangeKind = Literal["original", "amendment", "editorial", "heading"]


class Event(BaseModel):
    """An act that amended the code (or the original act itself)."""

    id: int | None = None
    date: date
    kind: Literal["original", "amendment"] = "amendment"
    amending_urn: str
    label: str


class HeadingChange(BaseModel):
    """A new article heading taking effect while the text of a version stays the same."""

    date: date
    heading: str | None = None
    changed_by: Event | None = None  # the amending act, when Normattiva marks the new heading


class UnitVersion(BaseModel):
    """The text of one unit over one validity interval [valid_from, valid_to).

    ``heading`` is the heading in force on ``valid_from``; later headings of the same text are
    in ``heading_changes``.
    """

    id: int | None = None
    unit_key: str
    kind: UnitKind
    article: str | None = None
    comma: str | None = None
    allegato: str | None = None
    heading: str | None = None
    text: str
    text_hash: str
    valid_from: date
    valid_to: date | None = None
    changed_by: Event | None = None
    change_kind: ChangeKind = "amendment"
    repealed: bool = False
    heading_changes: list[HeadingChange] = Field(default_factory=list)

    def in_force_at(self, day: date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or self.valid_to > day)

    def heading_at(self, day: date) -> str | None:
        """The heading in force on ``day`` (the first one before the version starts)."""
        heading = self.heading
        for change in self.heading_changes:
            if change.date <= day:
                heading = change.heading
        return heading


class Hit(BaseModel):
    """A retrieved unit version, ready to be shown and passed to the LLM as a source.

    ``heading`` is the heading in force at the query date; ``valid_from`` and ``valid_to`` are
    those of the text, which a change of the heading alone does not interrupt.
    """

    unit_key: str
    version_id: int
    heading: str | None = None
    text: str
    score: float
    valid_from: date
    valid_to: date | None = None
    normattiva_url: str
    modified_by: list[Event] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return key_label(self.unit_key)


class Citation(BaseModel):
    """A span of a source quoted in support of the answer."""

    unit_key: str
    cited_text: str
    source_index: int = Field(ge=0)  # index into Answer.sources


class Answer(BaseModel):
    question: str
    at_date: date
    text: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Hit] = Field(default_factory=list)
    abstained: bool = False
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @property
    def cited_keys(self) -> list[str]:
        """Distinct cited unit keys, in order of first citation."""
        return list(dict.fromkeys(c.unit_key for c in self.citations))
