"""Small lxml helpers for Akoma Ntoso documents."""

from __future__ import annotations

from collections.abc import Iterator

from lxml import etree

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
_SKIPPED = frozenset({"authorialNote"})  # editorial footnotes, not part of the legal text


def akn(name: str) -> str:
    """Clark notation for an AKN element name: akn('article') -> '{ns}article'."""
    return f"{{{AKN_NS}}}{name}"


def local_name(element: etree._Element) -> str:
    return etree.QName(element).localname if isinstance(element.tag, str) else ""


def children(element: etree._Element, *names: str) -> Iterator[etree._Element]:
    """Direct element children, optionally restricted to the given local names."""
    for child in element:
        if isinstance(child.tag, str) and (not names or local_name(child) in names):
            yield child


def child(element: etree._Element, name: str) -> etree._Element | None:
    return next(children(element, name), None)


def inline_text(element: etree._Element | None) -> str:
    """Text content of an element, skipping footnotes but keeping the text around them."""
    if element is None:
        return ""
    parts = [element.text or ""]
    for node in element:
        if isinstance(node.tag, str) and local_name(node) not in _SKIPPED:
            parts.append(inline_text(node))
        parts.append(node.tail or "")
    return "".join(parts)


def block_lines(element: etree._Element, *, keep_breaks: bool = False) -> list[str]:
    """Lines of a content-like element: one per ``<p>`` and one per table row.

    Whitespace inside a ``<p>`` is collapsed unless ``keep_breaks`` is set, in which case its
    own line breaks are kept (annex tables drawn as text). Table cells are joined with " | ".
    """
    lines: list[str] = []
    for node in children(element):
        name = local_name(node)
        if name in _SKIPPED:
            continue
        if name == "table":
            for row in node.iter(akn("tr")):
                cells = (" ".join(inline_text(cell).split()) for cell in children(row))
                lines.append(" | ".join(cell for cell in cells if cell))
        else:
            text = inline_text(node)
            lines.append(text if keep_breaks else " ".join(text.split()))
    return lines
