"""Structural accessibility checks (WCAG 2.1 AA) on the rendered HTML of every page."""

import json
from collections import Counter
from datetime import date
from itertools import pairwise
from pathlib import Path

import lxml.html
import pytest
from fastapi.testclient import TestClient

from appalti_qa.models import UnitVersion

PAGES = [
    ("get", "/"),
    ("post", "/chiedi"),
    ("post-invalid", "/chiedi"),
    ("get", "/norma/50/1"),
    ("get", "/valutazione"),
    ("get", "/informazioni"),
    ("get", "/pagina-inesistente"),
]


@pytest.fixture
def pages(client: TestClient, backend, hit_factory, results_path: Path) -> dict[str, str]:
    backend.hits = [hit_factory("50/1")]
    backend.versions = [
        UnitVersion(
            unit_key="50/1",
            kind="comma",
            text="1. Testo.",
            text_hash="a" * 64,
            valid_from=date(2023, 4, 1),
        )
    ]
    results_path.write_text(
        json.dumps(
            {
                "retrieval": {"overall": {"recall@5": 0.5}, "by_kind": {"multi": {"mrr": 1}}},
                "items": [{"id": "Q001", "score": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    rendered = {}
    for method, path in PAGES:
        if method == "get":
            response = client.get(path)
        else:
            question = "" if method == "post-invalid" else "Domanda?"
            response = client.post(path, data={"domanda": question})
        rendered[f"{method} {path}"] = response.text
    return rendered


def each_page(pages: dict[str, str]):
    for name, text in pages.items():
        yield name, lxml.html.document_fromstring(text)


def test_language_title_and_landmarks(pages) -> None:
    for name, page in each_page(pages):
        assert page.get("lang") == "it", name
        assert page.findtext(".//title").strip(), name
        assert len(page.xpath("//main")) == 1, name
        assert len(page.xpath("//body/header")) == 1, name
        assert len(page.xpath("//body/footer")) == 1, name
        assert all(nav.get("aria-label") for nav in page.xpath("//nav")), name


def test_skip_link_is_first_and_targets_main(pages) -> None:
    for name, page in each_page(pages):
        first_link = page.xpath("//body//a")[0]
        assert first_link.get("href") == "#contenuto", name
        assert page.xpath("//main")[0].get("id") == "contenuto", name


def test_single_h1_and_no_skipped_heading_levels(pages) -> None:
    for name, page in each_page(pages):
        levels = [int(h.tag[1]) for h in page.xpath("//h1|//h2|//h3|//h4|//h5|//h6")]
        assert levels.count(1) == 1 and levels[0] == 1, name
        for previous, current in pairwise(levels):
            assert current <= previous + 1, (name, levels)


def test_controls_have_labels_and_valid_references(pages) -> None:
    for name, page in each_page(pages):
        ids = Counter(page.xpath("//@id"))
        assert not [i for i, n in ids.items() if n > 1], name
        labelled = set(page.xpath("//label/@for"))
        for control in page.xpath("//input[not(@type='hidden')]|//textarea|//select"):
            assert control.get("id") in labelled or control.get("aria-label"), name
        for attribute in ("aria-describedby", "aria-labelledby"):
            for value in page.xpath(f"//@{attribute}"):
                assert all(ref in ids for ref in value.split()), (name, value)
        for anchor in page.xpath("//a[starts-with(@href, '#')]"):
            target = anchor.get("href")[1:]
            assert target in ids, (name, target)


def test_links_and_buttons_have_text(pages) -> None:
    for name, page in each_page(pages):
        for element in page.xpath("//a|//button"):
            assert element.text_content().strip(), (name, lxml.html.tostring(element))


def test_tables_have_captions_and_header_scope(pages) -> None:
    for name, page in each_page(pages):
        for table in page.xpath("//table"):
            assert table.find("caption") is not None, name
            assert all(th.get("scope") in {"col", "row"} for th in table.xpath(".//th")), name


def test_answer_region_is_live_and_indicator_has_text(pages) -> None:
    page = lxml.html.document_fromstring(pages["get /"])
    assert page.get_element_by_id("esito").get("aria-live") == "polite"
    form = page.get_element_by_id("modulo-domanda")
    indicator = page.get_element_by_id(form.get("hx-indicator").lstrip("#"))
    assert indicator.text_content().strip()
    assert page.get_element_by_id("stato-richiesta").get("role") == "status"


def test_third_party_assets_are_pinned_with_integrity(pages) -> None:
    page = lxml.html.document_fromstring(pages["get /"])
    external = [
        element
        for element in page.xpath("//script[@src]|//link[@rel='stylesheet']")
        if (element.get("src") or element.get("href")).startswith("https://")
    ]
    assert len(external) == 2
    for element in external:
        url = element.get("src") or element.get("href")
        assert "@" in url.split("/npm/")[1].split("/")[0]  # package@exact-version
        assert element.get("integrity", "").startswith("sha384-")
        assert element.get("crossorigin") == "anonymous"
