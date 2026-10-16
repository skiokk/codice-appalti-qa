from datetime import date

from markupsafe import Markup

from appalti_qa.models import Citation
from appalti_qa.web.formatting import (
    format_date,
    format_date_long,
    format_int,
    format_percent,
    format_seconds,
    validity,
)
from appalti_qa.web.views import answer_blocks, answer_view, link_markers, norma_path


def test_markers_become_links_and_text_is_escaped() -> None:
    html = link_markers("A <b>B</b> [1] e [2, 3] ma non [4] né [x].", 3)
    assert isinstance(html, Markup)
    assert "&lt;b&gt;B&lt;/b&gt;" in html
    assert html.count('class="rif-fonte"') == 3
    for n in (1, 2, 3):
        assert f'href="#fonte-{n}"' in html
    assert "[4]" in html and 'href="#fonte-4"' not in html
    assert "[x]" in html


MARKER_1 = '<a class="rif-fonte" href="#fonte-1"><span class="visually-hidden">fonte </span>[1]</a>'


def test_blocks_split_on_blank_lines_and_keep_line_breaks() -> None:
    assert answer_blocks("Uno.\n\n  \nDue [1].\nStessa riga.", 1) == [
        Markup("<p>Uno.</p>"),
        Markup(f"<p>Due {MARKER_1}.<br>Stessa riga.</p>"),
    ]


def test_blocks_rebuild_lists_bold_and_headings() -> None:
    text = "### Compiti\n- **coordinare** [1];\n- valutare.\n1. primo\n2) secondo\nFine."
    assert answer_blocks(text, 1) == [
        Markup("<p><strong>Compiti</strong></p>"),
        Markup(f"<ul><li><strong>coordinare</strong> {MARKER_1};</li><li>valutare.</li></ul>"),
        Markup("<ol><li>primo</li><li>secondo</li></ol>"),
        Markup("<p>Fine.</p>"),
    ]


def test_blocks_escape_model_output() -> None:
    (block,) = answer_blocks('**<script>alert("x")</script>** e *a* ** b **', 1)
    assert "<script>" not in block
    assert "<strong>&lt;script&gt;" in block
    assert "*a* ** b **" in block  # single asterisks and padded pairs stay as written


def test_answer_view_groups_cited_and_other_sources(answer_factory) -> None:
    answer = answer_factory("Domanda?", date(2025, 1, 1))
    answer.citations.append(
        Citation(unit_key="50/1", cited_text="Testo di prova del comma 50/1.", source_index=0)
    )
    view = answer_view(answer)
    assert [source.number for source in view.cited] == [1]
    assert view.cited[0].passages == ["Testo di prova del comma 50/1."]  # deduplicated
    assert [source.number for source in view.others] == [2]
    assert view.demo


def test_abstained_answer_has_no_blocks(answer_factory) -> None:
    answer = answer_factory("Domanda?", date(2025, 1, 1), abstained=True, citations=[])
    view = answer_view(answer)
    assert view.blocks == []
    assert view.cited == [] and len(view.others) == 2


def test_norma_path() -> None:
    assert norma_path("all:I.3/p1", date(2024, 1, 2)) == "/norma/all:I.3/p1?data=2024-01-02"
    assert norma_path("225-bis") == "/norma/225-bis"


def test_italian_formatting() -> None:
    assert format_date(date(2024, 12, 31)) == "31/12/2024"
    assert format_date_long(date(2026, 10, 1)) == "1 ottobre 2026"
    assert validity(date(2023, 4, 1), None) == "In vigore dal 01/04/2023"
    assert validity(date(2023, 4, 1), date(2024, 12, 31)) == (
        "In vigore dal 01/04/2023 al 30/12/2024"
    )
    assert format_int(1234567) == "1.234.567"
    assert format_percent(0.725) == "72,5%"
    assert format_seconds(1234) == "1,2 s"
