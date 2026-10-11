import pytest

from appalti_qa.ingest.labels import act_label, parse_act_href, short_label


@pytest.mark.parametrize(
    ("href", "label"),
    [
        ("/akn/it/act/DECRETO-LEGGE/stato/2023-05-10/51/!main", "D.L. 10 maggio 2023, n. 51"),
        ("/akn/it/act/LEGGE/stato/2023-07-03/87/!main", "L. 3 luglio 2023, n. 87"),
        (
            "/akn/it/act/DECRETO_LEGISLATIVO/stato/2024-12-31/209/!main",
            "D.Lgs. 31 dicembre 2024, n. 209",
        ),
        (
            "/akn/it/act/decretoLegislativo/stato/2017-07-03/117/!main#art_1",
            "D.Lgs. 3 luglio 2017, n. 117",
        ),
        (
            "/akn/it/act/CIRCOLARE/MINISTERO_DELLE_INFRASTRUTTURE_E_DEI_TRASPORTI/2023-11-20/298/!main",
            "Circolare Ministero delle infrastrutture e dei trasporti 20 novembre 2023, n. 298",
        ),
        ("/akn/it/act/ORDINANZA/stato/2024-01-02/7/!main", "Ordinanza 2 gennaio 2024, n. 7"),
        ("not an href", "not an href"),
    ],
)
def test_act_label(href: str, label: str) -> None:
    assert act_label(href) == label


def test_parse_act_href() -> None:
    ref = parse_act_href("/akn/it/act/DECRETO-LEGGE/stato/2025-03-14/25/!main")
    assert ref is not None
    assert (ref.type, ref.number, ref.date.isoformat(), ref.issuer) == (
        "DECRETO_LEGGE",
        "25",
        "2025-03-14",
        None,
    )


def test_short_label() -> None:
    assert short_label("/akn/it/act/decreto_legislativo/stato/2023-03-31/36/!main") == (
        "D.Lgs. 36/2023"
    )
    assert short_label("") is None
