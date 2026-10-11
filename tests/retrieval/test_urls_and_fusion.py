from datetime import date

import pytest

from appalti_qa.retrieval import normattiva_url
from appalti_qa.retrieval.fusion import reciprocal_rank_fusion

ACT = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2023-03-31;36"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("50", f"{ACT}~art50!vig=2024-06-01"),
        ("50/1/b", f"{ACT}~art50!vig=2024-06-01"),
        ("225-bis", f"{ACT}~art225bis!vig=2024-06-01"),
        ("all:I.1/2/1", f"{ACT}!vig=2024-06-01"),
        ("all:I.3/p3", f"{ACT}!vig=2024-06-01"),
    ],
)
def test_normattiva_url(key: str, expected: str) -> None:
    assert normattiva_url(key, date(2024, 6, 1)) == expected


def test_normattiva_url_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError):
        normattiva_url("art50", date(2024, 6, 1))


def test_rrf_sums_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert fused == pytest.approx({"a": 1 / 61 + 1 / 62, "c": 1 / 63 + 1 / 61, "b": 1 / 62})
    assert list(fused) == ["a", "c", "b"]


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([[], []]) == {}
    assert list(reciprocal_rank_fusion([[], ["x", "y"]], k=1)) == ["x", "y"]


def test_weighted_rrf_scales_each_list() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60, weights=[2.0, 1.0])
    assert fused == pytest.approx({"a": 2 / 61 + 1 / 62, "b": 2 / 62 + 1 / 61})
    assert list(fused) == ["a", "b"]  # the heavier list decides between symmetric ranks
    flipped = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 2.0])
    assert list(flipped) == ["b", "a"]


def test_rrf_with_unit_weights_is_plain_rrf() -> None:
    rankings = [["a", "b", "c"], ["c", "a"]]
    assert reciprocal_rank_fusion(rankings, weights=[1.0, 1.0]) == reciprocal_rank_fusion(rankings)


def test_rrf_needs_one_weight_per_list() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
