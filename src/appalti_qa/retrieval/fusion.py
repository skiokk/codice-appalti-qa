"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009), optionally weighted."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def reciprocal_rank_fusion[T: Hashable](
    rankings: Iterable[Sequence[T]], *, k: int = 60, weights: Sequence[float] | None = None
) -> dict[T, float]:
    """Fuse ranked lists: each item scores the sum of w / (k + rank) over the lists it is in.

    Ranks are 1-based and `w` is the list's weight (1 for every list when `weights` is None).
    The result is sorted by descending score; ties keep first-seen order.
    """
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is needed")
    scores: dict[T, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return dict(sorted(scores.items(), key=lambda pair: -pair[1]))
