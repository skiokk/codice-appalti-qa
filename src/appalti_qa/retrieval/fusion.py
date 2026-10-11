"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def reciprocal_rank_fusion[T: Hashable](
    rankings: Iterable[Sequence[T]], *, k: int = 60
) -> dict[T, float]:
    """Fuse ranked lists: each item scores the sum of 1 / (k + rank) over the lists it is in.

    Ranks are 1-based. The result is sorted by descending score; ties keep first-seen order.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda pair: -pair[1]))
