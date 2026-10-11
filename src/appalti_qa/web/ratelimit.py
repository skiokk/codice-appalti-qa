"""In-process sliding-window limits on answered questions: per client per hour, global per day."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

HOUR_S = 3600.0
DAY_S = 86400.0
_SWEEP_THRESHOLD = 4096  # forget idle clients once this many are tracked


@dataclass(frozen=True, slots=True)
class LimitExceeded:
    scope: Literal["client", "global"]
    limit: int
    retry_after_s: int


class RateLimiter:
    """Admits a request when both windows have room; a limit of 0 disables that window."""

    def __init__(
        self,
        per_client_per_hour: int,
        global_per_day: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.per_client_per_hour = per_client_per_hour
        self.global_per_day = global_per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    def acquire(self, client: str) -> LimitExceeded | None:
        """Record a request from `client` if allowed; otherwise say which limit was hit."""
        now = self._clock()
        with self._lock:
            if self.global_per_day:
                _expire(self._global, now - DAY_S)
                if len(self._global) >= self.global_per_day:
                    return _exceeded("global", self.global_per_day, self._global, DAY_S, now)
            if self.per_client_per_hour:
                window = self._clients.setdefault(client, deque())
                _expire(window, now - HOUR_S)
                if len(window) >= self.per_client_per_hour:
                    return _exceeded("client", self.per_client_per_hour, window, HOUR_S, now)
                window.append(now)
                if len(self._clients) > _SWEEP_THRESHOLD:
                    self._sweep(now)
            if self.global_per_day:
                self._global.append(now)
        return None

    def _sweep(self, now: float) -> None:
        for client, window in list(self._clients.items()):
            _expire(window, now - HOUR_S)
            if not window:
                del self._clients[client]


def _expire(window: deque[float], cutoff: float) -> None:
    while window and window[0] <= cutoff:
        window.popleft()


def _exceeded(
    scope: Literal["client", "global"], limit: int, window: deque[float], span: float, now: float
) -> LimitExceeded:
    retry_after = max(1, math.ceil(window[0] + span - now))
    return LimitExceeded(scope, limit, retry_after)
