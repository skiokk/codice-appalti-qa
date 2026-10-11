from appalti_qa.web.ratelimit import DAY_S, HOUR_S, LimitExceeded, RateLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_per_client_window_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(2, 0, clock=clock)
    assert limiter.acquire("a") is None
    clock.now += 600
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") == LimitExceeded("client", 2, 3000)
    assert limiter.acquire("b") is None  # other clients are independent
    clock.now += 3000  # the first request leaves the window
    assert limiter.acquire("a") is None


def test_global_limit_counts_every_client() -> None:
    clock = Clock()
    limiter = RateLimiter(0, 3, clock=clock)
    for client in "abc":
        assert limiter.acquire(client) is None
    exceeded = limiter.acquire("d")
    assert exceeded is not None and exceeded.scope == "global"
    assert exceeded.retry_after_s == DAY_S
    clock.now += DAY_S
    assert limiter.acquire("d") is None


def test_rejected_requests_do_not_count() -> None:
    clock = Clock()
    limiter = RateLimiter(1, 2, clock=clock)
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is not None  # client limit: not added to the global window
    assert limiter.acquire("b") is None
    assert limiter.acquire("c") is not None  # global window holds a and b only


def test_zero_disables_limits() -> None:
    limiter = RateLimiter(0, 0)
    assert all(limiter.acquire("a") is None for _ in range(100))


def test_idle_clients_are_forgotten(monkeypatch) -> None:
    monkeypatch.setattr("appalti_qa.web.ratelimit._SWEEP_THRESHOLD", 3)
    clock = Clock()
    limiter = RateLimiter(5, 0, clock=clock)
    for client in "abc":
        limiter.acquire(client)
    assert set(limiter._clients) == {"a", "b", "c"}
    clock.now += HOUR_S + 1
    limiter.acquire("d")  # a fourth client exceeds the threshold: idle ones are dropped
    assert set(limiter._clients) == {"d"}
