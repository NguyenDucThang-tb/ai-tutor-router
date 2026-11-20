from app.rate_limit import InMemoryRateLimiter


def test_more_than_4096_unique_keys_stay_bounded_and_overflow_is_limited():
    limiter = InMemoryRateLimiter(requests_per_minute=1)

    for index in range(4096):
        assert limiter.allow(f"client-{index}", now=10.0)

    assert limiter.allow("client-4096", now=10.0)
    assert not limiter.allow("client-4097", now=10.0)
    assert len(limiter._events) <= limiter.max_clients + 1


def test_overflow_does_not_reset_a_tracked_clients_window():
    limiter = InMemoryRateLimiter(requests_per_minute=1, max_clients=1)

    assert limiter.allow("tracked", now=10.0)
    assert limiter.allow("untracked-1", now=10.0)
    assert not limiter.allow("untracked-2", now=10.0)
    assert not limiter.allow("tracked", now=10.0)
