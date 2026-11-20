from app.rate_limit import InMemoryRateLimiter


def test_more_than_4096_unique_keys_stay_bounded_and_overflow_is_limited():
    limiter = InMemoryRateLimiter(requests_per_minute=1)

    for index in range(4096):
        assert limiter.allow(f"client-{index}", now=10.0)

    assert limiter.allow("client-4096", now=10.0)
    assert not limiter.allow("client-4097", now=10.0)
    assert len(limiter._events) <= limiter.max_clients + 1
    assert len(limiter._expirations) <= limiter.max_clients + 1

    # All original clients stay inactive; one later request globally reclaims them.
    assert limiter.allow("after-expiry", now=70.0)
    assert limiter._client_bucket_count() == 1
    assert limiter._overflow_key not in limiter._events
    assert len(limiter._events) == len(limiter._expirations) == 1


def test_overflow_does_not_reset_a_tracked_clients_window():
    limiter = InMemoryRateLimiter(requests_per_minute=1, max_clients=1)

    assert limiter.allow("tracked", now=10.0)
    assert limiter.allow("untracked-1", now=10.0)
    assert not limiter.allow("untracked-2", now=10.0)
    assert not limiter.allow("tracked", now=10.0)


def test_expired_slots_are_reclaimed_without_resetting_an_active_window():
    limiter = InMemoryRateLimiter(requests_per_minute=2, max_clients=2)

    assert limiter.allow("inactive", now=0.0)
    assert limiter.allow("active", now=0.0)
    assert limiter.allow("active", now=50.0)
    assert limiter.allow("overflow-1", now=50.0)
    assert len(limiter._events) == limiter.max_clients + 1
    assert limiter._overflow_key in limiter._events

    assert limiter.allow("new-client", now=61.0)
    assert "inactive" not in limiter._events
    assert "new-client" in limiter._events
    assert "active" in limiter._events
    assert limiter.allow("active", now=61.0)
    assert not limiter.allow("active", now=62.0)
    assert len(limiter._events) <= limiter.max_clients + 1


def test_repeated_expiry_cleanup_keeps_bucket_and_heap_state_consistent():
    limiter = InMemoryRateLimiter(requests_per_minute=1, max_clients=2)

    for index in range(100):
        assert limiter.allow(f"client-{index}", now=index * 60.0)
        assert limiter._client_bucket_count() == 1
        assert len(limiter._events) == len(limiter._expirations) == 1
        assert len(limiter._events) <= limiter.max_clients + 1
