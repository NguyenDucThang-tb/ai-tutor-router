"""Bounded process-local sliding window with expiry-heap bucket reclamation.

The limiter retains at most ``max_clients`` per-client windows plus one shared
overflow window. A min-heap tracks each bucket's oldest event expiry; expired
buckets are reclaimed on the next request without scanning every client. Requests
from excess unique clients share the overflow allowance, so high cardinality does
not create unbounded keys or disable limiting. Use a shared gateway for
multi-worker limits.
"""

import heapq
import itertools
import threading
import time
from collections import deque


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int, max_clients: int = 4096):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        if max_clients < 1:
            raise ValueError("max_clients must be positive")
        self.limit = requests_per_minute
        self.max_clients = max_clients
        self._overflow_key = object()
        self._events: dict[str | object, deque[float]] = {}
        self._expirations: list[tuple[float, int, str | object]] = []
        self._sequence = itertools.count()
        self._last_timestamp: float | None = None
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            if self._last_timestamp is not None:
                timestamp = max(timestamp, self._last_timestamp)
            self._last_timestamp = timestamp
            self._reclaim_expired(timestamp)

            bucket_key: str | object = key
            events = self._events.get(bucket_key)
            if events is None:
                if self._client_bucket_count() >= self.max_clients:
                    bucket_key = self._overflow_key
                events = self._events.setdefault(bucket_key, deque())
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            if len(events) == 1:
                self._schedule_expiry(bucket_key, events)
            return True

    def _client_bucket_count(self) -> int:
        return len(self._events) - (self._overflow_key in self._events)

    def _schedule_expiry(self, key: str | object, events: deque[float]) -> None:
        heapq.heappush(self._expirations, (events[0] + 60.0, next(self._sequence), key))

    def _reclaim_expired(self, timestamp: float) -> None:
        cutoff = timestamp - 60.0
        while self._expirations and self._expirations[0][0] <= timestamp:
            _, _, key = heapq.heappop(self._expirations)
            events = self._events.get(key)
            if events is None:
                continue
            while events and events[0] <= cutoff:
                events.popleft()
            if events:
                self._schedule_expiry(key, events)
            else:
                del self._events[key]
