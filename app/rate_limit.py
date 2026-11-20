"""Bounded process-local sliding window; excess unique clients share one overflow bucket.

The limiter retains at most ``max_clients`` per-client windows plus one shared window.
This keeps memory bounded without disabling limiting at high cardinality. Requests
from untracked clients consume the same overflow allowance, so they cannot create
unbounded keys or reset a per-client window by changing addresses. Use a shared
gateway for multi-worker limits.
"""

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
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - 60.0
        with self._lock:
            events = self._events.get(key)
            if events is not None:
                self._prune(events, cutoff)
                if not events:
                    del self._events[key]
                    events = None

            if events is None:
                if key in self._events:
                    events = self._events[key]
                elif self._client_bucket_count() < self.max_clients:
                    events = self._events.setdefault(key, deque())
                else:
                    events = self._events.setdefault(self._overflow_key, deque())
                    self._prune(events, cutoff)
                    if not events:
                        events.append(timestamp)
                        return True
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            return True

    def _client_bucket_count(self) -> int:
        return len(self._events) - (self._overflow_key in self._events)

    @staticmethod
    def _prune(events: deque[float], cutoff: float) -> None:
        while events and events[0] <= cutoff:
            events.popleft()
