"""Small process-local sliding-window limiter. Use a shared gateway for multi-worker limits."""

import threading
import time
from collections import defaultdict, deque


class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.limit = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - 60.0
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            # Periodically discard idle client buckets to bound memory use.
            if len(self._events) > 4096:
                for stale_key in [
                    k for k, values in self._events.items() if not values or values[-1] <= cutoff
                ]:
                    self._events.pop(stale_key, None)
            return True
