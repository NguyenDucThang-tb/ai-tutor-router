"""Pure ASGI request correlation, body-size check, and process-local rate limit."""

import json
import re
import time
import uuid

from app.rate_limit import InMemoryRateLimiter

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestContextMiddleware:
    def __init__(self, app, limiter: InMemoryRateLimiter, max_body_bytes: int = 65536):
        self.app = app
        self.limiter = limiter
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if REQUEST_ID_PATTERN.fullmatch(supplied) else str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["request_started_at"] = time.perf_counter()

        async def reject(status: int, code: str, message: str, extra_headers=()):
            body = json.dumps(
                {"error": {"code": code, "message": message, "request_id": request_id}}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"x-request-id", request_id.encode()),
                        *extra_headers,
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        path = scope.get("path", "")
        method = scope.get("method", "")
        if path == "/v1/tutor/ask" and method == "POST":
            length = headers.get(b"content-length", b"")
            if length.isdigit() and int(length) > self.max_body_bytes:
                await reject(413, "request_too_large", "Request body is too large")
                return
            client = scope.get("client")
            client_host = str(client[0]) if client else "unknown"
            if not self.limiter.allow(client_host):
                await reject(
                    429,
                    "rate_limit_exceeded",
                    "Request rate limit exceeded",
                    ((b"retry-after", b"60"),),
                )
                return

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)
