import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app.errors import ErrorCategory, GatewayError
from app.gateway import CostInfo, ExecutionAttempt, GatewayResult, UsageInfo
from app.main import create_app
from app.settings import Settings


class FakeGateway:
    provider = "mock"

    def __init__(self, error=None):
        self.error = error
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        attempt = ExecutionAttempt(
            request.request_id, 1, request.model_identifier, "mock", False, 0.2, "success"
        )
        return GatewayResult(
            request.request_id,
            "MOCK RESPONSE",
            request.model_identifier,
            "mock",
            None,
            CostInfo(),
            0.2,
            (attempt,),
        )


def make_settings(**kwargs):
    values = {"environment": "test", **kwargs}
    return Settings(_env_file=None, **values)


def make_app(settings=None, gateway=None):
    return create_app(settings or make_settings(), gateway or FakeGateway())


def request(app, method, path, **kwargs):
    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_health_strategies_and_ready_do_not_call_provider():
    gateway = FakeGateway()
    app = make_app(gateway=gateway)
    assert request(app, "GET", "/health/live").json() == {"status": "live"}
    assert request(app, "GET", "/health/ready").json() == {"status": "ready", "provider": "mock"}
    assert request(app, "GET", "/v1/router/strategies").json()["strategies"] == [
        "always_small",
        "always_strong",
        "random",
        "rule",
    ]
    assert not gateway.requests


def test_tutor_lifecycle_and_request_id_propagation():
    gateway = FakeGateway()
    app = make_app(gateway=gateway)
    response = request(
        app,
        "POST",
        "/v1/tutor/ask",
        headers={"X-Request-ID": "stable-123"},
        json={
            "question": "Solve x = 2",
            "task_type": "math",
            "mode": "auto",
            "student_level": "high_school",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert response.headers["X-Request-ID"] == body["request_id"] == "stable-123"
    assert body["selected_tier"] in {"small", "strong"}
    assert body["selected_model_id"] == body["executed_model_id"]
    assert body["router"]["router_strategy"] == "rule"
    assert body["provider"] == "mock"
    assert body["cost"]["status"] == "unknown"
    assert gateway.requests[0].request_id == "stable-123"
    assert gateway.requests[0].system_prompt.startswith("You are a careful mathematics tutor")
    assert gateway.requests[0].user_content == "Solve x = 2"


def test_missing_and_invalid_bearer_tokens_return_structured_401():
    app = make_app(settings=make_settings(auth_enabled=True, api_bearer_token="correct-token"))
    missing = request(app, "POST", "/v1/tutor/ask", json={"question": "2+2"})
    invalid = request(
        app,
        "POST",
        "/v1/tutor/ask",
        headers={"Authorization": "Bearer wrong"},
        json={"question": "2+2"},
    )
    assert missing.status_code == invalid.status_code == 401
    assert missing.json()["error"]["code"] == "authentication_required"
    assert invalid.json()["error"]["request_id"]


def test_development_mode_explicitly_allows_unauthenticated_access():
    app = make_app(settings=make_settings(environment="development"))
    assert request(app, "POST", "/v1/tutor/ask", json={"question": "2+2"}).status_code == 200


def test_invalid_input_and_unknown_parameters_have_consistent_errors():
    app = make_app()
    empty = request(app, "POST", "/v1/tutor/ask", json={"question": ""})
    extra = request(app, "POST", "/v1/tutor/ask", json={"question": "2+2", "prompt": "override"})
    assert empty.status_code == extra.status_code == 422
    assert empty.json()["error"]["code"] == "invalid_request"
    assert empty.json()["error"]["request_id"]
    assert "input" not in str(extra.json())


def test_oversized_http_body_is_rejected_before_parsing():
    response = request(
        make_app(),
        "POST",
        "/v1/tutor/ask",
        content=b" " * 65537,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_rate_limit_is_process_local_and_returns_429():
    app = make_app(settings=make_settings(rate_limit_requests_per_minute=1))
    assert request(app, "POST", "/v1/tutor/ask", json={"question": "2+2"}).status_code == 200
    limited = request(app, "POST", "/v1/tutor/ask", json={"question": "3+3"})
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"
    assert limited.json()["error"]["code"] == "rate_limit_exceeded"


@pytest.mark.parametrize(
    ("category", "status", "code"),
    [
        (ErrorCategory.TIMEOUT, 504, "provider_timeout"),
        (ErrorCategory.TEMPORARY_PROVIDER_FAILURE, 503, "provider_unavailable"),
        (ErrorCategory.AUTHENTICATION_FAILURE, 502, "provider_authentication_failure"),
    ],
)
def test_provider_errors_are_safe_and_structured(category, status, code):
    gateway = FakeGateway(GatewayError(category, "secret provider traceback must not escape"))
    response = request(make_app(gateway=gateway), "POST", "/v1/tutor/ask", json={"question": "2+2"})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "traceback" not in response.text


def test_failed_provider_attempt_metadata_is_preserved_in_error():
    attempt = ExecutionAttempt(
        request_id="failed-req",
        attempt_number=1,
        model_identifier="provider/small",
        provider="litellm",
        fallback=False,
        latency_ms=17.5,
        status="failed",
        error_category=ErrorCategory.RATE_LIMIT,
        usage=UsageInfo(prompt_tokens=9, completion_tokens=2, total_tokens=11),
        cost=CostInfo(status="available", calculated_usd=0.002, source="litellm_calculated"),
    )
    gateway = FakeGateway(
        GatewayError(ErrorCategory.RATE_LIMIT, "private provider text", (attempt,))
    )
    response = request(
        make_app(gateway=gateway),
        "POST",
        "/v1/tutor/ask",
        headers={"X-Request-ID": "failed-req"},
        json={"question": "2+2"},
    )
    assert response.status_code == 503
    failed = response.json()["attempts"][0]
    assert failed["usage"]["total_tokens"] == 11
    assert failed["cost"]["calculated_usd"] == 0.002
    assert failed["error_category"] == "rate_limit"
    assert "private provider text" not in response.text


def test_auth_token_is_not_returned_in_response():
    app = make_app(settings=make_settings(auth_enabled=True, api_bearer_token="very-secret-token"))
    response = request(
        app,
        "POST",
        "/v1/tutor/ask",
        headers={"Authorization": "Bearer very-secret-token"},
        json={"question": "2+2"},
    )
    assert response.status_code == 200
    assert "very-secret-token" not in response.text


def test_student_input_and_bearer_token_are_not_logged(caplog):
    app = make_app(settings=make_settings(auth_enabled=True, api_bearer_token="very-secret-token"))
    question = "private-student-data password=student-private-937"
    request(
        app,
        "POST",
        "/v1/tutor/ask",
        headers={"Authorization": "Bearer very-secret-token"},
        json={"question": question},
    )
    assert question not in caplog.text
    assert "very-secret-token" not in caplog.text


def test_invalid_incoming_request_id_is_replaced():
    response = request(
        make_app(),
        "POST",
        "/v1/tutor/ask",
        headers={"X-Request-ID": "bad id\nsecret"},
        json={"question": "2+2"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "bad id"
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
