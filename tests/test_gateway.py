import asyncio
from types import SimpleNamespace

import pytest

from app.errors import ErrorCategory, GatewayError
from app.gateway import GatewayRequest, LiteLLMGateway, MockGateway


def response(text="answer", usage=True, cost=0.004, model="provider-model"):
    usage_obj = (
        SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10) if usage else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=usage_obj,
        _hidden_params={"response_cost": cost} if cost is not None else {},
        model=model,
    )


def req(**overrides):
    values = {
        "request_id": "req-123",
        "model_identifier": "provider/small",
        "system_prompt": "system",
        "user_content": "hello",
        "timeout_seconds": 1,
        "max_retries": 1,
        "retry_delay_seconds": 0,
        "fallback_model_identifier": None,
        "budget_limit_usd": None,
    }
    values.update(overrides)
    return GatewayRequest(**values)


class FakeCompletion:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class ServiceUnavailableError(Exception):
    status_code = 503


class AuthenticationError(Exception):
    status_code = 401


class RateLimitError(Exception):
    status_code = 429


class ChargedServiceUnavailable(ServiceUnavailableError):
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0, total_tokens=5)
    response_cost = 0.001


def test_mock_gateway_is_deterministic_unknown_cost_no_usage():
    result = asyncio.run(MockGateway("MOCK: fixed").complete(req()))
    assert result.text == "MOCK: fixed"
    assert result.provider == "mock"
    assert result.request_id == "req-123"
    assert result.usage is None
    assert result.cost.calculated_usd is None
    assert result.attempts[0].request_id == "req-123"


def test_litellm_success_extracts_usage_calculated_cost_and_disables_nested_retries():
    fake = FakeCompletion([response()])
    result = asyncio.run(LiteLLMGateway("not-logged-secret", completion=fake).complete(req()))
    assert result.text == "answer"
    assert result.model_identifier == "provider-model"
    assert result.usage.total_tokens == 10
    assert result.cost.calculated_usd == 0.004
    assert result.cost.actual_billed_usd is None
    assert result.attempts[0].request_id == "req-123"
    assert fake.calls[0]["metadata"]["request_id"] == "req-123"
    assert fake.calls[0]["num_retries"] == fake.calls[0]["max_retries"] == 0


def test_missing_usage_and_unknown_price_are_explicit():
    fake = FakeCompletion([response(usage=False, cost=None)])
    result = asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert result.usage is None
    assert result.cost.status == "unknown"
    assert result.cost.calculated_usd is None


def test_timeout_is_typed_and_retry_can_succeed():
    class TimeoutOnce:
        def __init__(self):
            self.calls = []

        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                await asyncio.sleep(0.05)
            return response()

    fake = TimeoutOnce()
    result = asyncio.run(
        LiteLLMGateway("key", completion=fake).complete(req(timeout_seconds=0.005))
    )
    assert len(fake.calls) == 2
    assert result.attempts[0].error_category == ErrorCategory.TIMEOUT
    assert result.request_id == "req-123"


def test_transient_failure_retries_then_succeeds():
    fake = FakeCompletion([ServiceUnavailableError("private provider text"), response()])
    result = asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert len(fake.calls) == 2
    assert result.attempts[0].error_category == ErrorCategory.TEMPORARY_PROVIDER_FAILURE
    assert result.attempts[1].status == "success"


def test_retry_exhaustion_has_bounded_attempts():
    fake = FakeCompletion([ServiceUnavailableError(), ServiceUnavailableError()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert raised.value.category == ErrorCategory.TEMPORARY_PROVIDER_FAILURE
    assert len(raised.value.attempts) == len(fake.calls) == 2
    assert raised.value.cost.unknown_attempt_count == 2


def test_all_attempts_failing_preserves_aggregate_and_last_attempt_cost():
    fake = FakeCompletion([ChargedServiceUnavailable(), ChargedServiceUnavailable()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert len(raised.value.attempts) == 2
    assert raised.value.cost.calculated_usd == 0.002
    assert raised.value.cost.attempt_count == 2
    assert raised.value.final_attempt_cost.calculated_usd == 0.001


def test_failed_attempt_usage_and_cost_are_retained():
    fake = FakeCompletion([ChargedServiceUnavailable()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(LiteLLMGateway("key", completion=fake).complete(req(max_retries=0)))
    attempt = raised.value.attempts[0]
    assert attempt.usage.total_tokens == 5
    assert attempt.cost.calculated_usd == 0.001


def test_paid_failed_attempt_cost_is_included_in_success_aggregate():
    fake = FakeCompletion([ChargedServiceUnavailable(), response(cost=0.004)])
    result = asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert result.cost.calculated_usd == 0.005
    assert result.cost.known_calculated_usd == 0.005
    assert result.cost.attempt_count == 2
    assert result.cost.unknown_attempt_count == 0
    assert result.final_attempt_cost.calculated_usd == 0.004


def test_retry_then_fallback_aggregates_every_attempt_cost():
    fake = FakeCompletion(
        [ChargedServiceUnavailable(), ChargedServiceUnavailable(), response(cost=0.006)]
    )
    result = asyncio.run(
        LiteLLMGateway("key", completion=fake).complete(
            req(max_retries=1, fallback_model_identifier="provider/backup")
        )
    )
    assert [call["model"] for call in fake.calls] == [
        "provider/small",
        "provider/small",
        "provider/backup",
    ]
    assert result.cost.calculated_usd == pytest.approx(0.008)
    assert result.final_attempt_cost.calculated_usd == 0.006
    assert result.cost.attempt_count == 3


def test_unknown_attempt_cost_keeps_partial_total_uncertain():
    fake = FakeCompletion([ChargedServiceUnavailable(), response(cost=None)])
    result = asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert result.cost.status == "partial"
    assert result.cost.calculated_usd is None
    assert result.cost.known_calculated_usd == 0.001
    assert result.cost.unknown_attempt_count == 1
    assert result.final_attempt_cost.status == "unknown"


def test_fallback_is_separate_single_attempt():
    fake = FakeCompletion([ServiceUnavailableError(), response(model="unused")])
    result = asyncio.run(
        LiteLLMGateway("key", completion=fake).complete(
            req(max_retries=0, fallback_model_identifier="provider/backup")
        )
    )
    assert [call["model"] for call in fake.calls] == ["provider/small", "provider/backup"]
    assert [attempt.fallback for attempt in result.attempts] == [False, True]


def test_fallback_failure_is_bounded_and_reported():
    fake = FakeCompletion([ServiceUnavailableError(), RateLimitError()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(
            LiteLLMGateway("key", completion=fake).complete(
                req(max_retries=0, fallback_model_identifier="provider/backup")
            )
        )
    assert raised.value.category == ErrorCategory.RATE_LIMIT
    assert len(raised.value.attempts) == 2


def test_authentication_failure_is_not_retried_or_fallen_back():
    fake = FakeCompletion([AuthenticationError("secret"), response()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(
            LiteLLMGateway("key", completion=fake).complete(
                req(fallback_model_identifier="provider/backup")
            )
        )
    assert raised.value.category == ErrorCategory.AUTHENTICATION_FAILURE
    assert len(fake.calls) == 1


def test_invalid_response_is_not_retried():
    fake = FakeCompletion([response(text="")])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(LiteLLMGateway("key", completion=fake).complete(req()))
    assert raised.value.category == ErrorCategory.INVALID_PROVIDER_RESPONSE
    assert len(fake.calls) == 1


def test_budget_limit_refuses_unknown_pricing_before_provider_call():
    fake = FakeCompletion([response()])
    with pytest.raises(GatewayError) as raised:
        asyncio.run(LiteLLMGateway("key", completion=fake).complete(req(budget_limit_usd=0.01)))
    assert raised.value.category == ErrorCategory.BUDGET_EXCEEDED
    assert not fake.calls
