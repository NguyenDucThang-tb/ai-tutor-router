"""Async mock and LiteLLM gateways with bounded, observable attempts."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.errors import ErrorCategory, GatewayError, ProviderCallError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageInfo:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class CostInfo:
    status: str = "unknown"
    estimated_usd: float | None = None
    calculated_usd: float | None = None
    provider_reported_usd: float | None = None
    actual_billed_usd: float | None = None
    source: str | None = None
    known_estimated_usd: float | None = None
    known_calculated_usd: float | None = None
    known_provider_reported_usd: float | None = None
    known_actual_billed_usd: float | None = None
    attempt_count: int = 0
    unknown_attempt_count: int = 0


@dataclass(frozen=True)
class ExecutionAttempt:
    request_id: str
    attempt_number: int
    model_identifier: str
    provider: str
    fallback: bool
    latency_ms: float
    status: str
    error_category: ErrorCategory | None = None
    usage: UsageInfo | None = None
    cost: CostInfo = field(default_factory=CostInfo)


@dataclass(frozen=True)
class GatewayRequest:
    request_id: str
    model_identifier: str
    system_prompt: str
    user_content: str
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float
    fallback_model_identifier: str | None = None
    budget_limit_usd: float | None = None


@dataclass(frozen=True)
class GatewayResult:
    request_id: str
    text: str
    model_identifier: str
    provider: str
    usage: UsageInfo | None
    cost: CostInfo
    latency_ms: float
    attempts: tuple[ExecutionAttempt, ...]
    final_attempt_cost: CostInfo = field(default_factory=CostInfo)


def aggregate_attempt_costs(
    attempts: tuple[ExecutionAttempt, ...] | list[ExecutionAttempt],
) -> CostInfo:
    """Return complete totals plus known subtotals and explicit missing-cost counts."""
    if not attempts:
        return CostInfo()

    fields = (
        ("estimated_usd", "known_estimated_usd"),
        ("calculated_usd", "known_calculated_usd"),
        ("provider_reported_usd", "known_provider_reported_usd"),
        ("actual_billed_usd", "known_actual_billed_usd"),
    )
    totals: dict[str, float | None] = {}
    known_totals: dict[str, float | None] = {}
    for field_name, known_name in fields:
        values = [getattr(attempt.cost, field_name) for attempt in attempts]
        known = [float(value) for value in values if value is not None]
        totals[field_name] = sum(known) if len(known) == len(attempts) else None
        known_totals[known_name] = sum(known) if known else None

    calculated_values = [attempt.cost.calculated_usd for attempt in attempts]
    known_calculated_count = sum(value is not None for value in calculated_values)
    unknown_count = len(attempts) - known_calculated_count
    status = (
        "available" if unknown_count == 0 else "partial" if known_calculated_count else "unknown"
    )
    sources = [attempt.cost.source for attempt in attempts]
    source = (
        f"aggregate:{sources[0]}"
        if sources[0] is not None and all(value == sources[0] for value in sources)
        else None
    )
    return CostInfo(
        status=status,
        **totals,
        **known_totals,
        attempt_count=len(attempts),
        unknown_attempt_count=unknown_count,
        source=source,
    )


class Gateway(Protocol):
    provider: str

    async def complete(self, request: GatewayRequest) -> GatewayResult: ...


class MockGateway:
    """Deterministic response stub. It provides no mathematical correctness evidence."""

    provider = "mock"

    def __init__(self, response_text: str = "MOCK RESPONSE: no real model was called"):
        self.response_text = response_text

    async def complete(self, request: GatewayRequest) -> GatewayResult:
        start = time.perf_counter()
        attempt = ExecutionAttempt(
            request_id=request.request_id,
            attempt_number=1,
            model_identifier=request.model_identifier,
            provider=self.provider,
            fallback=False,
            latency_ms=0.0,
            status="success",
        )
        return GatewayResult(
            request_id=request.request_id,
            text=self.response_text,
            model_identifier=request.model_identifier,
            provider=self.provider,
            usage=None,
            cost=aggregate_attempt_costs((attempt,)),
            latency_ms=(time.perf_counter() - start) * 1000,
            attempts=(attempt,),
            final_attempt_cost=attempt.cost,
        )


class CompletionCallable(Protocol):
    def __call__(self, **kwargs: Any) -> Awaitable[Any] | Any: ...


class LiteLLMGateway:
    """Calls LiteLLM's async SDK; this class owns the only retry loop.

    `num_retries=0` and `max_retries=0` are passed to LiteLLM/provider clients so
    these bounded retries are not multiplied by a second adapter retry loop.
    """

    provider = "litellm"

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
        completion: CompletionCallable | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self._completion = completion

    def _get_completion(self) -> CompletionCallable:
        if self._completion is None:
            from litellm import acompletion

            self._completion = acompletion
        return self._completion

    async def complete(self, request: GatewayRequest) -> GatewayResult:
        started = time.perf_counter()
        attempts: list[ExecutionAttempt] = []
        if request.budget_limit_usd is not None:
            # No token-price estimator is configured in Phase 1. Refuse the paid call
            # instead of treating an unknown estimate as zero or checking post-spend.
            raise GatewayError(
                ErrorCategory.BUDGET_EXCEEDED,
                "pricing estimate unavailable; refusing request under an active budget limit",
            )
        models: list[tuple[str, bool]] = [(request.model_identifier, False)]
        if request.fallback_model_identifier:
            if request.fallback_model_identifier == request.model_identifier:
                raise ValueError("fallback model must differ from the primary model")
            models.append((request.fallback_model_identifier, True))
        # At most one configured fallback is tried; never recurse into another fallback.
        for model_index, (model, is_fallback) in enumerate(models):
            retries = request.max_retries if model_index == 0 else 0
            for retry_index in range(retries + 1):
                attempt_number = len(attempts) + 1
                attempt_started = time.perf_counter()
                try:
                    response = await self._invoke(model, request)
                    parsed = self._parse_response(response, model)
                    executed_model, text, usage, cost = parsed
                    attempt = ExecutionAttempt(
                        request_id=request.request_id,
                        attempt_number=attempt_number,
                        model_identifier=executed_model,
                        provider=self.provider,
                        fallback=is_fallback,
                        latency_ms=(time.perf_counter() - attempt_started) * 1000,
                        status="success",
                        usage=usage,
                        cost=cost,
                    )
                    attempts.append(attempt)
                    return GatewayResult(
                        request_id=request.request_id,
                        text=text,
                        model_identifier=executed_model,
                        provider=self.provider,
                        usage=usage,
                        cost=aggregate_attempt_costs(attempts),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        attempts=tuple(attempts),
                        final_attempt_cost=cost,
                    )
                except GatewayError:
                    raise
                except Exception as exc:  # noqa: BLE001 - LiteLLM providers expose many exception subclasses.
                    categorized = self._classify_exception(exc)
                    usage, cost = self._failure_metadata(exc)
                    attempts.append(
                        ExecutionAttempt(
                            request_id=request.request_id,
                            attempt_number=attempt_number,
                            model_identifier=model,
                            provider=self.provider,
                            fallback=is_fallback,
                            latency_ms=(time.perf_counter() - attempt_started) * 1000,
                            status="failed",
                            error_category=categorized,
                            usage=usage,
                            cost=cost,
                        )
                    )
                    logger.warning(
                        "provider_attempt_failed",
                        extra={
                            "request_id": request.request_id,
                            "provider": self.provider,
                            "model_identifier": model,
                            "attempt_number": attempt_number,
                            "error_category": categorized.value,
                            "latency_ms": (time.perf_counter() - attempt_started) * 1000,
                            "status": "failed",
                            "prompt_tokens": usage.prompt_tokens if usage else None,
                            "completion_tokens": usage.completion_tokens if usage else None,
                            "total_tokens": usage.total_tokens if usage else None,
                            "cost_usd": cost.calculated_usd,
                        },
                    )
                    retryable = categorized in {
                        ErrorCategory.RATE_LIMIT,
                        ErrorCategory.TIMEOUT,
                        ErrorCategory.TEMPORARY_PROVIDER_FAILURE,
                    }
                    if retryable and retry_index < retries:
                        if request.retry_delay_seconds:
                            await asyncio.sleep(request.retry_delay_seconds)
                        continue
                    # Configured failover is allowed only for transient failures or an
                    # unsupported primary model; auth/invalid-response failures stop here.
                    can_fallback = (
                        not is_fallback
                        and model_index == 0
                        and request.fallback_model_identifier is not None
                        and categorized
                        in {
                            ErrorCategory.RATE_LIMIT,
                            ErrorCategory.TIMEOUT,
                            ErrorCategory.TEMPORARY_PROVIDER_FAILURE,
                            ErrorCategory.UNSUPPORTED_MODEL,
                        }
                    )
                    if can_fallback:
                        break
                    raise GatewayError(
                        categorized,
                        categorized.value,
                        attempts,
                        aggregate_attempt_costs(attempts),
                        attempts[-1].cost if attempts else None,
                    ) from None
        category = attempts[-1].error_category or ErrorCategory.TEMPORARY_PROVIDER_FAILURE
        raise GatewayError(
            category,
            category.value,
            attempts,
            aggregate_attempt_costs(attempts),
            attempts[-1].cost if attempts else None,
        )

    async def _invoke(self, model: str, request: GatewayRequest) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_content},
            ],
            "timeout": request.timeout_seconds,
            "num_retries": 0,
            "max_retries": 0,
            "api_key": self.api_key,
            "metadata": {"request_id": request.request_id},
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        completion = self._get_completion()

        async def call():
            value = completion(**kwargs)
            if inspect.isawaitable(value):
                return await value
            return value

        try:
            return await asyncio.wait_for(call(), timeout=request.timeout_seconds)
        except TimeoutError as exc:
            raise ProviderCallError(ErrorCategory.TIMEOUT) from exc

    @classmethod
    def _parse_response(
        cls, response: Any, requested_model: str
    ) -> tuple[str, str, UsageInfo | None, CostInfo]:
        try:
            choices = response.choices
            content = choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError):
            raise ProviderCallError(ErrorCategory.INVALID_PROVIDER_RESPONSE) from None
        if not isinstance(content, str) or not content.strip():
            raise ProviderCallError(ErrorCategory.INVALID_PROVIDER_RESPONSE)
        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = UsageInfo(
                prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
                completion_tokens=getattr(usage_obj, "completion_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
            )
        hidden = getattr(response, "_hidden_params", {}) or {}
        calculated = hidden.get("response_cost") if isinstance(hidden, dict) else None
        actual_model = getattr(response, "model", None)
        if not isinstance(actual_model, str) or not actual_model.strip():
            actual_model = requested_model
        cost = CostInfo(
            status="available" if isinstance(calculated, (int, float)) else "unknown",
            calculated_usd=float(calculated) if isinstance(calculated, (int, float)) else None,
            source="litellm_calculated" if isinstance(calculated, (int, float)) else None,
        )
        return actual_model, content, usage, cost

    @staticmethod
    def _classify_exception(exc: Exception) -> ErrorCategory:
        if isinstance(exc, ProviderCallError):
            return exc.category
        name = type(exc).__name__.lower()
        status = getattr(exc, "status_code", None)
        if "auth" in name or "permission" in name or status in {401, 403}:
            return ErrorCategory.AUTHENTICATION_FAILURE
        if "ratelimit" in name or status == 429:
            return ErrorCategory.RATE_LIMIT
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name:
            return ErrorCategory.TIMEOUT
        if status == 404 or "notfound" in name or "unsupported" in name:
            return ErrorCategory.UNSUPPORTED_MODEL
        if status is not None and 400 <= status < 500:
            return ErrorCategory.INVALID_PROVIDER_RESPONSE
        if "connection" in name or "serviceunavailable" in name or "internalserver" in name:
            return ErrorCategory.TEMPORARY_PROVIDER_FAILURE
        if status is not None and status >= 500:
            return ErrorCategory.TEMPORARY_PROVIDER_FAILURE
        return ErrorCategory.TEMPORARY_PROVIDER_FAILURE

    @classmethod
    def _failure_metadata(cls, exc: Exception) -> tuple[UsageInfo | None, CostInfo]:
        usage_obj = getattr(exc, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = UsageInfo(
                prompt_tokens=getattr(usage_obj, "prompt_tokens", None),
                completion_tokens=getattr(usage_obj, "completion_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
            )
        calculated = getattr(exc, "response_cost", None)
        cost = CostInfo(
            status="available" if isinstance(calculated, (int, float)) else "unknown",
            calculated_usd=float(calculated) if isinstance(calculated, (int, float)) else None,
            source="litellm_calculated" if isinstance(calculated, (int, float)) else None,
        )
        return usage, cost
