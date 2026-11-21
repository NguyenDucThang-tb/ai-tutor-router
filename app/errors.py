"""Typed gateway/API failures; messages never contain provider exception bodies."""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TEMPORARY_PROVIDER_FAILURE = "temporary_provider_failure"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    UNSUPPORTED_MODEL = "unsupported_model"
    BUDGET_EXCEEDED = "budget_exceeded"


class GatewayError(Exception):
    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        attempts=(),
        cost=None,
        final_attempt_cost=None,
        retry_delay_ms: float = 0.0,
    ):
        super().__init__(message)
        self.category = category
        self.attempts = tuple(attempts)
        self.cost = cost
        self.final_attempt_cost = final_attempt_cost
        self.retry_delay_ms = retry_delay_ms


class ProviderCallError(Exception):
    """Safe internal representation used to classify an adapter exception."""

    def __init__(self, category: ErrorCategory, *, usage=None, cost=None):
        super().__init__(category.value)
        self.category = category
        self.usage = usage
        self.cost = cost
