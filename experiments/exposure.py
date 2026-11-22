"""Conservative monetary exposure calculation for complete provider envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from typing import Protocol

from app.gateway import GatewayRequest
from experiments.config import BenchmarkConfig
from experiments.provider_capabilities import price_provenance_is_current, resolve_capability
from experiments.token_accounting import TokenCounter, UnsupportedTokenCounter


class UnboundedExposureError(ValueError):
    """Raised when a complete request envelope cannot be priced conservatively."""


@dataclass(frozen=True)
class VerifiedExposure:
    amount_micros_usd: int
    provenance: dict


class ExposureCalculator(Protocol):
    def calculate(
        self,
        config: BenchmarkConfig,
        request: GatewayRequest,
        model_identifiers: tuple[str, ...],
    ) -> VerifiedExposure: ...


class ProviderExposureCalculator:
    """Price a bounded full envelope; estimates and incomplete billing fail closed."""

    def __init__(self, token_counter: TokenCounter | None = None):
        self.token_counter = token_counter or UnsupportedTokenCounter()

    def calculate(
        self,
        config: BenchmarkConfig,
        request: GatewayRequest,
        model_identifiers: tuple[str, ...],
    ) -> VerifiedExposure:
        if not model_identifiers:
            raise UnboundedExposureError("request envelope has no provider attempts")
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_content},
        ]
        total_usd = Decimal(0)
        model_evidence = []
        for model_id in model_identifiers:
            capability = resolve_capability(model_id)
            if capability is None or capability.model_id != model_id:
                raise UnboundedExposureError(f"unverified model identity: {model_id}")
            if capability.status != "documented_current":
                raise UnboundedExposureError(f"model is not documented current: {model_id}")
            today = datetime.now(UTC).date()
            if not price_provenance_is_current(capability, today=today):
                raise UnboundedExposureError(f"price provenance is stale: {model_id}")
            if not capability.all_billable_categories_supported:
                raise UnboundedExposureError(f"incomplete billable categories: {model_id}")
            if (
                capability.output_token_bound is None
                or config.max_tokens > capability.output_token_bound
            ):
                raise UnboundedExposureError(f"missing verified output bound: {model_id}")
            count = self.token_counter.count_messages(model_id, messages)
            if not count.supports_admission:
                raise UnboundedExposureError(f"input token upper bound unavailable: {model_id}")
            # Uncached input pricing is conservative for cached-token discounts.
            total_usd += (
                Decimal(count.input_tokens) * capability.input_usd_per_million
                + Decimal(config.max_tokens) * capability.output_usd_per_million
            ) / Decimal(1_000_000)
            model_evidence.append(
                {
                    "model_id": model_id,
                    "pricing_version": capability.pricing_version,
                    "input_token_method": count.method,
                    "input_token_version": count.version,
                    "input_tokens_upper_bound": count.input_tokens,
                    "output_tokens_upper_bound": config.max_tokens,
                    "input_price_per_million": str(capability.input_usd_per_million),
                    "output_price_per_million": str(capability.output_usd_per_million),
                    "billing_categories_complete": True,
                }
            )
        micros = int((total_usd * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
        return VerifiedExposure(
            amount_micros_usd=micros,
            provenance={
                "method": "verified_full_envelope_upper_bound_v1",
                "attempt_model_identifiers": list(model_identifiers),
                "models": model_evidence,
                "amount_micros_usd": micros,
            },
        )


def attempt_model_envelope(config: BenchmarkConfig, tier: str) -> tuple[str, ...]:
    model = config.small_model_identifier if tier == "small" else config.strong_model_identifier
    fallback = (
        config.small_fallback_identifier if tier == "small" else config.strong_fallback_identifier
    )
    primary_attempts = config.max_retries + 1 if config.mode == "real" else 1
    models = [model] * primary_attempts
    if config.mode == "real" and config.fallback_policy == "operational":
        if not fallback:
            raise UnboundedExposureError("operational fallback has no exact model identifier")
        models.append(fallback)
    return tuple(models)
