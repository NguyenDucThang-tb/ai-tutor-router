"""Central paid-request admission; unknown external limits always block."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from experiments.config import BenchmarkConfig
from experiments.model_catalog import resolve_model
from experiments.provider_capabilities import price_provenance_is_current, resolve_capability
from experiments.token_accounting import UnsupportedTokenCounter


@dataclass(frozen=True)
class GateDecision:
    status: Literal["PASS", "FAIL", "BLOCKED"]
    reason_codes: tuple[str, ...]
    details: tuple[str, ...]


def assess_paid_execution(config: BenchmarkConfig, *, gateway=None) -> GateDecision:
    """Assess structural controls; gateway injection is never an authorization."""
    reasons: list[str] = []
    failures: list[str] = []
    details: list[str] = []
    if config.mode != "real" or config.provider != "litellm":
        failures.append("paid_mode_provider_mismatch")
    if not config.allow_real_provider or not config.experiment_approval_id:
        failures.append("explicit_experiment_approval_missing")

    for model_id in (config.small_model_identifier, config.strong_model_identifier):
        capability = resolve_capability(model_id)
        if capability is None:
            reasons.append("provider_model_capability_unverified")
            details.append(f"{model_id}: no current capability record")
            continue
        if capability.status != "documented_current":
            reasons.append("provider_model_not_current")
        catalog_model = resolve_model(model_id)
        if catalog_model is None or catalog_model.model_id != capability.model_id:
            reasons.append("provider_model_identity_mismatch")
        elif catalog_model.status != "documented_current":
            reasons.append("provider_model_not_current")
        today = datetime.now(UTC).date()
        if not price_provenance_is_current(capability, today=today):
            reasons.append("price_provenance_expired")
        if capability.account_access != "verified":
            reasons.append("provider_account_access_unverified")
        if capability.input_token_method is None:
            reasons.append("input_token_accounting_unverified")
        if capability.output_token_bound is None:
            reasons.append("output_token_bound_missing")
        if (
            not capability.pricing_version
            or capability.input_usd_per_million < 0
            or capability.output_usd_per_million < 0
        ):
            reasons.append("pricing_provenance_incomplete")
        if not capability.all_billable_categories_supported:
            reasons.append("billable_categories_incomplete")
        if not capability.supports_provable_run_bound:
            reasons.append("external_run_spend_bound_unverified")
        details.append(f"{model_id}: {capability.limitation}")

    token_count = UnsupportedTokenCounter().count_messages(config.small_model_identifier, [])
    if not token_count.supports_admission:
        reasons.append("exact_request_token_count_unavailable")
    if config.per_request_budget_usd is None:
        failures.append("per_request_budget_missing")
    if config.max_run_budget_usd is None:
        failures.append("run_budget_missing")
    if config.max_requests < 2 * config.max_samples * (config.max_retries + 1):
        failures.append("request_cap_insufficient")
    if (config.max_retries or config.retry_failed or config.rerun_completed) and not (
        config.allow_paid_retries and config.paid_retry_approval_id
    ):
        failures.append("paid_retry_approval_missing")
    if config.allow_uncertain_reissue and not config.uncertain_reissue_approval_id:
        failures.append("uncertain_reissue_approval_missing")
    if config.fallback_policy != "disabled":
        failures.append("paid_fallback_not_approved_by_gate")

    # Injected adapters are useful for offline simulation but can never establish
    # production provider identity, billing scope, or cap configuration.
    if gateway is not None:
        reasons.append("injected_gateway_not_trusted_for_paid_execution")
    if failures:
        return GateDecision("FAIL", tuple(sorted(set(failures + reasons))), tuple(details))
    if reasons:
        return GateDecision("BLOCKED", tuple(sorted(set(reasons))), tuple(details))
    return GateDecision("PASS", (), tuple(details))
