"""Immutable, signed paid-experiment approval binding.

The artifact is only an additional authorization check. It cannot establish an
account-level spend limit or make an otherwise blocked provider gate pass.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.schemas import DatasetManifest, DatasetRecord
from experiments.config import BenchmarkConfig
from experiments.model_catalog import resolve_model
from experiments.prompts import PROMPT_SHA256, PROMPT_VERSION
from experiments.provider_capabilities import resolve_capability

APPROVAL_KEY_ENV = "TUTOR_PAID_APPROVAL_HMAC_KEY"


def approval_context(
    config: BenchmarkConfig,
    records: list[DatasetRecord],
    manifest: DatasetManifest,
    split: str,
    collection_plan: dict | None,
) -> dict[str, Any]:
    """Return material execution settings, excluding mutable approval controls."""
    attempts_per_request = config.max_retries + 1
    if config.fallback_policy == "operational":
        attempts_per_request += 1
    pricing = {}
    for tier, model_id in (
        ("small", config.small_model_identifier),
        ("strong", config.strong_model_identifier),
    ):
        model = resolve_model(model_id)
        capability = resolve_capability(model_id)
        pricing[tier] = {
            "model_id": model_id,
            "input_usd_per_million": str(model.input_usd_per_million) if model else None,
            "output_usd_per_million": str(model.output_usd_per_million) if model else None,
            "pricing_url": model.pricing_url if model else None,
            "pricing_retrieved": model.retrieved.isoformat() if model else None,
            "pricing_effective_date": model.pricing_effective_date if model else None,
            "capability_price_version": capability.pricing_version if capability else None,
        }
    return {
        "execution_mode": config.mode,
        "provider": config.provider,
        "provider_region": config.provider_region,
        "small_model_id": config.small_model_identifier,
        "strong_model_id": config.strong_model_identifier,
        "dataset": manifest.dataset,
        "dataset_revision": manifest.revision,
        "dataset_content_sha256": manifest.dataset_content_sha256,
        "split": split,
        "sample_ids": [row.sample_id for row in records],
        "sample_count": len(records),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(row.sample_id for row in records).encode()
        ).hexdigest(),
        "collection_plan_sha256": (
            collection_plan.get("plan_sha256") if collection_plan is not None else None
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "max_primary_requests": len(records) * 2,
        "maximum_total_attempts": len(records) * 2 * attempts_per_request,
        "max_requests": config.max_requests,
        "max_run_budget_usd": config.max_run_budget_usd,
        "per_request_budget_usd": config.per_request_budget_usd,
        "max_retries": config.max_retries,
        "retry_delay_seconds": config.retry_delay_seconds,
        "retry_failed": config.retry_failed,
        "rerun_completed": config.rerun_completed,
        "retry_interrupted": config.retry_interrupted,
        "allow_uncertain_reissue": config.allow_uncertain_reissue,
        "fallback_policy": config.fallback_policy,
        "small_fallback_identifier": config.small_fallback_identifier,
        "strong_fallback_identifier": config.strong_fallback_identifier,
        "concurrency": config.concurrency,
        "timeout_seconds": config.timeout_seconds,
        "router_strategy": config.router_strategy,
        "pricing_source": config.pricing_source,
        "pricing_effective_date": config.pricing_effective_date,
        "pricing_policy": pricing,
    }


def context_digest(context: dict[str, Any]) -> str:
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def approval_signature_payload(artifact: dict[str, Any]) -> bytes:
    approval = artifact["approval"]
    signed = {
        "schema_version": artifact["schema_version"],
        "immutable_config_sha256": artifact["immutable_config_sha256"],
        "approval_id": approval["approval_id"],
        "approved_by": approval["approved_by"],
        "approval_timestamp": approval["approval_timestamp"],
        "status": approval["status"],
    }
    return json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()


def verify_approval_artifact(
    path: Path | str | None,
    *,
    config: BenchmarkConfig,
    records: list[DatasetRecord],
    manifest: DatasetManifest,
    split: str,
    collection_plan: dict | None,
    signing_key: str | None = None,
) -> tuple[bool, str]:
    if path is None:
        return False, "paid approval artifact is not configured"
    try:
        artifact = json.loads(Path(path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"paid approval artifact cannot be read: {type(exc).__name__}"
    if not isinstance(artifact, dict) or artifact.get("schema_version") != "paid-approval-v1":
        return False, "paid approval artifact schema is unsupported"
    context = approval_context(config, records, manifest, split, collection_plan)
    digest = context_digest(context)
    if artifact.get("immutable_config") != context:
        return False, "paid approval material configuration does not match this run"
    if artifact.get("immutable_config_sha256") != digest:
        return False, "paid approval configuration digest does not match this run"
    approval = artifact.get("approval")
    if not isinstance(approval, dict) or approval.get("status") != "APPROVED":
        return False, "paid approval status is not APPROVED"
    for name in ("approval_id", "approved_by", "approval_timestamp", "signature"):
        if not isinstance(approval.get(name), str) or not approval[name].strip():
            return False, f"paid approval {name} is missing"
    if not config.allow_real_provider or approval["approval_id"] != config.experiment_approval_id:
        return False, "paid approval does not match the configured experiment approval reference"
    try:
        timestamp = datetime.fromisoformat(approval["approval_timestamp"])
    except ValueError:
        return False, "paid approval timestamp is invalid"
    if timestamp.tzinfo is None:
        return False, "paid approval timestamp must include a timezone"
    key = signing_key if signing_key is not None else os.environ.get(APPROVAL_KEY_ENV)
    if not key:
        return False, f"trusted signing key {APPROVAL_KEY_ENV} is unavailable"
    if len(key.encode()) < 32:
        return False, f"trusted signing key {APPROVAL_KEY_ENV} must contain at least 32 bytes"
    expected = hmac.new(
        key.encode(), approval_signature_payload(artifact), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, approval["signature"]):
        return False, "paid approval signature is invalid"
    return True, "signed approval matches the exact frozen run configuration"


def write_not_approved_artifact(path: Path, context: dict[str, Any]) -> None:
    """Write a reviewable unsigned draft; its presence never authorizes execution."""
    artifact = {
        "schema_version": "paid-approval-v1",
        "immutable_config": context,
        "immutable_config_sha256": context_digest(context),
        "approval": {
            "approval_id": None,
            "approved_by": None,
            "approval_timestamp": None,
            "status": "NOT_APPROVED",
            "signature": None,
        },
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
