"""Offline-only provider metadata and readiness report.

This command deliberately does not import LiteLLM, inspect keys, or contact an
account endpoint. Account verification is unsupported until a reviewed,
non-billable provider endpoint and its side effects are approved.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from experiments.config import load_config
from experiments.model_catalog import resolve_model
from experiments.provider_capabilities import price_provenance_is_current, resolve_capability


def make_report(config_path: Path) -> dict:
    config = load_config(config_path)
    models = []
    documented = True
    for tier, model_id in (
        ("small", config.small_model_identifier),
        ("strong", config.strong_model_identifier),
    ):
        model = resolve_model(model_id)
        capability = resolve_capability(model_id)
        today = date.today()
        valid = bool(
            model
            and capability
            and model.status == "documented_current"
            and capability.status == "documented_current"
            and model.model_id == capability.model_id
            and model.provider == capability.provider
            and model.input_usd_per_million == capability.input_usd_per_million
            and model.output_usd_per_million == capability.output_usd_per_million
            and model.retrieved + timedelta(days=capability.max_price_age_days) >= today
            and price_provenance_is_current(capability, today=today)
        )
        documented = documented and valid
        models.append(
            {
                "tier": tier,
                "model_id": model_id,
                "status": "PROVIDER_DOCUMENTED" if valid else "UNSUPPORTED_OR_STALE",
                "identity_url": capability.identity_url if capability else None,
                "pricing_url": capability.pricing_url if capability else None,
                "pricing_version": capability.pricing_version if capability else None,
                "pricing_current": valid,
                "input_count_method": capability.input_token_method if capability else None,
                "account_access": "NOT_VERIFIED",
                "model_access": "NOT_VERIFIED",
                "billing_scope": "NOT_VERIFIED",
                "complete_billable_categories": bool(
                    capability and capability.all_billable_categories_supported
                ),
                "provable_external_run_bound": bool(
                    capability and capability.supports_provable_run_bound
                ),
            }
        )
    return {
        "verification_mode": "offline",
        "network_calls": 0,
        "secrets_read": False,
        "statuses": {
            "PROVIDER_DOCUMENTED": documented,
            "ACCOUNT_ACCESS_VERIFIED": False,
            "MODEL_ACCESS_VERIFIED": False,
            "BILLING_SCOPE_VERIFIED": False,
            "PAID_EXECUTION_AUTHORIZED": False,
        },
        "models": models,
        "limitations": [
            "Offline metadata does not verify model access or effective prices.",
            "No account endpoint or inference endpoint was contacted.",
            "Provider billing controls are not run-scoped guarantees.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline provider readiness verification")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    if not args.offline:
        parser.error(
            "account verification is unsupported; pass --offline (no network mode is implemented)"
        )
    print(json.dumps(make_report(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
