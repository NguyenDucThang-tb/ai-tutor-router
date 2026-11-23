"""Offline preflight for real-provider experiments; never makes provider requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path

from datasets.schemas import DatasetManifest
from datasets.validation import load_processed_dataset
from experiments.authorization import verify_approval_artifact
from experiments.collection_plan import validate_collection_plan
from experiments.config import BenchmarkConfig, load_config
from experiments.model_catalog import resolve_model
from experiments.paid_execution_gate import assess_paid_execution
from experiments.pricing import estimate_usd
from experiments.prompts import PROMPT_SHA256
from experiments.provider_capabilities import resolve_capability
from experiments.runner import run_id_for, scientific_eligibility_reasons
from experiments.store import RunStore


def _check(checks: list[dict], status: str, name: str, detail: str) -> None:
    checks.append({"status": status, "check": name, "detail": detail})


def _usd_micro_units(value: float | None) -> int | None:
    if value is None:
        return None
    micros = Decimal(str(value)) * Decimal(1_000_000)
    if micros != micros.to_integral_value():
        return None
    return int(micros)


def make_report(
    config: BenchmarkConfig,
    *,
    dataset_dir: Path,
    manifest_path: Path,
    run_dir: Path | None = None,
    environment: dict[str, str] | None = None,
    collection_plan: dict | Path | None = None,
    split: str = "validation",
) -> dict:
    env = os.environ if environment is None else environment
    checks: list[dict] = []
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text("utf-8"))
        rows = load_processed_dataset(dataset_dir, manifest)
        if split not in {"train", "validation"}:
            raise ValueError("preflight only permits train or validation splits")
        if isinstance(collection_plan, Path):
            plan_data = json.loads(collection_plan.read_text("utf-8"))
        else:
            plan_data = collection_plan
        if plan_data is None:
            selected = sorted(
                (row for row in rows if row.assigned_split == split),
                key=lambda row: row.sample_id,
            )[: config.max_samples]
            selection_policy = "legacy_sorted_prefix_v1"
        else:
            selected = validate_collection_plan(
                plan_data,
                rows,
                manifest,
                expected_split=split,
                expected_count=config.max_samples,
            )
            selection_policy = "versioned_collection_plan_v1"
        _check(
            checks, "PASS", "dataset_integrity", f"verified {len(rows)} rows and manifest hashes"
        )
        _check(
            checks,
            "PASS",
            "collection_plan" if plan_data is not None else "legacy_selection",
            (
                f"validated plan {plan_data['plan_sha256']} for {len(selected)} exact sample IDs"
                if plan_data is not None
                else "no plan supplied; legacy sorted-prefix selection is explicit"
            ),
        )
        provenance_reasons = scientific_eligibility_reasons(config, rows, manifest)
        if provenance_reasons:
            _check(checks, "FAIL", "scientific_dataset_provenance", ";".join(provenance_reasons))
        else:
            _check(
                checks,
                "PASS",
                "scientific_dataset_provenance",
                "matches pinned non-synthetic GSM8K source and preparation policy",
            )
        _check(
            checks,
            "PASS",
            "split_policy",
            f"sample selection is {split}-only; official test is not selected",
        )
        if not selected:
            _check(checks, "FAIL", "dataset_sample", f"{split} split has no selected rows")
        selected_split = split
    except (OSError, ValueError, TypeError, KeyError) as exc:
        manifest, selected = None, []
        selection_policy, plan_data = "invalid", None
        selected_split = None
        _check(checks, "FAIL", "dataset_integrity", str(exc))

    if config.mode != "real":
        _check(checks, "FAIL", "mode", "preflight requires mode=real")
    else:
        _check(checks, "PASS", "mode", "real mode requested; this command performs no model calls")
    small = resolve_model(config.small_model_identifier)
    strong = resolve_model(config.strong_model_identifier)
    for tier, model in (("small", small), ("strong", strong)):
        if model is None:
            _check(
                checks, "BLOCKED", f"{tier}_model", "exact model ID has no verified catalog price"
            )
        else:
            _check(checks, "PASS", f"{tier}_model", model.model_id)
        capability = resolve_capability(
            config.small_model_identifier if tier == "small" else config.strong_model_identifier
        )
        if capability is None:
            _check(checks, "BLOCKED", f"{tier}_capability", "provider capability is not verified")
        else:
            _check(
                checks,
                "BLOCKED" if not capability.supports_provable_run_bound else "PASS",
                f"{tier}_capability",
                f"input counting={capability.input_token_method}; output cap="
                f"{capability.output_token_bound}; account access={capability.account_access}; "
                f"hard run bound={capability.supports_provable_run_bound}; "
                f"billing control={capability.billing_control}",
            )
    if small and strong and small.provider != strong.provider:
        _check(
            checks, "WARNING", "provider_pair", "models use different providers and pricing systems"
        )
    elif small and strong:
        _check(checks, "PASS", "provider_pair", f"same provider: {small.provider}")

    missing = [] if env.get("TUTOR_LITELLM_API_KEY") else ["TUTOR_LITELLM_API_KEY"]
    _check(
        checks,
        "WARNING" if missing else "PASS",
        "credentials",
        "missing: " + ", ".join(sorted(missing))
        if missing
        else "credential variable names present; values are not reported",
    )

    approval_valid = False
    approval_detail = "dataset selection unavailable; approval could not be validated"
    if manifest is not None and selected:
        approval_valid, approval_detail = verify_approval_artifact(
            config.paid_approval_artifact,
            config=config,
            records=list(selected),
            manifest=manifest,
            split=selected_split,
            collection_plan=plan_data,
            signing_key=env.get("TUTOR_PAID_APPROVAL_HMAC_KEY", ""),
        )
    _check(
        checks,
        "PASS" if approval_valid else "FAIL",
        "signed_frozen_config_approval",
        approval_detail,
    )

    paid_gate = assess_paid_execution(config, approval_verified=approval_valid)
    _check(
        checks,
        paid_gate.status,
        "paid_execution_gate",
        ",".join(paid_gate.reason_codes)
        if paid_gate.reason_codes
        else "all admission controls passed",
    )
    if config.max_run_budget_usd is None:
        _check(checks, "FAIL", "run_budget", "max_run_budget_usd must be explicit")
    else:
        _check(checks, "PASS", "run_budget", f"configured USD {config.max_run_budget_usd:.6f}")
    if config.per_request_budget_usd is None:
        _check(checks, "FAIL", "per_request_budget", "per_request_budget_usd must be explicit")
    else:
        _check(
            checks,
            "PASS",
            "per_request_budget",
            f"configured USD {config.per_request_budget_usd:.6f}",
        )
        if _usd_micro_units(config.per_request_budget_usd) is None:
            _check(
                checks,
                "FAIL",
                "per_request_budget_precision",
                "budget must be an integer number of micro-USD",
            )
    if (
        config.max_run_budget_usd is not None
        and _usd_micro_units(config.max_run_budget_usd) is None
    ):
        _check(
            checks,
            "FAIL",
            "run_budget_precision",
            "budget must be an integer number of micro-USD",
        )

    retries = config.max_retries + (1 if config.fallback_policy == "operational" else 0)
    request_attempts = len(selected) * 2 * (1 + retries)
    _check(
        checks,
        "PASS",
        "attempt_envelope",
        f"at most {request_attempts} attempts including configured retry/fallback envelope",
    )
    if request_attempts > config.max_requests:
        _check(
            checks,
            "FAIL",
            "request_cap",
            f"configured envelope {request_attempts} exceeds max_requests={config.max_requests}",
        )
    if config.mode == "real" and config.max_retries and not config.allow_paid_retries:
        _check(
            checks,
            "FAIL",
            "retry_approval",
            "paid retry configuration lacks explicit retry approval",
        )
    if (
        config.mode == "real"
        and config.fallback_policy == "operational"
        and not config.allow_paid_retries
    ):
        _check(
            checks,
            "FAIL",
            "fallback_approval",
            "paid fallback configuration lacks explicit retry approval",
        )
    _check(checks, "PASS", "frozen_prompt", f"{config.temperature=}; prompt sha256={PROMPT_SHA256}")
    _check(checks, "PASS", "evaluator", "numeric-symbolic-v2")
    _check(
        checks,
        "PASS",
        "recovery",
        f"uncertain reissue={config.allow_uncertain_reissue}; completed rerun={config.rerun_completed}",
    )

    projections: dict[str, str | None] = {}
    if small and strong and selected:
        projections = _project(selected, config, small.model_id, strong.model_id)
        _check(
            checks,
            "WARNING",
            "cost_projection",
            "heuristic word-count expectation only; byte-count upper proxy is not token-bound",
        )
        if config.max_run_budget_usd is not None:
            conservative = Decimal(projections["conservative_byte_count_scenario"])
            if conservative > Decimal(str(config.max_run_budget_usd)):
                _check(
                    checks,
                    "WARNING",
                    "projected_budget_comparison",
                    "scenario projection exceeds configured budget; projections remain non-binding",
                )
    _check(
        checks,
        "BLOCKED",
        "hard_spend_bound",
        "no verified provider-compatible tokenizer plus provider-enforced hard spend ceiling; paid invocation is refused",
    )

    checkpoint = "not_checked"
    budget_cap_micros = _usd_micro_units(config.max_run_budget_usd)
    ledger = {
        "run_id": None,
        "local_exposure_micros_usd": 0,
        "uncertain_exposure_micros_usd": 0,
        "observed_usage_micros_usd": 0,
        "request_reservations": 0,
        "execution_status_counts": {},
        "run_cap_micros_usd": budget_cap_micros,
        "remaining_local_budget_micros_usd": budget_cap_micros,
        "scope_note": "local SQLite ledger only; external/shared-account usage is not included",
    }
    if run_dir:
        checkpoint_path = run_dir / "run.sqlite3"
        checkpoint = "exists" if checkpoint_path.is_file() else "not_found"
        if checkpoint_path.is_file() and manifest is not None and selected:
            run_id = run_id_for(
                config,
                selected,
                plan_data.get("plan_sha256") if plan_data is not None else None,
            )
            ledger["run_id"] = run_id
            store = RunStore(checkpoint_path)
            try:
                store.stored_manifest(run_id)
            except KeyError:
                checkpoint = "exists_without_matching_run"
            else:
                reservations = store.budget_reservations(run_id)
                ledger["local_exposure_micros_usd"] = sum(
                    item["exposure_micros"] for item in reservations
                )
                ledger["uncertain_exposure_micros_usd"] = sum(
                    item["exposure_micros"]
                    for item in reservations
                    if item["status"] == "uncertain"
                )
                ledger["observed_usage_micros_usd"] = sum(
                    item["observed_micros"] or 0 for item in reservations
                )
                ledger["request_reservations"] = sum(
                    item["reserved_requests"] for item in store.request_reservations(run_id)
                )
                ledger["execution_status_counts"] = store.statuses(run_id)
                if budget_cap_micros is not None:
                    ledger["remaining_local_budget_micros_usd"] = (
                        budget_cap_micros - ledger["local_exposure_micros_usd"]
                    )
        _check(
            checks,
            "WARNING",
            "checkpoint",
            f"{checkpoint}; manifest compatibility must be checked by the runner",
        )
    selection_failed = any(
        row["status"] == "FAIL"
        and row["check"] in {"dataset_integrity", "collection_plan", "split_policy"}
        for row in checks
    )
    state = (
        "FAIL"
        if selection_failed
        else (
            "BLOCKED"
            if any(row["status"] == "BLOCKED" for row in checks)
            else ("FAIL" if any(row["status"] == "FAIL" for row in checks) else "PASS")
        )
    )
    return {
        "schema_version": "phase3a-preflight-v1",
        "status": state,
        "provider_calls_made": 0,
        "dataset": manifest.model_dump(mode="json") if manifest else None,
        "split": selected_split,
        "requested_split": split,
        "selected_sample_count": len(selected),
        "selection_policy": selection_policy,
        "selected_sample_ids": [row.sample_id for row in selected],
        "selected_sample_ids_sha256": hashlib.sha256(
            "\n".join(row.sample_id for row in selected).encode()
        ).hexdigest(),
        "collection_plan_sha256": plan_data.get("plan_sha256") if plan_data else None,
        "max_requests": config.max_requests,
        "maximum_attempt_envelope": request_attempts,
        "budget_usd": str(Decimal(str(config.max_run_budget_usd)))
        if config.max_run_budget_usd is not None
        else None,
        "cost_projection_usd": projections,
        "pricing_catalog": {
            "small": _price_metadata(small),
            "strong": _price_metadata(strong),
        },
        "pricing_is_actual_billing": False,
        "checkpoint": checkpoint,
        "budget_ledger": ledger,
        "checks": checks,
    }


def _project(rows, config, small_id: str, strong_id: str) -> dict[str, str | None]:
    # Existing token_count is a deterministic word count, not a tokenizer. Show it
    # only as a scenario estimate and do not claim it bounds provider billing.
    from app.features import extract_features
    from experiments.prompts import PROMPT_TEMPLATE

    expected_input_counts = [
        len(PROMPT_TEMPLATE.split()) + extract_features(row.question).token_count for row in rows
    ]
    byte_scenario_input_counts = [
        len((PROMPT_TEMPLATE + row.question).encode("utf-8")) for row in rows
    ]
    expected = Decimal(0)
    conservative = Decimal(0)
    for expected_count, byte_count in zip(
        expected_input_counts, byte_scenario_input_counts, strict=True
    ):
        for model_id in (small_id, strong_id):
            expected += estimate_usd(
                model_id, expected_count, max(1, config.max_tokens // 2)
            ) or Decimal(0)
            conservative += estimate_usd(model_id, byte_count, config.max_tokens) or Decimal(0)
    attempt_multiplier = (
        1 + config.max_retries + (1 if config.fallback_policy == "operational" else 0)
    )
    return {
        "expected_word_count_scenario": str(expected * attempt_multiplier),
        "conservative_byte_count_scenario": str(conservative * attempt_multiplier),
        "hard_upper_bound": None,
    }


def _price_metadata(model) -> dict | None:
    if model is None:
        return None
    return {
        "provider": model.provider,
        "model_id": model.model_id,
        "currency": "USD",
        "unit": "per 1,000,000 uncached text tokens",
        "input_price": str(model.input_usd_per_million),
        "output_price": str(model.output_usd_per_million),
        "pricing_source_url": model.pricing_url,
        "identity_source_url": model.identity_url,
        "retrieved_date": model.retrieved.isoformat(),
        "effective_date": model.pricing_effective_date,
        "pricing_assumptions": model.notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/gsm8k"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/gsm8k.json"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--collection-plan", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("preflight_report.json"))
    parser.add_argument(
        "--dry-run", action="store_true", help="accepted for explicit offline intent"
    )
    args = parser.parse_args(argv)
    report = make_report(
        load_config(args.config),
        dataset_dir=args.dataset_dir,
        manifest_path=args.manifest,
        run_dir=args.run_dir,
        collection_plan=args.collection_plan,
        split=args.split,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": report["status"], "report": str(args.output), "provider_calls_made": 0}
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
