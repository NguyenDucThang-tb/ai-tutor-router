from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.errors import ErrorCategory, GatewayError
from app.features import extract_features
from app.gateway import (
    CostInfo,
    ExecutionAttempt,
    Gateway,
    GatewayRequest,
    LiteLLMGateway,
    MockGateway,
    UsageInfo,
    aggregate_attempt_costs,
)
from app.routing import RouterConfig, build_router
from app.settings import Settings
from datasets.gsm8k import LICENSE as GSM8K_LICENSE
from datasets.gsm8k import SOURCE_ORGANIZATION as GSM8K_ORGANIZATION
from datasets.gsm8k import SOURCE_REPOSITORY as GSM8K_REPOSITORY
from datasets.gsm8k import SOURCE_REVISION as GSM8K_REVISION
from datasets.gsm8k import VERIFIED_PREPARED_CONTENT_SHA256 as GSM8K_CONTENT_SHA256
from datasets.gsm8k import VERIFIED_RAW_FILE_SHA256 as GSM8K_RAW_FILE_SHA256
from datasets.gsm8k import VERIFIED_RAW_FILE_SIZES as GSM8K_RAW_FILE_SIZES
from datasets.gsm8k import VERIFIED_SPLIT_COUNTS as GSM8K_SPLIT_COUNTS
from datasets.gsm8k import raw_file_url as gsm8k_raw_file_url
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from experiments.authorization import verify_approval_artifact
from experiments.collection_plan import validate_collection_plan
from experiments.config import BenchmarkConfig
from experiments.exposure import (
    ExposureCalculator,
    ProviderExposureCalculator,
    UnboundedExposureError,
    attempt_model_envelope,
)
from experiments.paid_execution_gate import assess_paid_execution
from experiments.prompts import PROMPT_SHA256, PROMPT_TEMPLATE, PROMPT_VERSION
from experiments.records import ExecutionMetrics, ExperimentRecord, RunManifest
from experiments.store import ExecutionClaim, RunStore


class LeaseLostError(RuntimeError):
    pass


def run_id_for(
    config: BenchmarkConfig,
    records: Sequence[DatasetRecord],
    collection_plan_sha256: str | None = None,
) -> str:
    identity_config = config.model_dump(mode="json")
    for control in (
        "retry_interrupted",
        "retry_failed",
        "rerun_completed",
        "allow_uncertain_reissue",
        "uncertain_reissue_approval_id",
        "allow_paid_retries",
        "paid_retry_approval_id",
        "paid_approval_artifact",
    ):
        identity_config.pop(control, None)
    material = {
        "config": identity_config,
        "dataset_content_sha256": sha256_records(records),
        "sample_ids": [record.sample_id for record in records],
    }
    if collection_plan_sha256 is not None:
        material["collection_plan_sha256"] = collection_plan_sha256
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return f"run-{hashlib.sha256(encoded).hexdigest()[:20]}"


def scientific_eligibility_reasons(
    config: BenchmarkConfig, records: Sequence[DatasetRecord], source_manifest: DatasetManifest
) -> list[str]:
    """Check protocol provenance; this is not an accuracy or coverage assertion."""
    reasons = []
    if config.mode != "real":
        reasons.append("execution_mode_is_not_real")
    if config.provider != "litellm":
        reasons.append("provider_is_not_litellm")
    if (
        source_manifest.dataset != "gsm8k"
        or source_manifest.source_url != GSM8K_REPOSITORY
        or source_manifest.source_organization != GSM8K_ORGANIZATION
        or source_manifest.revision != GSM8K_REVISION
        or source_manifest.license != GSM8K_LICENSE
    ):
        reasons.append("dataset_not_in_verified_source_registry")
    expected_raw_splits = {"train", "test"}
    if set(source_manifest.raw_files) != expected_raw_splits or any(
        source_manifest.raw_files[name].source_url != gsm8k_raw_file_url(name)
        or source_manifest.raw_files[name].sha256 != GSM8K_RAW_FILE_SHA256[name]
        or source_manifest.raw_files[name].size_bytes != GSM8K_RAW_FILE_SIZES[name]
        for name in expected_raw_splits.intersection(source_manifest.raw_files)
    ):
        reasons.append("dataset_raw_sources_do_not_match_verified_gsm8k")
    if (
        source_manifest.dataset_content_sha256 != GSM8K_CONTENT_SHA256
        or source_manifest.assigned_split_counts != GSM8K_SPLIT_COUNTS
        or source_manifest.seed != 42
        or source_manifest.train_fraction != 0.9
        or source_manifest.near_duplicate_threshold != 0.9
    ):
        reasons.append("dataset_preparation_does_not_match_verified_gsm8k")
    if not source_manifest.raw_files or any(
        not re.fullmatch(r"[0-9a-f]{64}", item.sha256)
        for item in source_manifest.raw_files.values()
    ):
        reasons.append("dataset_source_file_checksums_missing")
    if not source_manifest.source_organization or not source_manifest.license:
        reasons.append("dataset_source_or_license_metadata_missing")
    if any(
        record.dataset != source_manifest.dataset
        or record.dataset_revision != source_manifest.revision
        or len(record.source_hash) != 64
        for record in records
    ):
        reasons.append("dataset_record_provenance_mismatch")
    if config.temperature != 0:
        reasons.append("temperature_not_frozen_at_zero")
    if PROMPT_VERSION != "math-direct-v1" or len(PROMPT_SHA256) != 64:
        reasons.append("benchmark_prompt_not_frozen")
    if not config.small_model_identifier or not config.strong_model_identifier:
        reasons.append("model_identifiers_missing")
    return reasons


def create_run_manifest(
    config: BenchmarkConfig,
    records: Sequence[DatasetRecord],
    source_manifest: DatasetManifest,
    assigned_split: str,
    collection_plan: dict | None = None,
    approval_artifact_sha256: str | None = None,
    approval_config_sha256: str | None = None,
) -> RunManifest:
    selected_hash = sha256_records(records)
    ids_hash = hashlib.sha256(
        "\n".join(record.sample_id for record in records).encode()
    ).hexdigest()
    packages = {}
    for package in ("ai-tutor-router", "litellm", "sympy", "pydantic"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    eligibility_reasons = scientific_eligibility_reasons(config, records, source_manifest)
    return RunManifest(
        run_id=run_id_for(
            config,
            records,
            collection_plan.get("plan_sha256") if collection_plan is not None else None,
        ),
        created_at_utc=datetime.now(UTC).isoformat(),
        git_commit_sha=_git_commit_sha(),
        execution_mode=config.mode,
        dataset=source_manifest.dataset,
        dataset_source_url=source_manifest.source_url,
        dataset_source_organization=source_manifest.source_organization,
        dataset_license=source_manifest.license,
        dataset_revision=source_manifest.revision,
        dataset_source_file_sha256={
            name: item.sha256 for name, item in source_manifest.raw_files.items()
        },
        dataset_content_sha256=selected_hash,
        source_dataset_content_sha256=(
            source_manifest.dataset_content_sha256 if collection_plan is not None else None
        ),
        sample_source_hashes={record.sample_id: record.source_hash for record in records},
        assigned_split=assigned_split,
        sample_count=len(records),
        sample_ids_sha256=ids_hash,
        collection_plan_schema_version=(
            collection_plan.get("schema_version") if collection_plan is not None else None
        ),
        collection_plan_sha256=(
            collection_plan.get("plan_sha256") if collection_plan is not None else None
        ),
        collection_plan_sample_ids=(
            list(collection_plan["sample_ids"]) if collection_plan is not None else []
        ),
        collection_plan_sample_ids_sha256=(
            collection_plan.get("sample_ids_sha256") if collection_plan is not None else None
        ),
        collection_plan_split=(collection_plan.get("split") if collection_plan else None),
        selection_policy=(
            "versioned_collection_plan_v1"
            if collection_plan is not None
            else "legacy_sorted_prefix_v1"
        ),
        small_model_identifier=config.small_model_identifier,
        strong_model_identifier=config.strong_model_identifier,
        provider=config.provider,
        provider_region=config.provider_region,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=PROMPT_SHA256,
        evaluator_version=EVALUATOR_VERSION,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        random_seed=config.seed,
        concurrency=config.concurrency,
        timeout_seconds=config.timeout_seconds,
        max_requests=config.max_requests,
        max_retries=config.max_retries,
        retry_delay_seconds=config.retry_delay_seconds,
        fallback_policy=config.fallback_policy,
        small_fallback_identifier=config.small_fallback_identifier,
        strong_fallback_identifier=config.strong_fallback_identifier,
        budget_limit_usd=config.per_request_budget_usd,
        max_run_budget_usd=config.max_run_budget_usd,
        pricing_source=config.pricing_source,
        pricing_effective_date=config.pricing_effective_date,
        experiment_approval_id=config.experiment_approval_id,
        approval_artifact_sha256=approval_artifact_sha256,
        approval_config_sha256=approval_config_sha256,
        software_versions=packages,
        scientific_protocol_eligible=not eligibility_reasons,
        scientific_eligibility_reasons=eligibility_reasons,
    )


async def run_benchmark(
    config: BenchmarkConfig,
    records: Sequence[DatasetRecord],
    source_manifest: DatasetManifest,
    run_directory: Path,
    assigned_split: str,
    gateway: Gateway | None = None,
    collection_plan: dict | None = None,
) -> tuple[RunManifest, list[ExperimentRecord]]:
    if config.mode == "replay":
        raise ValueError("replay mode uses replay_run(), not run_benchmark()")
    if config.mode == "real" and (
        not config.allow_real_provider or not config.experiment_approval_id
    ):
        raise ValueError(
            "real provider execution requires allow_real_provider and an approval reference"
        )
    if config.mode == "real":
        static_gate = assess_paid_execution(config, gateway=gateway)
        static_blockers = tuple(
            reason
            for reason in static_gate.reason_codes
            if reason != "signed_frozen_config_approval_missing"
        )
        if static_blockers:
            raise ValueError(
                "paid invocation blocked by centralized gate: " + ",".join(static_blockers)
            )
    if sha256_records(records) != source_manifest.dataset_content_sha256:
        raise ValueError("benchmark records do not match the dataset manifest content hash")
    split_counts = {split: 0 for split in ("train", "validation", "test")}
    for record in records:
        split_counts[record.assigned_split] += 1
    if split_counts != source_manifest.assigned_split_counts:
        raise ValueError("benchmark records do not match the dataset manifest split counts")
    if collection_plan is not None:
        selected = validate_collection_plan(
            collection_plan,
            records,
            source_manifest,
            expected_split=assigned_split,
            expected_count=config.max_samples,
        )
    else:
        selected = sorted(
            (record for record in records if record.assigned_split == assigned_split),
            key=lambda record: record.sample_id,
        )
    worst_case_attempts = config.max_retries + 1
    if config.fallback_policy == "operational":
        worst_case_attempts += 1
    if config.mode != "real":
        worst_case_attempts = 1
    request_sample_limit = config.max_requests // (2 * worst_case_attempts)
    if collection_plan is not None:
        if request_sample_limit < len(selected):
            raise ValueError("request cap cannot accommodate the complete collection plan")
    else:
        selected = selected[: min(config.max_samples, request_sample_limit)]
    if not selected:
        raise ValueError(f"no records found in assigned split '{assigned_split}'")
    approval_artifact_sha256 = None
    approval_config_sha256 = None
    if config.mode == "real":
        approval_valid, _approval_detail = verify_approval_artifact(
            config.paid_approval_artifact,
            config=config,
            records=list(selected),
            manifest=source_manifest,
            split=assigned_split,
            collection_plan=collection_plan,
        )
        decision = assess_paid_execution(config, gateway=gateway, approval_verified=approval_valid)
        if decision.status != "PASS":
            raise ValueError(
                "paid invocation blocked by centralized gate: " + ",".join(decision.reason_codes)
            )
        artifact_bytes = Path(config.paid_approval_artifact).read_bytes()
        approval_artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        approval_document = json.loads(artifact_bytes)
        approval_config_sha256 = approval_document["immutable_config_sha256"]
    manifest = create_run_manifest(
        config,
        selected,
        source_manifest,
        assigned_split,
        collection_plan,
        approval_artifact_sha256,
        approval_config_sha256,
    )
    manifest_key = _manifest_key(manifest)
    run_directory.mkdir(parents=True, exist_ok=True)
    store = RunStore(run_directory / "run.sqlite3")
    store.ensure_run(manifest, manifest_key)
    tasks = [(sample.sample_id, tier) for sample in selected for tier in ("small", "strong")]
    store.ensure_tasks(manifest.run_id, tasks)
    stored_manifest = store.stored_manifest(manifest.run_id)
    atomic_write(
        run_directory / "run_manifest.json",
        json.dumps(stored_manifest, indent=2, sort_keys=True).encode(),
    )

    model_gateway = gateway or _create_gateway(config)
    router = build_router(
        config.router_strategy,
        RouterConfig(
            small_model_id=config.small_model_identifier,
            strong_model_id=config.strong_model_identifier,
            random_seed=str(config.seed),
        ),
    )
    contexts = {}
    enqueued_times = {}
    for sample in selected:
        pipeline_started = time.perf_counter()
        request_id = _request_id(manifest.run_id, sample.sample_id, "pair")
        feature_started = time.perf_counter()
        features = extract_features(sample.question, sample.task_type)
        feature_ms = (time.perf_counter() - feature_started) * 1000
        decision = router.decide(features, request_id)
        contexts[sample.sample_id] = (pipeline_started, feature_ms, decision)
        enqueued_times[sample.sample_id] = time.perf_counter()

    jobs = [
        (sample, tier, enqueued_times[sample.sample_id])
        for sample in selected
        for tier in ("small", "strong")
    ]
    semaphore = asyncio.Semaphore(config.concurrency)
    owner_id = str(uuid.uuid4())
    stop = object()
    job_iterator = iter(jobs)

    async def worker() -> None:
        while True:
            job = next(job_iterator, stop)
            if job is stop:
                return
            sample, tier, enqueued_at = job
            async with semaphore:
                await _execute_one(
                    config,
                    sample,
                    tier,
                    contexts[sample.sample_id],
                    enqueued_at,
                    manifest,
                    model_gateway,
                    store,
                    owner_id,
                )

    async with asyncio.TaskGroup() as group:
        for _ in range(config.concurrency):
            group.create_task(worker())

    result_records = store.list_records(manifest.run_id)
    existing = {(row.sample_id, row.model_tier) for row in result_records}
    for sample in selected:
        for tier in ("small", "strong"):
            if (sample.sample_id, tier) in existing:
                continue
            state = store.execution_state(manifest.run_id, sample.sample_id, tier)
            if state in {"pending", "claimed"}:
                _, feature_ms, decision = contexts[sample.sample_id]
                result_records.append(
                    _not_started_record(
                        manifest, sample, tier, config, model_gateway, feature_ms, decision
                    )
                )
            elif state == "uncertain":
                _, feature_ms, decision = contexts[sample.sample_id]
                result_records.append(
                    _legacy_uncertain_record(
                        manifest, sample, tier, config, model_gateway, feature_ms, decision
                    )
                )
    result_records.sort(key=lambda row: (row.sample_id, row.model_tier))
    atomic_write(
        run_directory / "records.jsonl",
        b"".join((record.model_dump_json() + "\n").encode() for record in result_records),
    )
    recovery_audit = store.recovery_audit(manifest.run_id)
    atomic_write(
        run_directory / "recovery_audit.jsonl",
        b"".join((json.dumps(item, sort_keys=True) + "\n").encode() for item in recovery_audit),
    )
    statuses = store.statuses(manifest.run_id)
    atomic_write(
        run_directory / "run_summary.json",
        json.dumps(
            {
                "run_id": manifest.run_id,
                "execution_mode": config.mode,
                "scientific_protocol_eligible": manifest.scientific_protocol_eligible,
                "scientific_eligibility_reasons": manifest.scientific_eligibility_reasons,
                "expected_requests": len(selected) * 2,
                "statuses": statuses,
                "records_written": len(result_records),
                "mock_outputs_are_scientific": False,
            },
            indent=2,
            sort_keys=True,
        ).encode(),
    )
    return manifest, result_records


async def _execute_one(
    config: BenchmarkConfig,
    sample: DatasetRecord,
    tier: str,
    context,
    enqueued_at: float,
    manifest: RunManifest,
    gateway: Gateway,
    store: RunStore,
    owner_id: str,
    *,
    _test_exposure_calculator: ExposureCalculator | None = None,
) -> None:
    """Execute one claimed task.

    ``_test_exposure_calculator`` is a private offline seam used only by the
    deterministic integration harness. The public runner never accepts it and
    still applies ``assess_paid_execution`` before constructing work.
    """
    request_attempt_limit = config.max_retries + 1
    if config.fallback_policy == "operational" and config.mode == "real":
        request_attempt_limit += 1
    if config.mode != "real":
        request_attempt_limit = 1
    claim = store.claim(
        manifest.run_id,
        sample.sample_id,
        tier,
        owner_id=owner_id,
        lease_seconds=config.execution_lease_seconds,
        now=time.time(),
        retry_failed=config.retry_failed,
        rerun_completed=config.rerun_completed,
        retry_interrupted=config.retry_interrupted,
        allow_uncertain_reissue=(config.mode != "real" or config.allow_uncertain_reissue),
        uncertain_reissue_approval_id=config.uncertain_reissue_approval_id,
        allow_paid_retries=config.allow_paid_retries,
        paid_retry_approval_id=config.paid_retry_approval_id,
        provider_is_paid=config.mode == "real",
        request_attempt_limit=request_attempt_limit,
        git_commit_sha=manifest.git_commit_sha,
    )
    if claim is None:
        return

    pipeline_started, feature_ms, decision = context
    queue_ms = max(0.0, (time.perf_counter() - enqueued_at) * 1000)
    model_identifier = (
        config.small_model_identifier if tier == "small" else config.strong_model_identifier
    )
    fallback_identifier = (
        config.small_fallback_identifier if tier == "small" else config.strong_fallback_identifier
    )
    request_id = _request_id(manifest.run_id, sample.sample_id, tier, claim.token)
    prior_attempts = (
        list(claim.prior_uncertain_record.attempts) if claim.prior_uncertain_record else []
    )
    if claim.legacy_unknown_invocation:
        prior_attempts.append(
            ExecutionAttempt(
                request_id=f"legacy-unknown-{claim.fencing_token}",
                attempt_number=len(prior_attempts) + 1,
                model_identifier="unknown-legacy-model",
                provider="unknown",
                fallback=False,
                latency_ms=0,
                status="outcome_unknown",
                cost=CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
            )
        )
    attempt_number = len(prior_attempts) + 1
    unknown_attempt = ExecutionAttempt(
        request_id=request_id,
        attempt_number=attempt_number,
        model_identifier=model_identifier,
        provider=gateway.provider,
        fallback=False,
        latency_ms=0.0,
        status="outcome_unknown",
        cost=CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
    )
    provisional = _make_record(
        manifest,
        sample,
        tier,
        model_identifier,
        gateway.provider,
        config.mode,
        feature_ms,
        decision,
        execution_status="outcome_unknown",
        attempts=[*prior_attempts, unknown_attempt],
        cost=aggregate_attempt_costs([*prior_attempts, unknown_attempt]),
        queue_ms=queue_ms,
        pipeline_started=pipeline_started,
        provenance_extra={
            "request_id": request_id,
            "provider_outcome_unknown": True,
            "possible_additional_cost": True,
            "uncertain_reissue_approval_id": config.uncertain_reissue_approval_id,
        },
    )
    invoked = False
    gateway_started: float | None = None
    try:
        gateway_request = GatewayRequest(
            request_id=request_id,
            model_identifier=model_identifier,
            system_prompt=PROMPT_TEMPLATE,
            user_content=sample.question,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_delay_seconds=config.retry_delay_seconds,
            fallback_model_identifier=fallback_identifier,
            budget_limit_usd=config.per_request_budget_usd,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        if config.mode == "real":
            calculator = _test_exposure_calculator or ProviderExposureCalculator()
            exposure = calculator.calculate(
                config,
                gateway_request,
                attempt_model_envelope(config, tier),
            )
            if exposure.amount_micros_usd <= 0:
                raise UnboundedExposureError("exposure bound must be positive")
            if not exposure.provenance:
                raise UnboundedExposureError("exposure bound is missing audit provenance")
            store.reserve_budget(
                claim,
                exposure.amount_micros_usd,
                pricing_provenance=exposure.provenance,
                now=time.time(),
            )
        store.mark_invoked(
            claim,
            provisional,
            {
                "request_id": request_id,
                "provider": gateway.provider,
                "requested_model_identifier": model_identifier,
                "fallback_model_identifier": fallback_identifier,
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "user_content_sha256": hashlib.sha256(sample.question.encode()).hexdigest(),
                "reserved_requests": request_attempt_limit,
                "reserved_exposure_micros_usd": (
                    exposure.amount_micros_usd if config.mode == "real" else None
                ),
                "paid_retry_approval_id": config.paid_retry_approval_id
                if claim.prior_status in {"provider_failure", "success"}
                else None,
                "uncertain_reissue_approval_id": config.uncertain_reissue_approval_id
                if claim.prior_status == "uncertain"
                else None,
            },
            now=time.time(),
        )
        invoked = True
        gateway_started = time.perf_counter()
        result = await _complete_with_lease(
            gateway, gateway_request, store, claim, config.execution_lease_seconds
        )
        evaluator_started = time.perf_counter()
        evaluation = evaluate_answer(result.text, sample.reference_answer)
        evaluator_ms = (time.perf_counter() - evaluator_started) * 1000
        attempts = _renumber_attempts([*prior_attempts, *result.attempts])
        cost = aggregate_attempt_costs(attempts)
        record = _make_record(
            manifest,
            sample,
            tier,
            model_identifier,
            gateway.provider,
            config.mode,
            feature_ms,
            decision,
            execution_status="success",
            terminal_reached=True,
            attempts=attempts,
            cost=cost,
            final_attempt_cost=result.final_attempt_cost,
            queue_ms=queue_ms,
            pipeline_started=pipeline_started,
            model_output=result.text,
            evaluation=evaluation,
            usage=result.usage,
            evaluator_ms=evaluator_ms,
            gateway_elapsed_ms=result.latency_ms,
            retry_delay_ms=result.retry_delay_ms,
            provenance_extra={
                "request_id": request_id,
                "output_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                "provider_outcome_unknown": False,
                "possible_additional_cost": bool(prior_attempts),
                "paid_retry_approval_id": config.paid_retry_approval_id
                if claim.prior_status in {"provider_failure", "success"}
                else None,
                "uncertain_reissue_approval_id": config.uncertain_reissue_approval_id
                if claim.prior_status == "uncertain"
                else None,
            },
        )
        record = record.model_copy(
            update={
                "scientific_eligible": _record_scientifically_eligible(record, manifest, config)
            }
        )
        store.save(record, claim)
    except GatewayError as exc:
        attempts = _renumber_attempts([*prior_attempts, *exc.attempts])
        cost = aggregate_attempt_costs(attempts) if attempts else exc.cost or CostInfo()
        outcome_unknown = (
            config.mode == "real"
            and bool(exc.attempts)
            and exc.category
            not in {
                ErrorCategory.AUTHENTICATION_FAILURE,
                ErrorCategory.UNSUPPORTED_MODEL,
                ErrorCategory.BUDGET_EXCEEDED,
            }
        )
        failed = _make_record(
            manifest,
            sample,
            tier,
            model_identifier,
            gateway.provider,
            config.mode,
            feature_ms,
            decision,
            execution_status="outcome_unknown" if outcome_unknown else "provider_failure",
            terminal_reached=True,
            attempts=attempts,
            cost=cost,
            final_attempt_cost=exc.final_attempt_cost,
            queue_ms=queue_ms,
            pipeline_started=pipeline_started,
            retry_delay_ms=exc.retry_delay_ms,
            gateway_elapsed_ms=(time.perf_counter() - gateway_started) * 1000
            if gateway_started is not None
            else None,
            provenance_extra={
                "request_id": request_id,
                "error_category": exc.category.value,
                "provider_outcome_unknown": outcome_unknown,
                "possible_additional_cost": bool(prior_attempts) or outcome_unknown,
                "paid_retry_approval_id": config.paid_retry_approval_id
                if claim.prior_status in {"provider_failure", "success"}
                else None,
                "uncertain_reissue_approval_id": config.uncertain_reissue_approval_id
                if claim.prior_status == "uncertain"
                else None,
            },
        )
        if outcome_unknown:
            store.save_uncertain(failed, claim)
        else:
            store.save(failed, claim)
    except asyncio.CancelledError:
        if invoked:
            _mark_uncertain_if_owned(store, claim)
        else:
            store.release_unstarted(claim, now=time.time())
        raise
    except Exception:
        if invoked:
            _mark_uncertain_if_owned(store, claim)
        else:
            store.release_unstarted(claim, now=time.time())
        raise


async def _maintain_lease(store: RunStore, claim: ExecutionClaim, lease_seconds: float) -> bool:
    interval = max(0.05, min(30.0, lease_seconds / 3))
    while True:
        await asyncio.sleep(interval)
        renewed = store.renew(claim, now=time.time(), lease_seconds=lease_seconds)
        if not renewed:
            return False


async def _complete_with_lease(
    gateway: Gateway,
    request: GatewayRequest,
    store: RunStore,
    claim: ExecutionClaim,
    lease_seconds: float,
):
    provider_task = asyncio.create_task(gateway.complete(request))
    heartbeat_task = asyncio.create_task(_maintain_lease(store, claim, lease_seconds))
    try:
        done, _ = await asyncio.wait(
            {provider_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat_task in done:
            if not provider_task.done():
                provider_task.cancel()
                await asyncio.gather(provider_task, return_exceptions=True)
            raise LeaseLostError("execution lease expired while provider outcome may be in flight")
        return provider_task.result()
    except asyncio.CancelledError:
        provider_task.cancel()
        await asyncio.gather(provider_task, return_exceptions=True)
        raise
    finally:
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


def _mark_uncertain_if_owned(store: RunStore, claim: ExecutionClaim) -> None:
    try:
        store.mark_uncertain(claim, now=time.time())
    except RuntimeError:
        # A newer fencing token owns the task; never overwrite its checkpoint.
        return


def _make_record(
    manifest: RunManifest,
    sample: DatasetRecord,
    tier: str,
    model_identifier: str,
    provider: str,
    mode: str,
    feature_ms: float,
    decision,
    *,
    execution_status: str,
    terminal_reached: bool = False,
    attempts: list[ExecutionAttempt],
    cost: CostInfo,
    queue_ms: float,
    pipeline_started: float,
    model_output: str | None = None,
    evaluation=None,
    usage: UsageInfo | None = None,
    evaluator_ms: float | None = None,
    gateway_elapsed_ms: float | None = None,
    retry_delay_ms: float = 0,
    final_attempt_cost: CostInfo | None = None,
    provenance_extra: dict | None = None,
) -> ExperimentRecord:
    primary_attempts = [item for item in attempts if not item.fallback]
    fallback_attempts = [item for item in attempts if item.fallback]
    eligible = bool(evaluation is not None and model_output is not None)
    elapsed_ms = (time.perf_counter() - pipeline_started) * 1000
    provenance = {
        "dataset": sample.dataset,
        "dataset_revision": sample.dataset_revision,
        "source_hash": sample.source_hash,
        "commit_sha": manifest.git_commit_sha,
        "sample_is_synthetic": sample.dataset.startswith("synthetic"),
        **_plan_record_provenance(manifest),
        **(provenance_extra or {}),
    }
    return ExperimentRecord(
        **_record_common(manifest, sample, tier, model_identifier, provider, mode, decision),
        executed_model_identifier=attempts[-1].model_identifier if attempts else None,
        execution_status=execution_status,
        model_output=model_output,
        evaluation=evaluation,
        evaluation_complete=eligible,
        scientific_eligible=False,
        metrics=ExecutionMetrics(
            feature_extraction_ms=feature_ms,
            router_decision_ms=decision.router_latency_ms,
            scheduling_queue_ms=queue_ms,
            model_only_latency_ms=sum(item.latency_ms for item in primary_attempts),
            fallback_latency_ms=sum(item.latency_ms for item in fallback_attempts),
            gateway_elapsed_ms=gateway_elapsed_ms,
            evaluator_latency_ms=evaluator_ms,
            pipeline_end_to_end_ms=elapsed_ms if eligible else None,
            pipeline_elapsed_to_terminal_ms=elapsed_ms if terminal_reached else None,
            retry_delay_ms=retry_delay_ms,
            retry_count=max(0, len(primary_attempts) - 1),
            fallback_count=len(fallback_attempts),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            calculated_cost_usd=cost.calculated_usd,
            known_calculated_cost_usd=cost.known_calculated_usd,
            unknown_cost_attempts=cost.unknown_attempt_count,
        ),
        cost=cost,
        final_attempt_cost=final_attempt_cost,
        attempts=attempts,
        usage=usage,
        provenance=provenance,
    )


def _not_started_record(manifest, sample, tier, config, gateway, feature_ms, decision):
    identifier = (
        config.small_model_identifier if tier == "small" else config.strong_model_identifier
    )
    return ExperimentRecord(
        **_record_common(
            manifest, sample, tier, identifier, gateway.provider, config.mode, decision
        ),
        execution_status="not_started",
        provenance={
            "dataset": sample.dataset,
            "dataset_revision": sample.dataset_revision,
            "source_hash": sample.source_hash,
            "commit_sha": manifest.git_commit_sha,
            "checkpoint_state": "known_unstarted",
            **_plan_record_provenance(manifest),
        },
        metrics=ExecutionMetrics(
            feature_extraction_ms=feature_ms,
            router_decision_ms=decision.router_latency_ms,
        ),
    )


def _plan_record_provenance(manifest: RunManifest) -> dict:
    if manifest.collection_plan_sha256 is None:
        return {"selection_policy": manifest.selection_policy}
    return {
        "collection_plan_sha256": manifest.collection_plan_sha256,
        "collection_plan_schema_version": manifest.collection_plan_schema_version,
        "collection_plan_sample_ids_sha256": manifest.collection_plan_sample_ids_sha256,
        "collection_plan_split": manifest.collection_plan_split,
        "selection_policy": manifest.selection_policy,
    }


def _legacy_uncertain_record(manifest, sample, tier, config, gateway, feature_ms, decision):
    record = _not_started_record(manifest, sample, tier, config, gateway, feature_ms, decision)
    attempt = ExecutionAttempt(
        request_id=f"legacy-unknown-{manifest.run_id}-{sample.sample_id}-{tier}",
        attempt_number=1,
        model_identifier="unknown-legacy-model",
        provider="unknown",
        fallback=False,
        latency_ms=0,
        status="outcome_unknown",
        cost=CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
    )
    return record.model_copy(
        update={
            "execution_status": "outcome_unknown",
            "attempts": [attempt],
            "cost": aggregate_attempt_costs([attempt]),
            "provenance": {
                **record.provenance,
                "checkpoint_state": "legacy_uncertain",
                "provider_outcome_unknown": True,
                "possible_additional_cost": True,
            },
            "metrics": ExecutionMetrics(unknown_cost_attempts=1),
        }
    )


def _record_common(
    manifest: RunManifest,
    sample: DatasetRecord,
    tier: str,
    model_identifier: str,
    provider: str,
    mode: str,
    decision,
) -> dict:
    return {
        "run_id": manifest.run_id,
        "sample_id": sample.sample_id,
        "split": sample.assigned_split,
        "model_tier": tier,
        "requested_model_identifier": model_identifier,
        "selected_tier": tier,
        "router_selected_tier": decision.selected_tier,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "evaluator_version": EVALUATOR_VERSION,
        "provider": provider,
        "fallback_policy": manifest.fallback_policy,
        "execution_mode": mode,
        "task_type": sample.task_type,
        "difficulty": sample.difficulty,
    }


def _record_scientifically_eligible(
    record: ExperimentRecord, manifest: RunManifest, config: BenchmarkConfig
) -> bool:
    if not manifest.scientific_protocol_eligible or not record.evaluation_complete:
        return False
    if manifest.execution_mode != "real" or manifest.provider != "litellm":
        return False
    if manifest.temperature != 0:
        return False
    if record.execution_status != "success" or record.execution_mode != "real":
        return False
    if record.provider != "litellm" or not record.attempts:
        return False
    expected_model = (
        config.small_model_identifier
        if record.model_tier == "small"
        else config.strong_model_identifier
    )
    if record.requested_model_identifier != expected_model:
        return False
    if record.fallback_policy != manifest.fallback_policy:
        return False
    if record.prompt_version != PROMPT_VERSION or record.prompt_sha256 != PROMPT_SHA256:
        return False
    if record.evaluator_version != EVALUATOR_VERSION:
        return False
    if record.model_output is None or record.evaluation is None:
        return False
    if (
        record.provenance.get("dataset") != manifest.dataset
        or record.provenance.get("dataset_revision") != manifest.dataset_revision
    ):
        return False
    source_hash = record.provenance.get("source_hash")
    if (
        not isinstance(source_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
        or manifest.sample_source_hashes.get(record.sample_id) != source_hash
    ):
        return False
    if (
        record.provenance.get("output_sha256")
        != hashlib.sha256((record.model_output or "").encode()).hexdigest()
    ):
        return False
    configured = {record.requested_model_identifier}
    fallback = (
        config.small_fallback_identifier
        if record.model_tier == "small"
        else config.strong_fallback_identifier
    )
    if fallback:
        configured.add(fallback)
    successful = [attempt for attempt in record.attempts if attempt.status == "success"]
    return bool(
        record.executed_model_identifier in configured
        and successful
        and successful[-1].provider == "litellm"
        and successful[-1].model_identifier == record.executed_model_identifier
        and successful[-1].request_id == record.provenance.get("request_id")
    )


def _renumber_attempts(attempts: list[ExecutionAttempt]) -> list[ExecutionAttempt]:
    return [replace(item, attempt_number=number) for number, item in enumerate(attempts, 1)]


def _create_gateway(config: BenchmarkConfig) -> Gateway:
    if config.mode == "mock":
        return MockGateway()
    settings = Settings()
    if settings.provider != "litellm" or not settings.litellm_api_key:
        raise ValueError("real mode requires TUTOR_PROVIDER=litellm and TUTOR_LITELLM_API_KEY")
    if settings.per_request_budget_usd is not None and config.per_request_budget_usd is None:
        raise ValueError("TUTOR_PER_REQUEST_BUDGET_USD is active; configure a benchmark budget")
    return LiteLLMGateway(
        settings.litellm_api_key.get_secret_value(),
        settings.litellm_api_base,
    )


def _manifest_key(manifest: RunManifest) -> str:
    material = manifest.model_dump(mode="json")
    material.pop("created_at_utc", None)
    material.pop("git_commit_sha", None)
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request_id(run_id: str, sample_id: str, tier: str, invocation_token: str = "") -> str:
    material = f"{run_id}\0{sample_id}\0{tier}\0{invocation_token}"
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def _git_commit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=2
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)
