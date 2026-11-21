from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.errors import GatewayError
from app.features import extract_features
from app.gateway import CostInfo, Gateway, GatewayRequest, LiteLLMGateway, MockGateway
from app.routing import RouterConfig, build_router
from app.settings import Settings
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from experiments.config import BenchmarkConfig
from experiments.prompts import PROMPT_SHA256, PROMPT_TEMPLATE, PROMPT_VERSION
from experiments.records import ExecutionMetrics, ExperimentRecord, RunManifest
from experiments.store import RunStore


def run_id_for(config: BenchmarkConfig, records: Sequence[DatasetRecord]) -> str:
    material = {
        "config": config.model_dump(mode="json"),
        "dataset_content_sha256": sha256_records(records),
        "sample_ids": [record.sample_id for record in records],
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return f"run-{hashlib.sha256(encoded).hexdigest()[:20]}"


def create_run_manifest(
    config: BenchmarkConfig,
    records: Sequence[DatasetRecord],
    source_manifest: DatasetManifest,
    assigned_split: str,
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
    return RunManifest(
        run_id=run_id_for(config, records),
        created_at_utc=datetime.now(UTC).isoformat(),
        git_commit_sha=_git_commit_sha(),
        execution_mode=config.mode,
        dataset=source_manifest.dataset,
        dataset_revision=source_manifest.revision,
        dataset_content_sha256=selected_hash,
        assigned_split=assigned_split,
        sample_count=len(records),
        sample_ids_sha256=ids_hash,
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
        pricing_source=config.pricing_source,
        pricing_effective_date=config.pricing_effective_date,
        experiment_approval_id=config.experiment_approval_id,
        software_versions=packages,
        scientific_eligible=config.mode == "real",
    )


async def run_benchmark(
    config: BenchmarkConfig,
    records: Sequence[DatasetRecord],
    source_manifest: DatasetManifest,
    run_directory: Path,
    assigned_split: str,
    gateway: Gateway | None = None,
) -> tuple[RunManifest, list[ExperimentRecord]]:
    if config.mode == "replay":
        raise ValueError("replay mode uses replay_run(), not run_benchmark()")
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
    selected = selected[: min(config.max_samples, request_sample_limit)]
    if not selected:
        raise ValueError(f"no records found in assigned split '{assigned_split}'")
    manifest = create_run_manifest(config, selected, source_manifest, assigned_split)
    manifest_key = _manifest_key(manifest)
    run_directory.mkdir(parents=True, exist_ok=True)
    store = RunStore(run_directory / "run.sqlite3")
    store.ensure_run(manifest, manifest_key)
    atomic_write(run_directory / "run_manifest.json", manifest.model_dump_json(indent=2).encode())

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
    for sample in selected:
        request_id = _request_id(manifest.run_id, sample.sample_id, "pair")
        feature_started = time.perf_counter()
        features = extract_features(sample.question, sample.task_type)
        feature_ms = (time.perf_counter() - feature_started) * 1000
        decision = router.decide(features, request_id)
        contexts[sample.sample_id] = (features, feature_ms, decision)

    jobs = ((sample, tier) for sample in selected for tier in ("small", "strong"))
    semaphore = asyncio.Semaphore(config.concurrency)
    stop = object()

    async def worker() -> None:
        while True:
            job = next(jobs, stop)
            if job is stop:
                return
            sample, tier = job
            async with semaphore:
                await _execute_one(
                    config,
                    sample,
                    tier,
                    contexts[sample.sample_id],
                    manifest,
                    model_gateway,
                    store,
                )

    async with asyncio.TaskGroup() as group:
        for _ in range(config.concurrency):
            group.create_task(worker())

    result_records = store.list_records(manifest.run_id)
    atomic_write(
        run_directory / "records.jsonl",
        b"".join((record.model_dump_json() + "\n").encode() for record in result_records),
    )
    statuses = store.statuses(manifest.run_id)
    atomic_write(
        run_directory / "run_summary.json",
        json.dumps(
            {
                "run_id": manifest.run_id,
                "execution_mode": config.mode,
                "scientific_eligible": manifest.scientific_eligible,
                "expected_requests": len(selected) * 2,
                "statuses": statuses,
                "completed_records": len(result_records),
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
    manifest: RunManifest,
    gateway: Gateway,
    store: RunStore,
) -> None:
    token = store.claim(
        manifest.run_id,
        sample.sample_id,
        tier,
        retry_interrupted=config.retry_interrupted,
        retry_failed=config.retry_failed,
        rerun_completed=config.rerun_completed,
    )
    if token is None:
        return
    _, feature_ms, decision = context
    model_identifier = (
        config.small_model_identifier if tier == "small" else config.strong_model_identifier
    )
    fallback_identifier = (
        config.small_fallback_identifier if tier == "small" else config.strong_fallback_identifier
    )
    request_id = _request_id(manifest.run_id, sample.sample_id, tier)
    started = time.perf_counter()
    try:
        result = await gateway.complete(
            GatewayRequest(
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
        )
        evaluator_started = time.perf_counter()
        evaluation = evaluate_answer(result.text, sample.reference_answer)
        evaluator_ms = (time.perf_counter() - evaluator_started) * 1000
        record = ExperimentRecord(
            **_record_common(
                manifest,
                sample,
                tier,
                model_identifier,
                gateway.provider,
                config.mode,
                feature_ms,
                decision,
            ),
            executed_model_identifier=result.model_identifier,
            execution_status="success",
            model_output=result.text,
            evaluation=evaluation,
            metrics=ExecutionMetrics(
                feature_extraction_ms=feature_ms,
                router_decision_ms=decision.router_latency_ms,
                gateway_latency_ms=result.latency_ms,
                evaluator_latency_ms=evaluator_ms,
                end_to_end_latency_ms=(time.perf_counter() - started) * 1000,
                retry_delay_ms=result.retry_delay_ms,
                prompt_tokens=result.usage.prompt_tokens if result.usage else None,
                completion_tokens=result.usage.completion_tokens if result.usage else None,
                total_tokens=result.usage.total_tokens if result.usage else None,
                calculated_cost_usd=result.cost.calculated_usd,
                known_calculated_cost_usd=result.cost.known_calculated_usd,
                unknown_cost_attempts=result.cost.unknown_attempt_count,
            ),
            usage=result.usage,
            cost=result.cost,
            final_attempt_cost=result.final_attempt_cost,
            attempts=list(result.attempts),
            provenance={
                "dataset_revision": sample.dataset_revision,
                "source_hash": sample.source_hash,
                "commit_sha": manifest.git_commit_sha,
                "sample_is_synthetic": sample.dataset.startswith("synthetic"),
            },
        )
        store.save(record, token)
    except GatewayError as exc:
        record = ExperimentRecord(
            **_record_common(
                manifest,
                sample,
                tier,
                model_identifier,
                gateway.provider,
                config.mode,
                feature_ms,
                decision,
            ),
            execution_status="provider_failure",
            metrics=ExecutionMetrics(
                feature_extraction_ms=feature_ms,
                router_decision_ms=decision.router_latency_ms,
                end_to_end_latency_ms=(time.perf_counter() - started) * 1000,
                retry_delay_ms=exc.retry_delay_ms,
                calculated_cost_usd=exc.cost.calculated_usd if exc.cost else None,
                known_calculated_cost_usd=exc.cost.known_calculated_usd if exc.cost else None,
                unknown_cost_attempts=exc.cost.unknown_attempt_count
                if exc.cost
                else len(exc.attempts),
            ),
            cost=exc.cost or CostInfo(),
            final_attempt_cost=exc.final_attempt_cost,
            attempts=list(exc.attempts),
            provenance={
                "dataset_revision": sample.dataset_revision,
                "source_hash": sample.source_hash,
                "commit_sha": manifest.git_commit_sha,
                "error_category": exc.category.value,
            },
        )
        store.save(record, token)
    except asyncio.CancelledError:
        interrupted = ExperimentRecord(
            **_record_common(
                manifest,
                sample,
                tier,
                model_identifier,
                gateway.provider,
                config.mode,
                feature_ms,
                decision,
            ),
            execution_status="interrupted",
            metrics=ExecutionMetrics(end_to_end_latency_ms=(time.perf_counter() - started) * 1000),
            provenance={"dataset_revision": sample.dataset_revision, "interrupted": True},
        )
        store.save(interrupted, token)
        raise


def _record_common(
    manifest: RunManifest,
    sample: DatasetRecord,
    tier: str,
    model_identifier: str,
    provider: str,
    mode: str,
    feature_ms: float,
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
        "provider": provider,
        "execution_mode": mode,
        "task_type": sample.task_type,
        "difficulty": sample.difficulty,
    }


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
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request_id(run_id: str, sample_id: str, tier: str) -> str:
    return hashlib.sha256(f"{run_id}\0{sample_id}\0{tier}".encode()).hexdigest()[:32]


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
