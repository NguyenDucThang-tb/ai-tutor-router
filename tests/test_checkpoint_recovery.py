from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.errors import ErrorCategory
from app.gateway import CostInfo, ExecutionAttempt
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from evaluation.evaluator import evaluate_answer
from experiments.config import BenchmarkConfig
from experiments.records import ExperimentRecord
from experiments.runner import _manifest_key, create_run_manifest, run_benchmark, run_id_for
from experiments.store import RunStore


def sample(index: int) -> DatasetRecord:
    return DatasetRecord(
        sample_id=f"synthetic-{index}",
        dataset="synthetic-test",
        dataset_revision="fixture-v1",
        original_split="train",
        assigned_split="train",
        question=f"What is {index} plus one?",
        reference_answer=str(index + 1),
        task_type="word_problem",
        source_id=str(index),
        source_hash=f"{index:064x}",
    )


def _record() -> ExperimentRecord:
    return ExperimentRecord(
        run_id="old-run",
        sample_id="synthetic-1",
        split="validation",
        model_tier="small",
        requested_model_identifier="small",
        executed_model_identifier="small",
        prompt_version="math-direct-v1",
        prompt_sha256="b" * 64,
        provider="mock",
        execution_mode="mock",
        execution_status="success",
        model_output="FINAL ANSWER: 2",
        evaluation=evaluate_answer("FINAL ANSWER: 2", "2"),
        evaluation_complete=True,
        task_type="word_problem",
    )


def initialized_store(path: Path):
    config = BenchmarkConfig(
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        max_retries=0,
    )
    row = sample(1).model_copy(update={"assigned_split": "validation"})
    manifest = DatasetManifest(
        dataset="synthetic-test",
        source_organization="test",
        source_url="local://fixture",
        revision="fixture-v1",
        license="CC0-1.0",
        access_note="test",
        downloaded_at_utc=datetime.now(UTC),
        raw_files={},
        transformation_history=["fixture"],
        seed=42,
        train_fraction=0.9,
        near_duplicate_threshold=0.9,
        assigned_split_counts={"validation": 1},
        dataset_content_sha256="a" * 64,
        processed_file_sha256={},
    )
    run_manifest = create_run_manifest(config, [row], manifest, "validation")
    store = RunStore(path)
    store.ensure_run(run_manifest, _manifest_key(run_manifest))
    store.ensure_tasks(run_manifest.run_id, [(row.sample_id, "small")])
    return store, run_manifest, row


def claim(store, run_id, sample_id, *, owner, now, lease=30, **kwargs):
    return store.claim(
        run_id,
        sample_id,
        "small",
        owner_id=owner,
        lease_seconds=lease,
        now=now,
        **kwargs,
    )


def unknown_record(run_id, sample_id):
    attempt = ExecutionAttempt(
        request_id="request-old",
        attempt_number=1,
        model_identifier="small",
        provider="mock",
        fallback=False,
        latency_ms=0,
        status="outcome_unknown",
        cost=CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
    )
    return _record().model_copy(
        update={
            "run_id": run_id,
            "sample_id": sample_id,
            "execution_status": "outcome_unknown",
            "model_output": None,
            "evaluation": None,
            "evaluation_complete": False,
            "scientific_eligible": False,
            "attempts": [attempt],
            "cost": CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
            "provenance": {
                "dataset_revision": "fixture-v1",
                "source_hash": sample(1).source_hash,
                "possible_additional_cost": True,
                "provider_outcome_unknown": True,
            },
        }
    )


def mark_invoked(store, claim_value, record, now):
    store.mark_invoked(
        claim_value,
        record,
        {"request_id": "request-old", "provider": "mock", "model": "small"},
        now=now,
    )


def test_concurrent_runners_get_one_claim(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")

    def get_claim(owner):
        return claim(store, manifest.run_id, row.sample_id, owner=owner, now=100)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(get_claim, ("runner-a", "runner-b")))
    assert sum(item is not None for item in outcomes) == 1


def test_live_lease_prevents_second_runner_claim(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="a", now=100, lease=10)
    second = claim(store, manifest.run_id, row.sample_id, owner="b", now=105, lease=10)
    assert first is not None
    assert second is None


def test_crash_before_invocation_reclaims_as_known_unstarted(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="crashed", now=100, lease=10)
    recovered = claim(store, manifest.run_id, row.sample_id, owner="recovery", now=111, lease=10)
    assert first is not None and recovered is not None
    assert recovered.prior_uncertain_record is None
    assert not recovered.legacy_unknown_invocation
    assert recovered.fencing_token > first.fencing_token
    with pytest.raises(RuntimeError, match="claim was lost"):
        mark_invoked(store, first, unknown_record(manifest.run_id, row.sample_id), 112)


def test_crash_after_invocation_is_uncertain_and_paid_resume_is_opt_in(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="crashed", now=100, lease=10)
    mark_invoked(store, first, unknown_record(manifest.run_id, row.sample_id), 100)

    denied = claim(
        store,
        manifest.run_id,
        row.sample_id,
        owner="default-resume",
        now=111,
        lease=10,
        provider_is_paid=True,
    )
    assert denied is None
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "uncertain"
    saved = store.list_records(manifest.run_id)
    assert len(saved) == 1 and saved[0].execution_status == "outcome_unknown"
    assert saved[0].cost.unknown_attempt_count == 1
    assert saved[0].provenance.get("possible_additional_cost") is True


def test_explicit_uncertain_recovery_preserves_attempt_and_fences_stale_writer(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="a", now=100, lease=10)
    uncertain = unknown_record(manifest.run_id, row.sample_id)
    mark_invoked(store, first, uncertain, 100)
    recovered = claim(
        store,
        manifest.run_id,
        row.sample_id,
        owner="approved-recovery",
        now=111,
        lease=10,
        provider_is_paid=True,
        allow_uncertain_reissue=True,
        uncertain_reissue_approval_id="approval-123",
    )
    assert recovered is not None
    assert recovered.prior_uncertain_record is not None
    assert recovered.prior_uncertain_record.attempts[0].status == "outcome_unknown"
    assert recovered.fencing_token > first.fencing_token
    with pytest.raises(ValueError, match="approval reference"):
        claim(
            store,
            manifest.run_id,
            row.sample_id,
            owner="invalid-recovery",
            now=112,
            provider_is_paid=True,
            allow_uncertain_reissue=True,
        )
    with pytest.raises(RuntimeError, match="stale execution claim"):
        store.save(
            _record().model_copy(update={"run_id": manifest.run_id, "sample_id": row.sample_id}),
            first,
        )


def test_crash_before_approved_reissue_invocation_restores_uncertain_checkpoint(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="a", now=100, lease=10)
    prior = unknown_record(manifest.run_id, row.sample_id)
    mark_invoked(store, first, prior, 100)
    recovered = claim(
        store,
        manifest.run_id,
        row.sample_id,
        owner="approved-recovery",
        now=111,
        lease=10,
        provider_is_paid=True,
        allow_uncertain_reissue=True,
        uncertain_reissue_approval_id="approval-123",
    )
    assert recovered is not None
    store.release_unstarted(recovered, now=112)
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "uncertain"
    restored = store.list_records(manifest.run_id)
    assert len(restored) == 1 and restored[0].execution_status == "outcome_unknown"


def test_expired_approved_retry_claim_restores_failed_record_and_request_budget(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    now = time.time()
    first = claim(store, manifest.run_id, row.sample_id, owner="initial", now=now, lease=30)
    assert first is not None
    provisional = unknown_record(manifest.run_id, row.sample_id)
    mark_invoked(store, first, provisional, now)
    failed_attempt = replace(
        provisional.attempts[0],
        status="failed",
        error_category=ErrorCategory.AUTHENTICATION_FAILURE,
    )
    failed = provisional.model_copy(
        update={"execution_status": "provider_failure", "attempts": [failed_attempt]}
    )
    store.save(failed, first)

    abandoned_recovery = claim(
        store,
        manifest.run_id,
        row.sample_id,
        owner="recovery-crashed-before-invoke",
        now=now + 1,
        lease=1,
        provider_is_paid=True,
        retry_failed=True,
        allow_paid_retries=True,
        paid_retry_approval_id="approved-retry",
    )
    assert abandoned_recovery is not None
    resumed = claim(
        store,
        manifest.run_id,
        row.sample_id,
        owner="recovery-resumed",
        now=now + 3,
        lease=10,
        provider_is_paid=True,
        retry_failed=True,
        allow_paid_retries=True,
        paid_retry_approval_id="approved-retry",
    )
    assert resumed is not None
    assert resumed.prior_status == "provider_failure"
    assert resumed.prior_uncertain_record is not None
    assert resumed.prior_uncertain_record.attempts[0].request_id == failed_attempt.request_id
    assert (
        sum(
            reservation["reserved_requests"]
            for reservation in store.request_reservations(manifest.run_id)
        )
        == 2
    )
    store.release_unstarted(resumed, now=now + 4)
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "provider_failure"
    assert (
        store.list_records(manifest.run_id)[0].attempts[0].request_id == failed_attempt.request_id
    )
    assert (
        sum(
            reservation["reserved_requests"]
            for reservation in store.request_reservations(manifest.run_id)
        )
        == 1
    )
    assert [item["state"] for item in store.recovery_audit(manifest.run_id)] == [
        "lease_expired_before_invoke",
        "not_invoked",
    ]


def test_expired_live_claim_can_be_fenced_and_recovered(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    first = claim(store, manifest.run_id, row.sample_id, owner="a", now=100, lease=1)
    assert first
    assert not store.renew(first, now=102, lease_seconds=5)
    second = claim(store, manifest.run_id, row.sample_id, owner="b", now=102, lease=5)
    assert second and second.fencing_token == first.fencing_token + 1


def test_live_lease_renewal_protects_provider_invocation(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    now = 100
    first = claim(store, manifest.run_id, row.sample_id, owner="active", now=now, lease=10)
    assert first
    assert store.renew(first, now=105, lease_seconds=10)
    assert claim(store, manifest.run_id, row.sample_id, owner="second", now=112, lease=10) is None


def test_expired_owner_cannot_write_even_before_recovery_claim(tmp_path):
    store, manifest, row = initialized_store(tmp_path / "run.sqlite3")
    now = time.time()
    first = claim(store, manifest.run_id, row.sample_id, owner="expired", now=now, lease=0.01)
    assert first
    provisional = unknown_record(manifest.run_id, row.sample_id)
    mark_invoked(store, first, provisional, now)
    time.sleep(0.02)
    with pytest.raises(RuntimeError, match="stale execution claim"):
        store.save(
            _record().model_copy(update={"run_id": manifest.run_id, "sample_id": row.sample_id}),
            first,
        )


def test_real_reissue_approval_config_is_explicit():
    base = {
        "mode": "real",
        "provider": "litellm",
        "small_model_identifier": "small",
        "strong_model_identifier": "strong",
        "allow_real_provider": True,
        "experiment_approval_id": "exp-1",
        "allow_uncertain_reissue": True,
    }
    with pytest.raises(ValueError, match="uncertain_reissue_approval_id"):
        BenchmarkConfig.model_validate(base)
    accepted = BenchmarkConfig.model_validate(
        {**base, "uncertain_reissue_approval_id": "reissue-approval-1"}
    )
    assert accepted.allow_uncertain_reissue
    recovered = accepted.model_copy(
        update={
            "allow_uncertain_reissue": False,
            "uncertain_reissue_approval_id": None,
        }
    )
    row = sample(1)
    assert run_id_for(accepted, [row]) == run_id_for(recovered, [row])


def test_paid_runner_rejects_injected_gateway_even_with_retry_approval(tmp_path):
    import asyncio

    class CountingGateway:
        provider = "litellm"
        calls = 0

        async def complete(self, request):
            self.calls += 1
            raise AssertionError("paid provider adapter must not be invoked")

    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="unverified-small",
        strong_model_identifier="unverified-strong",
        allow_real_provider=True,
        experiment_approval_id="offline-approval-reference",
        allow_paid_retries=True,
        paid_retry_approval_id="offline-recovery-reference",
        retry_failed=True,
    )
    gateway = CountingGateway()
    with pytest.raises(ValueError, match="paid invocation blocked"):
        asyncio.run(
            run_benchmark(config, [], None, tmp_path / "blocked-run", "validation", gateway)
        )
    assert gateway.calls == 0
    assert not (tmp_path / "blocked-run").exists()


def test_paid_retry_cap_does_not_override_external_spend_gate(tmp_path):
    import asyncio

    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
        allow_real_provider=True,
        experiment_approval_id="offline-approval-reference",
        allow_paid_retries=True,
        paid_retry_approval_id="offline-recovery-reference",
        retry_failed=True,
        max_samples=1,
        max_requests=4,
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
    )
    with pytest.raises(ValueError, match="external_run_spend_bound_unverified"):
        asyncio.run(run_benchmark(config, [], None, tmp_path / "blocked-run", "validation"))
    assert not (tmp_path / "blocked-run").exists()


def test_runner_cancellation_after_provider_start_persists_uncertain(tmp_path):
    import asyncio

    class HangingGateway:
        provider = "mock"

        def __init__(self):
            self.started = asyncio.Event()

        async def complete(self, request):
            self.started.set()
            await asyncio.Event().wait()

    async def exercise():
        config = BenchmarkConfig(
            small_model_identifier="small",
            strong_model_identifier="strong",
            max_samples=1,
            max_requests=2,
            max_retries=0,
            concurrency=1,
            execution_lease_seconds=3,
        )
        row = sample(1).model_copy(update={"assigned_split": "validation"})
        source = DatasetManifest(
            dataset="synthetic-test",
            source_organization="test",
            source_url="local://fixture",
            revision="fixture-v1",
            license="CC0-1.0",
            access_note="test",
            downloaded_at_utc=datetime.now(UTC),
            raw_files={},
            transformation_history=["fixture"],
            seed=42,
            train_fraction=0.9,
            near_duplicate_threshold=0.9,
            assigned_split_counts={"train": 0, "validation": 1, "test": 0},
            dataset_content_sha256=sha256_records([row]),
            processed_file_sha256={},
        )
        gateway = HangingGateway()
        directory = tmp_path / "cancelled-run"
        task = asyncio.create_task(
            run_benchmark(config, [row], source, directory, "validation", gateway)
        )
        await asyncio.wait_for(gateway.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        store = RunStore(directory / "run.sqlite3")
        run_id = run_id_for(config, [row])
        assert store.execution_state(run_id, row.sample_id, "small") == "uncertain"
        saved = store.list_records(run_id)
        assert len(saved) == 1
        assert saved[0].execution_status == "outcome_unknown"
        assert saved[0].provenance["possible_additional_cost"] is True

    asyncio.run(exercise())
