from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from experiments import provider_capabilities
from experiments.config import BenchmarkConfig
from experiments.paid_execution_gate import assess_paid_execution
from experiments.provider_capabilities import price_provenance_is_current
from experiments.records import ExperimentRecord
from experiments.runner import _manifest_key, create_run_manifest, run_id_for
from experiments.store import RunStore
from experiments.token_accounting import CountKind, UnsupportedTokenCounter


def _setup(path):
    row = DatasetRecord(
        sample_id="fixture-1",
        dataset="synthetic-test",
        dataset_revision="fixture-v1",
        original_split="train",
        assigned_split="validation",
        question="1+1?",
        reference_answer="2",
        task_type="word_problem",
        source_id="1",
        source_hash="a" * 64,
    )
    source = DatasetManifest(
        dataset="synthetic-test",
        source_organization="test",
        source_url="local://test",
        revision="fixture-v1",
        license="CC0",
        access_note="fixture",
        downloaded_at_utc=datetime.now(UTC),
        raw_files={},
        transformation_history=["fixture"],
        seed=42,
        train_fraction=0.9,
        near_duplicate_threshold=0.9,
        assigned_split_counts={"validation": 1},
        dataset_content_sha256=sha256_records([row]),
        processed_file_sha256={},
    )
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        per_request_budget_usd=0.001,
        max_run_budget_usd=0.001,
    )
    manifest = create_run_manifest(config, [row], source, "validation")
    store = RunStore(path)
    store.ensure_run(manifest, _manifest_key(manifest))
    store.ensure_tasks(manifest.run_id, [(row.sample_id, "small"), (row.sample_id, "strong")])
    return store, manifest, row, config


def _claim(store, manifest, row, tier, *, now=100, lease=60):
    claim = store.claim(
        manifest.run_id,
        row.sample_id,
        tier,
        owner_id=f"owner-{tier}",
        lease_seconds=lease,
        now=now,
        provider_is_paid=True,
        request_attempt_limit=1,
    )
    assert claim is not None
    return claim


def test_micro_usd_budget_reservations_are_atomic_under_concurrency(tmp_path):
    store, manifest, row, _ = _setup(tmp_path / "runs.sqlite3")
    claims = [_claim(store, manifest, row, tier) for tier in ("small", "strong")]

    def reserve(claim):
        try:
            store.reserve_budget(
                claim,
                700,
                pricing_provenance={"test_price_version": "fixture-v1"},
                now=101,
            )
            return "reserved"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, claims))
    assert outcomes.count("reserved") == 1
    reservations = store.budget_reservations(manifest.run_id)
    assert len(reservations) == 1
    assert reservations[0]["exposure_micros"] == 700
    restarted = RunStore(tmp_path / "runs.sqlite3")
    assert restarted.budget_reservations(manifest.run_id) == reservations


def test_budget_reservation_rejects_request_cap_and_stale_claim(tmp_path):
    store, manifest, row, _ = _setup(tmp_path / "runs.sqlite3")
    claim = _claim(store, manifest, row, "small", now=100, lease=60)
    with pytest.raises(ValueError, match="per-request budget"):
        store.reserve_budget(claim, 1001, pricing_provenance={"version": "x"}, now=101)
    stale = _claim(store, manifest, row, "strong", now=100, lease=1)
    with pytest.raises(RuntimeError, match="stale execution claim"):
        store.reserve_budget(stale, 100, pricing_provenance={"version": "x"}, now=102)


def test_expired_unstarted_claim_releases_only_its_held_exposure(tmp_path):
    store, manifest, row, _ = _setup(tmp_path / "runs.sqlite3")
    first = _claim(store, manifest, row, "small", now=100, lease=1)
    store.reserve_budget(first, 600, pricing_provenance={"version": "v1"}, now=100.5)
    next_claim = store.claim(
        manifest.run_id,
        row.sample_id,
        "small",
        owner_id="new-owner",
        lease_seconds=60,
        now=102,
        provider_is_paid=True,
    )
    assert next_claim is not None
    reservation = store.budget_reservations(manifest.run_id)[0]
    assert reservation["claim_token"] == first.token
    assert reservation["status"] == "released"
    assert reservation["exposure_micros"] == 0


def test_invoked_budget_is_retained_as_uncertain_after_crash(tmp_path):
    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    claim = _claim(store, manifest, row, "small")
    store.reserve_budget(
        claim,
        800,
        pricing_provenance={"provider": "fake", "price_sha256": "f" * 64},
        now=101,
    )
    record = ExperimentRecord(
        run_id=manifest.run_id,
        sample_id=row.sample_id,
        split="validation",
        model_tier="small",
        requested_model_identifier=config.small_model_identifier,
        prompt_version=manifest.prompt_version,
        prompt_sha256=manifest.prompt_sha256,
        provider="litellm",
        execution_mode="real",
        execution_status="not_started",
        task_type=row.task_type,
    )
    store.mark_invoked(claim, record, {"model": "small"}, now=102)
    store.mark_uncertain(claim, now=103)
    row = RunStore(tmp_path / "runs.sqlite3").budget_reservations(manifest.run_id)[0]
    assert row["status"] == "uncertain"
    assert row["exposure_micros"] == 800
    assert row["observed_micros"] is None


def test_paid_invocation_requires_monetary_reservation(tmp_path):
    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    claim = _claim(store, manifest, row, "small")
    record = ExperimentRecord(
        run_id=manifest.run_id,
        sample_id=row.sample_id,
        split="validation",
        model_tier="small",
        requested_model_identifier=config.small_model_identifier,
        prompt_version=manifest.prompt_version,
        prompt_sha256=manifest.prompt_sha256,
        provider="litellm",
        execution_mode="real",
        execution_status="not_started",
        task_type=row.task_type,
    )
    with pytest.raises(RuntimeError, match="monetary budget reservation is required"):
        store.mark_invoked(claim, record, {"model": "small"}, now=101)


def test_unknown_local_token_count_is_not_a_zero_count():
    result = UnsupportedTokenCounter().count_messages("provider/model", [])
    assert result.kind == CountKind.UNKNOWN
    assert result.input_tokens is None
    assert not result.supports_admission


def test_paid_gate_blocks_current_candidate_without_account_and_hard_bound():
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
        allow_real_provider=True,
        experiment_approval_id="offline-test-only",
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
        max_samples=1,
        max_requests=2,
    )
    decision = assess_paid_execution(config)
    assert decision.status == "BLOCKED"
    assert "external_run_spend_bound_unverified" in decision.reason_codes
    assert "provider_account_access_unverified" in decision.reason_codes
    assert "exact_request_token_count_unavailable" in decision.reason_codes


def test_injected_gateway_never_authorizes_paid_execution():
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="small",
        strong_model_identifier="strong",
        allow_real_provider=True,
        experiment_approval_id="offline-test-only",
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
    )
    decision = assess_paid_execution(config, gateway=object())
    assert decision.status == "BLOCKED"
    assert "injected_gateway_not_trusted_for_paid_execution" in decision.reason_codes


def test_unknown_model_and_invalid_provider_fail_admission():
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="provider/unlisted-model",
        strong_model_identifier="also-unlisted",
        allow_real_provider=True,
        experiment_approval_id="offline-test-only",
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
    )
    decision = assess_paid_execution(config)
    assert decision.status == "BLOCKED"
    assert "provider_model_capability_unverified" in decision.reason_codes

    invalid_provider = config.model_copy(update={"provider": "unknown-provider"})
    assert "paid_mode_provider_mismatch" in assess_paid_execution(invalid_provider).reason_codes


def test_recovery_approval_keeps_identity_but_run_budget_is_immutable():
    base = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
        max_samples=1,
        max_requests=2,
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
    )
    recovery = base.model_copy(
        update={
            "retry_failed": True,
            "allow_paid_retries": True,
            "paid_retry_approval_id": "offline-recovery-approval",
        }
    )
    row = DatasetRecord(
        sample_id="same-sample",
        dataset="synthetic-test",
        dataset_revision="fixture-v1",
        original_split="train",
        assigned_split="validation",
        question="1+1?",
        reference_answer="2",
        task_type="word_problem",
        source_id="1",
        source_hash="a" * 64,
    )
    assert run_id_for(base, [row]) == run_id_for(recovery, [row])
    increased = base.model_copy(update={"max_run_budget_usd": 2})
    assert run_id_for(base, [row]) != run_id_for(increased, [row])


def test_expired_price_provenance_and_missing_output_bound_fail_closed(monkeypatch):
    capability = provider_capabilities.resolve_capability("gemini/gemini-3.8-flash")
    assert capability is not None
    assert not price_provenance_is_current(
        replace(capability, verified_on=date(2020, 1, 1)), today=date(2026, 11, 22)
    )
    monkeypatch.setitem(
        provider_capabilities.CAPABILITIES,
        capability.model_id,
        replace(capability, output_token_bound=None),
    )
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
        allow_real_provider=True,
        experiment_approval_id="offline-test-only",
        per_request_budget_usd=0.1,
        max_run_budget_usd=1,
    )
    decision = assess_paid_execution(config)
    assert "output_token_bound_missing" in decision.reason_codes
