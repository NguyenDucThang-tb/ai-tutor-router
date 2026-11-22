from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import ROUND_CEILING, Decimal

import pytest

from app.errors import ErrorCategory, GatewayError
from app.gateway import CostInfo, ExecutionAttempt, GatewayResult, aggregate_attempt_costs
from app.routing import RouteDecision
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
from experiments import model_catalog, provider_capabilities
from experiments.config import BenchmarkConfig
from experiments.exposure import (
    ProviderExposureCalculator,
    UnboundedExposureError,
    VerifiedExposure,
    attempt_model_envelope,
)
from experiments.paid_execution_gate import assess_paid_execution
from experiments.provider_capabilities import price_provenance_is_current
from experiments.records import ExperimentRecord
from experiments.runner import _execute_one, _manifest_key, create_run_manifest, run_id_for
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
    store.reserve_budget(next_claim, 800, pricing_provenance={"version": "v2"}, now=103)
    store.release_unstarted(first, now=104)
    current = store.budget_reservations(manifest.run_id)[1]
    assert current["claim_token"] == next_claim.token
    assert current["status"] == "held"
    assert current["exposure_micros"] == 800


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


def test_real_mode_runner_reserves_before_isolated_fake_provider_invocation(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    seen = {}

    class TrustedOfflineFakeProvider:
        provider = "litellm"

        async def complete(self, request):
            reservations = store.budget_reservations(manifest.run_id)
            assert len(reservations) == 1
            assert reservations[0]["status"] == "uncertain"
            assert reservations[0]["reserved_micros"] == 700
            assert reservations[0]["pricing_provenance"]["offline_fixture_bound"] is True
            seen["request"] = request
            attempt = ExecutionAttempt(
                request_id=request.request_id,
                attempt_number=1,
                model_identifier=request.model_identifier,
                provider=self.provider,
                fallback=False,
                latency_ms=1,
                status="success",
                cost=CostInfo(status="calculated", calculated_usd=0.0002),
            )
            cost = aggregate_attempt_costs((attempt,))
            return GatewayResult(
                request_id=request.request_id,
                text="2",
                model_identifier=request.model_identifier,
                provider=self.provider,
                usage=None,
                cost=cost,
                latency_ms=1,
                attempts=(attempt,),
                final_attempt_cost=attempt.cost,
            )

    class OfflineFixtureBound:
        def calculate(self, config, request, model_identifiers):
            assert request.system_prompt and request.user_content == row.question
            assert model_identifiers == (config.small_model_identifier,)
            return VerifiedExposure(
                700,
                {
                    "offline_fixture_bound": True,
                    "attempt_model_identifiers": list(model_identifiers),
                },
            )

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )
    asyncio.run(
        _execute_one(
            config,
            row,
            "small",
            (time.perf_counter(), 0.0, decision),
            time.perf_counter(),
            manifest,
            TrustedOfflineFakeProvider(),
            store,
            "offline-owner",
            _test_exposure_calculator=OfflineFixtureBound(),
        )
    )
    assert seen["request"].request_id
    reservation = store.budget_reservations(manifest.run_id)[0]
    assert reservation["status"] == "observed"
    assert reservation["exposure_micros"] == 700
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "success"


def test_real_mode_default_exposure_calculator_fails_closed_before_provider(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    calls = []

    class NeverCallProvider:
        provider = "litellm"

        async def complete(self, request):
            calls.append(request)
            raise AssertionError("unbounded request must not reach provider")

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )
    with pytest.raises(UnboundedExposureError, match="unverified model identity"):
        asyncio.run(
            _execute_one(
                config,
                row,
                "small",
                (time.perf_counter(), 0.0, decision),
                time.perf_counter(),
                manifest,
                NeverCallProvider(),
                store,
                "offline-owner",
            )
        )
    assert calls == []
    assert store.budget_reservations(manifest.run_id) == []
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "pending"


def test_insufficient_request_budget_prevents_provider_invocation(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    calls = []

    class NeverCallProvider:
        provider = "litellm"

        async def complete(self, request):
            calls.append(request)
            raise AssertionError("insufficient budget must not reach provider")

    class OverCapBound:
        def calculate(self, config, request, model_identifiers):
            return VerifiedExposure(1001, {"verified_fixture_bound": True})

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )
    with pytest.raises(ValueError, match="per-request budget"):
        asyncio.run(
            _execute_one(
                config,
                row,
                "small",
                (time.perf_counter(), 0.0, decision),
                time.perf_counter(),
                manifest,
                NeverCallProvider(),
                store,
                "offline-owner",
                _test_exposure_calculator=OverCapBound(),
            )
        )
    assert calls == []
    assert store.budget_reservations(manifest.run_id) == []
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "pending"


def test_cancellation_before_intent_releases_claim_and_no_reservation(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")

    class CancelBeforeIntent:
        def calculate(self, config, request, model_identifiers):
            raise asyncio.CancelledError

    class NeverCallProvider:
        provider = "litellm"

        async def complete(self, request):
            raise AssertionError("provider invocation must not start")

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _execute_one(
                config,
                row,
                "small",
                (time.perf_counter(), 0.0, decision),
                time.perf_counter(),
                manifest,
                NeverCallProvider(),
                store,
                "offline-owner",
                _test_exposure_calculator=CancelBeforeIntent(),
            )
        )
    assert store.budget_reservations(manifest.run_id) == []
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "pending"


def test_cancellation_after_intent_keeps_uncertain_exposure_across_restart(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    started = None

    class HangingFakeProvider:
        provider = "litellm"

        async def complete(self, request):
            started.set()
            await asyncio.Event().wait()

    class OfflineFixtureBound:
        def calculate(self, config, request, model_identifiers):
            return VerifiedExposure(700, {"offline_fixture_bound": True})

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )

    async def exercise():
        nonlocal started
        started = asyncio.Event()
        task = asyncio.create_task(
            _execute_one(
                config,
                row,
                "small",
                (time.perf_counter(), 0.0, decision),
                time.perf_counter(),
                manifest,
                HangingFakeProvider(),
                store,
                "offline-owner",
                _test_exposure_calculator=OfflineFixtureBound(),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    restarted = RunStore(tmp_path / "runs.sqlite3")
    reservation = restarted.budget_reservations(manifest.run_id)[0]
    assert reservation["status"] == "uncertain"
    assert reservation["exposure_micros"] == 700
    assert restarted.execution_state(manifest.run_id, row.sample_id, "small") == "uncertain"


def test_known_failed_attempt_keeps_reserved_exposure_and_audit(tmp_path):
    import asyncio
    import time

    store, manifest, row, config = _setup(tmp_path / "runs.sqlite3")
    attempt = ExecutionAttempt(
        request_id="offline-request",
        attempt_number=1,
        model_identifier=config.small_model_identifier,
        provider="litellm",
        fallback=False,
        latency_ms=1,
        status="failed",
        error_category=ErrorCategory.AUTHENTICATION_FAILURE,
        cost=CostInfo(status="calculated", calculated_usd=0.0003),
    )

    class FailedFakeProvider:
        provider = "litellm"

        async def complete(self, request):
            raise GatewayError(
                ErrorCategory.AUTHENTICATION_FAILURE,
                "fixture rejection",
                [attempt],
                aggregate_attempt_costs([attempt]),
                attempt.cost,
            )

    class OfflineFixtureBound:
        def calculate(self, config, request, model_identifiers):
            return VerifiedExposure(
                700,
                {
                    "offline_fixture_bound": True,
                    "attempt_model_identifiers": list(model_identifiers),
                },
            )

    decision = RouteDecision(
        router_strategy="always_small",
        selected_tier="small",
        selected_model_id=config.small_model_identifier,
        reason_code="fixture",
        router_latency_ms=0,
    )
    asyncio.run(
        _execute_one(
            config,
            row,
            "small",
            (time.perf_counter(), 0.0, decision),
            time.perf_counter(),
            manifest,
            FailedFakeProvider(),
            store,
            "offline-owner",
            _test_exposure_calculator=OfflineFixtureBound(),
        )
    )
    reservation = RunStore(tmp_path / "runs.sqlite3").budget_reservations(manifest.run_id)[0]
    assert reservation["status"] == "observed"
    assert reservation["observed_micros"] == 300
    assert reservation["exposure_micros"] == 700
    assert store.execution_state(manifest.run_id, row.sample_id, "small") == "provider_failure"


def test_retry_and_fallback_exposure_envelope_covers_every_attempt():
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="provider/primary-small",
        strong_model_identifier="provider/primary-strong",
        small_fallback_identifier="provider/fallback-small",
        strong_fallback_identifier="provider/fallback-strong",
        fallback_policy="operational",
        max_retries=2,
        allow_paid_retries=True,
        paid_retry_approval_id="offline-test-retry",
    )
    assert attempt_model_envelope(config, "small") == (
        "provider/primary-small",
        "provider/primary-small",
        "provider/primary-small",
        "provider/fallback-small",
    )
    assert len(attempt_model_envelope(config, "strong")) == config.max_retries + 2


def test_provider_exposure_calculation_rejects_unknown_tokenizer(monkeypatch):
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="gemini/gemini-3.5-flash-lite",
        strong_model_identifier="gemini/gemini-3.8-flash",
    )
    from app.gateway import GatewayRequest

    request = GatewayRequest(
        request_id="offline",
        model_identifier=config.small_model_identifier,
        system_prompt="system",
        user_content="user",
        timeout_seconds=1,
        max_retries=0,
        retry_delay_seconds=0,
        max_tokens=config.max_tokens,
    )
    capability = provider_capabilities.resolve_capability(config.small_model_identifier)
    assert capability is not None
    monkeypatch.setitem(
        provider_capabilities.CAPABILITIES,
        capability.model_id,
        replace(capability, all_billable_categories_supported=True),
    )
    with pytest.raises(UnboundedExposureError, match="input token upper bound unavailable"):
        ProviderExposureCalculator().calculate(config, request, (config.small_model_identifier,))


def test_exposure_amount_covers_retries_and_fallback(monkeypatch):
    from app.gateway import GatewayRequest

    primary = "gemini/gemini-3.5-flash-lite"
    fallback = "gemini/gemini-3.8-flash"
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier=primary,
        strong_model_identifier="gemini/gemini-3.8-flash",
        small_fallback_identifier=fallback,
        strong_fallback_identifier="gemini/gemini-3.5-flash-lite",
        fallback_policy="operational",
        max_retries=1,
        allow_paid_retries=True,
        paid_retry_approval_id="offline-test-retry",
        max_tokens=5,
    )
    for model_id in (primary, fallback):
        capability = provider_capabilities.resolve_capability(model_id)
        assert capability is not None
        monkeypatch.setitem(
            provider_capabilities.CAPABILITIES,
            model_id,
            replace(capability, all_billable_categories_supported=True),
        )

    class ExactFixtureCounter:
        def count_messages(self, model_id, messages):
            from experiments.token_accounting import CountKind, TokenCount

            assert messages[0]["role"] == "system"
            return TokenCount(10, "fixture-exact", "v1", CountKind.EXACT, True, True)

    request = GatewayRequest(
        request_id="offline",
        model_identifier=primary,
        system_prompt="system",
        user_content="user",
        timeout_seconds=1,
        max_retries=1,
        retry_delay_seconds=0,
        fallback_model_identifier=fallback,
        max_tokens=5,
    )
    model_sequence = attempt_model_envelope(config, "small")
    exposure = ProviderExposureCalculator(ExactFixtureCounter()).calculate(
        config, request, model_sequence
    )
    expected_usd = Decimal(2) * (
        Decimal(10) * Decimal("0.30") + Decimal(5) * Decimal("2.50")
    ) / Decimal(1_000_000) + (
        Decimal(10) * Decimal("0.75") + Decimal(5) * Decimal("3.75")
    ) / Decimal(1_000_000)
    assert len(model_sequence) == 3
    assert exposure.amount_micros_usd == int(
        (expected_usd * Decimal(1_000_000)).to_integral_value(rounding=ROUND_CEILING)
    )
    assert exposure.provenance["attempt_model_identifiers"] == list(model_sequence)


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


def test_catalog_price_mismatch_and_provider_model_identity_mismatch_fail_closed(monkeypatch):
    model = model_catalog.resolve_model("gemini/gemini-3.8-flash")
    capability = provider_capabilities.resolve_capability("gemini/gemini-3.8-flash")
    assert model is not None and capability is not None
    monkeypatch.setitem(
        model_catalog.CANDIDATES,
        "gemini_flash",
        replace(model, input_usd_per_million=Decimal(999)),
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
    mismatch = assess_paid_execution(config)
    assert "pricing_catalog_mismatch" in mismatch.reason_codes

    monkeypatch.setitem(
        provider_capabilities.CAPABILITIES,
        capability.model_id,
        replace(capability, provider="wrong-provider"),
    )
    identity = assess_paid_execution(config)
    assert "provider_model_identity_mismatch" in identity.reason_codes
