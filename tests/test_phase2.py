from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.errors import ErrorCategory, GatewayError
from app.gateway import CostInfo, ExecutionAttempt, MockGateway
from datasets.gsm8k import (
    LICENSE,
    SOURCE_ORGANIZATION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    VERIFIED_PREPARED_CONTENT_SHA256,
    VERIFIED_RAW_FILE_SHA256,
    VERIFIED_RAW_FILE_SIZES,
    VERIFIED_SPLIT_COUNTS,
    raw_file_url,
)
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord, SourceFileManifest
from datasets.splits import split_training_records
from evaluation.evaluator import evaluate_answer
from evaluation.metrics import summarize_records
from evaluation.rubrics import (
    HumanLabel,
    TutoringRubric,
    correctness_sufficiency_label,
    review_queue,
)
from evaluation.schemas import EvaluationStatus
from experiments.config import BenchmarkConfig
from experiments.prompts import PROMPT_SHA256, PROMPT_VERSION
from experiments.records import ExperimentRecord, RunManifest
from experiments.replay import replay_records
from experiments.runner import (
    _manifest_key,
    _record_scientifically_eligible,
    create_run_manifest,
    run_benchmark,
)
from experiments.store import RunStore


def sample(index: int, question: str | None = None) -> DatasetRecord:
    source_hash = hashlib.sha256(f"line-{index}".encode()).hexdigest()
    return DatasetRecord(
        sample_id=f"gsm8k-train-{index:06d}",
        dataset="gsm8k",
        dataset_revision="a" * 40,
        original_split="train",
        assigned_split="train",
        question=question or f"What is {index} plus one?",
        reference_answer=str(index + 1),
        reference_solution=f"{index}+1",
        task_type="word_problem",
        source_id=str(index),
        source_hash=source_hash,
    )


@pytest.mark.parametrize(
    ("output", "reference", "status"),
    [
        ("Work: 1+1\nFINAL ANSWER: 2", "2", EvaluationStatus.CORRECT),
        ("FINAL ANSWER: 2", "3", EvaluationStatus.INCORRECT),
        ("answer 2", "2", EvaluationStatus.UNPARSEABLE),
        ("FINAL ANSWER: 2 or 3", "2", EvaluationStatus.AMBIGUOUS),
        ("FINAL ANSWER: \\sin(x)", "0", EvaluationStatus.UNSUPPORTED),
        ("FINAL ANSWER: 1/0", "0", EvaluationStatus.UNPARSEABLE),
    ],
)
def test_evaluator_has_explicit_statuses(output, reference, status):
    assert evaluate_answer(output, reference).status == status


def test_safe_symbolic_equivalence():
    assert evaluate_answer("FINAL ANSWER: x+x", "2*x").is_correct is True
    result = evaluate_answer("FINAL ANSWER: __import__('os')", "0")
    assert result.status in {EvaluationStatus.UNSUPPORTED, EvaluationStatus.UNPARSEABLE}


def test_dataset_split_is_deterministic_and_groups_duplicates():
    records = [
        sample(1, "Find the total of 4 red balls and 5 blue balls."),
        sample(2, "Find the total of 4 red balls and 5 blue balls!"),
        sample(3),
        sample(4),
        sample(5),
    ]
    first, pairs = split_training_records(records, seed=27, train_fraction=0.6)
    second, _ = split_training_records(records, seed=27, train_fraction=0.6)
    split_by_id = {row.sample_id: row.assigned_split for row in first}
    assert split_by_id == {row.sample_id: row.assigned_split for row in second}
    duplicate_ids = {pairs[0].left_sample_id, pairs[0].right_sample_id}
    assert len({split_by_id[sample_id] for sample_id in duplicate_ids}) == 1


def test_dataset_schema_hash_and_official_test_preservation():
    row = sample(7)
    with pytest.raises(ValueError):
        DatasetRecord.model_validate({"sample_id": "incomplete"})
    official_test = row.model_copy(
        update={
            "sample_id": "gsm8k-test-000007",
            "original_split": "test",
            "assigned_split": "train",
        }
    )
    assigned, _ = split_training_records(
        [row, sample(8), sample(9), official_test], seed=42, train_fraction=0.5
    )
    preserved = next(value for value in assigned if value.sample_id == official_test.sample_id)
    assert preserved.assigned_split == "test"
    assert sha256_records([row, sample(8)]) == sha256_records([sample(8), row])


def test_real_provider_requires_explicit_gate_and_approval():
    data = {
        "mode": "real",
        "provider": "litellm",
        "small_model_identifier": "a",
        "strong_model_identifier": "b",
    }
    # A real config can be loaded for offline preflight without granting execution.
    dry_run_config = BenchmarkConfig.model_validate(data)
    assert dry_run_config.mode == "real" and not dry_run_config.allow_real_provider
    with pytest.raises(ValueError, match="allow_real_provider"):
        BenchmarkConfig.model_validate({**data, "allow_real_provider": True})
    config = BenchmarkConfig.model_validate(
        {**data, "allow_real_provider": True, "experiment_approval_id": "approved-1"}
    )
    assert config.mode == "real"
    with pytest.raises(ValueError, match="paid retries/reruns"):
        BenchmarkConfig.model_validate(
            {
                **data,
                "allow_real_provider": True,
                "experiment_approval_id": "approved-1",
                "max_retries": 1,
            }
        )
    paid_retry_config = BenchmarkConfig.model_validate(
        {
            **data,
            "allow_real_provider": True,
            "experiment_approval_id": "approved-1",
            "max_retries": 1,
            "allow_paid_retries": True,
            "paid_retry_approval_id": "retry-approved-2",
        }
    )
    assert paid_retry_config.max_retries == 1


def test_scientific_eligibility_requires_provenance_and_is_not_correctness():
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="small",
        strong_model_identifier="strong",
        allow_real_provider=True,
        experiment_approval_id="exp-approved",
        max_samples=1,
        max_requests=2,
        max_retries=0,
    )
    row = sample(1).model_copy(
        update={
            "dataset": "gsm8k",
            "dataset_revision": SOURCE_REVISION,
            "assigned_split": "validation",
        }
    )
    source = _manifest([row]).model_copy(
        update={
            "dataset": "gsm8k",
            "source_organization": SOURCE_ORGANIZATION,
            "source_url": SOURCE_REPOSITORY,
            "license": LICENSE,
            "revision": SOURCE_REVISION,
            "raw_files": {
                split: SourceFileManifest(
                    original_split=split,
                    source_url=raw_file_url(split),
                    sha256=VERIFIED_RAW_FILE_SHA256[split],
                    size_bytes=VERIFIED_RAW_FILE_SIZES[split],
                    downloaded_at_utc=datetime.now(UTC),
                )
                for split in ("train", "test")
            },
            "dataset_content_sha256": VERIFIED_PREPARED_CONTENT_SHA256,
            "assigned_split_counts": VERIFIED_SPLIT_COUNTS,
        }
    )
    manifest = create_run_manifest(config, [row], source, "validation")
    assert manifest.scientific_protocol_eligible is True
    synthetic_manifest = create_run_manifest(
        config,
        [sample(1).model_copy(update={"assigned_split": "validation"})],
        _manifest(),
        "validation",
    )
    assert synthetic_manifest.scientific_protocol_eligible is False
    assert synthetic_manifest.scientific_eligibility_reasons
    mock_record = _record()
    attempt = replace(
        mock_record.attempts[0],
        request_id="req-real",
        provider="litellm",
        model_identifier="small",
    )
    record = mock_record.model_copy(
        update={
            "sample_id": row.sample_id,
            "requested_model_identifier": "small",
            "executed_model_identifier": "small",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "provider": "litellm",
            "execution_mode": "real",
            "model_output": "no delimiter",
            "evaluation": evaluate_answer("no delimiter", row.reference_answer),
            "evaluation_complete": True,
            "attempts": [attempt],
            "provenance": {
                "dataset": row.dataset,
                "dataset_revision": row.dataset_revision,
                "source_hash": row.source_hash,
                "commit_sha": "c" * 40,
                "request_id": "req-real",
                "output_sha256": hashlib.sha256(b"no delimiter").hexdigest(),
            },
        }
    )
    assert record.evaluation.is_correct is None
    assert _record_scientifically_eligible(record, manifest, config) is True
    assert (
        _record_scientifically_eligible(
            record.model_copy(update={"provider": "mock"}), manifest, config
        )
        is False
    )
    assert (
        _record_scientifically_eligible(
            record.model_copy(update={"executed_model_identifier": "unconfigured-model"}),
            manifest,
            config,
        )
        is False
    )


def test_human_review_labels_remain_unreviewed():
    base = {
        "sample_id": "id",
        "run_id": "run",
        "model_tier": "small",
        "label": "correct",
        "rationale": "checked",
        "reviewer_id": "r1",
        "reviewed_at_utc": "2026-01-01T00:00:00Z",
    }
    assert HumanLabel(**base).reviewed is False
    assert HumanLabel(**base, reviewed=True).reviewed is True
    assert (
        TutoringRubric(
            sample_id="id",
            run_id="run",
            model_tier="small",
            reviewer_id="r1",
            reviewed_at_utc="2026-01-01T00:00:00Z",
            educational_usefulness=4,
        ).educational_usefulness
        == 4
    )


def _record() -> ExperimentRecord:
    evaluation = evaluate_answer("FINAL ANSWER: 2", "2")
    attempt = ExecutionAttempt(
        request_id="req-1",
        attempt_number=1,
        model_identifier="small",
        provider="mock",
        fallback=False,
        latency_ms=1.0,
        status="success",
    )
    return ExperimentRecord(
        run_id="run-old",
        sample_id="gsm8k-train-000001",
        split="validation",
        model_tier="small",
        requested_model_identifier="small",
        executed_model_identifier="small",
        selected_tier="small",
        prompt_version="test",
        prompt_sha256="b" * 64,
        provider="mock",
        execution_mode="mock",
        execution_status="success",
        model_output="FINAL ANSWER: 2",
        evaluation=evaluation,
        attempts=[attempt],
        task_type="word_problem",
        provenance={
            "source_hash": hashlib.sha256(b"line-1").hexdigest(),
            "dataset_revision": "a" * 40,
            "collection_plan_sha256": "d" * 64,
            "collection_plan_sample_ids_sha256": "e" * 64,
            "selection_policy": "versioned_collection_plan_v1",
            "commit_sha": "c" * 40,
            "request_id": "req-1",
            "output_sha256": hashlib.sha256(b"FINAL ANSWER: 2").hexdigest(),
        },
    )


def test_replay_matches_source_hash_and_re_evaluates(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    replay = replay_records(
        path, [sample(1).model_copy(update={"assigned_split": "validation"})], "run-new"
    )
    assert replay[0].execution_mode == "replay"
    assert replay[0].run_id == "run-new"
    assert replay[0].evaluation.is_correct is True
    assert replay[0].scientific_eligible is False
    assert replay[0].provenance["replay_provenance_status"] == "verified"
    assert replay[0].provenance["collection_plan_sha256"] == "d" * 64
    assert replay[0].provenance["collection_plan_sample_ids_sha256"] == "e" * 64


def test_replay_rejects_mismatched_source_or_output_provenance(tmp_path):
    row = sample(1).model_copy(update={"assigned_split": "validation"})
    source = tmp_path / "record.jsonl"
    source.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        replay_records(source, [row.model_copy(update={"source_hash": "f" * 64})], "run-x")
    corrupted = _record().model_copy(
        update={"provenance": {**_record().provenance, "output_sha256": "f" * 64}}
    )
    source.write_text(corrupted.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        replay_records(source, [row], "run-x")


def test_replay_retains_failures_interruptions_unknowns_and_incomplete_pairs(tmp_path):
    success = _record()
    failure = success.model_copy(
        update={
            "model_tier": "strong",
            "requested_model_identifier": "strong",
            "executed_model_identifier": None,
            "execution_status": "provider_failure",
            "model_output": None,
            "evaluation": None,
            "evaluation_complete": False,
            "attempts": [replace(success.attempts[0], status="failed")],
            "cost": CostInfo(status="partial", unknown_attempt_count=1),
            "provenance": {**success.provenance, "error_category": "timeout"},
        }
    )
    interrupted_sample = sample(2).model_copy(update={"assigned_split": "validation"})
    interrupted = success.model_copy(
        update={
            "sample_id": interrupted_sample.sample_id,
            "execution_status": "interrupted",
            "model_output": None,
            "evaluation": None,
            "evaluation_complete": False,
            "attempts": [],
            "provenance": {
                **success.provenance,
                "source_hash": interrupted_sample.source_hash,
            },
        }
    )
    unknown_sample = sample(3).model_copy(update={"assigned_split": "validation"})
    uncertain_attempt = replace(
        success.attempts[0],
        status="outcome_unknown",
        cost=CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
    )
    unknown = success.model_copy(
        update={
            "sample_id": unknown_sample.sample_id,
            "model_tier": "strong",
            "requested_model_identifier": "strong",
            "execution_status": "outcome_unknown",
            "model_output": None,
            "evaluation": None,
            "evaluation_complete": False,
            "attempts": [uncertain_attempt],
            "cost": CostInfo(status="unknown", attempt_count=1, unknown_attempt_count=1),
            "provenance": {
                **success.provenance,
                "source_hash": unknown_sample.source_hash,
                "provider_outcome_unknown": True,
            },
        }
    )
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        "".join(row.model_dump_json() + "\n" for row in (success, failure, interrupted, unknown)),
        encoding="utf-8",
    )
    replayed = replay_records(
        source,
        [
            sample(1).model_copy(update={"assigned_split": "validation"}),
            interrupted_sample,
            unknown_sample,
        ],
        "replay-mixed",
    )
    assert len(replayed) == 4
    by_status = {row.execution_status: row for row in replayed}
    assert by_status["provider_failure"].cost.unknown_attempt_count == 1
    assert by_status["provider_failure"].provenance["error_category"] == "timeout"
    assert by_status["interrupted"].execution_status == "interrupted"
    assert by_status["interrupted"].provenance["replay_provenance_status"] == "incomplete"
    assert by_status["outcome_unknown"].provenance["provider_outcome_unknown"] is True
    metrics = summarize_records(replayed)
    assert metrics["paired_outcomes"]["execution_failure"] == 1
    assert metrics["paired_outcomes"]["incomplete_pair"] == 2


def test_metrics_separate_unknown_cost_and_report_intervals():
    result = summarize_records([_record()])
    row = result["groups"]["validation:small:disabled"]
    assert row["accuracy"] == 1
    assert row["accuracy_wilson_95"]["lower"] < 1
    assert row["known_calculated_cost_usd"] is None
    assert row["cost_is_complete"] is False


def test_unsupported_output_does_not_enter_accuracy_denominator():
    unknown = _record().model_copy(update={"evaluation": evaluate_answer("MOCK RESPONSE", "2")})
    row = summarize_records([unknown])["groups"]["validation:small:disabled"]
    assert row["accuracy"] is None
    assert row["evaluated"] == 0
    assert row["successful_evaluated"] == 1
    assert row["evaluator_coverage"] == 0


def test_review_queue_only_contains_nonautomatic_cases():
    ambiguous = _record().model_copy(
        update={
            "evaluation": evaluate_answer("FINAL ANSWER: 2 or 3", "2"),
        }
    )
    queued = review_queue([_record(), ambiguous])
    assert len(queued) == 1
    assert queued[0]["reviewed"] is False


def test_sufficiency_labels_are_real_only_and_test_is_forbidden():
    record = _record()
    criterion = "correctness-based-sufficiency-v1"
    assert correctness_sufficiency_label(record, frozen_criterion=criterion) is None
    real = record.model_copy(update={"execution_mode": "real"})
    assert correctness_sufficiency_label(real, frozen_criterion=criterion) == 1
    with pytest.raises(ValueError):
        correctness_sufficiency_label(
            real.model_copy(update={"split": "test"}), frozen_criterion=criterion
        )


def _manifest(records: list[DatasetRecord] | None = None) -> DatasetManifest:
    records = records or [sample(1).model_copy(update={"assigned_split": "validation"})]
    split_counts = {split: 0 for split in ("train", "validation", "test")}
    for record in records:
        split_counts[record.assigned_split] += 1
    return DatasetManifest(
        dataset="synthetic-test",
        source_organization="test",
        source_url="local://fixture",
        revision="fixture-v1",
        license="CC0-1.0",
        access_note="synthetic fixture",
        downloaded_at_utc=datetime.now(UTC),
        raw_files={},
        transformation_history=["synthetic"],
        seed=42,
        train_fraction=0.9,
        near_duplicate_threshold=0.9,
        assigned_split_counts=split_counts,
        dataset_content_sha256=sha256_records(records),
        processed_file_sha256={},
    )


def test_offline_benchmark_is_paired_and_completed_calls_are_not_repeated(tmp_path: Path):
    class AnswerGateway(MockGateway):
        def __init__(self):
            super().__init__("FINAL ANSWER: 2")
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            return await super().complete(request)

    config = BenchmarkConfig(
        mode="mock",
        provider="mock",
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        max_retries=0,
    )
    rows = [sample(1).model_copy(update={"assigned_split": "validation"})]
    gateway = AnswerGateway()
    run_dir = tmp_path / "run"
    _, first = asyncio.run(
        run_benchmark(config, rows, _manifest(rows), run_dir, "validation", gateway)
    )
    _, second = asyncio.run(
        run_benchmark(config, rows, _manifest(rows), run_dir, "validation", gateway)
    )
    assert len(first) == len(second) == 2
    assert gateway.calls == 2
    assert {row.model_tier for row in first} == {"small", "strong"}
    assert all(row.execution_mode == "mock" for row in first)


def test_latency_boundaries_include_queue_and_separate_model_from_pipeline(tmp_path: Path):
    class DelayedGateway(MockGateway):
        async def complete(self, request):
            await asyncio.sleep(0.025)
            result = await super().complete(request)
            attempt = replace(result.attempts[0], latency_ms=7.0)
            return replace(result, attempts=(attempt,), latency_ms=12.0)

    config = BenchmarkConfig(
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        max_retries=0,
        concurrency=1,
    )
    row = sample(1).model_copy(update={"assigned_split": "validation"})
    _, records = asyncio.run(
        run_benchmark(
            config, [row], _manifest([row]), tmp_path / "timed-run", "validation", DelayedGateway()
        )
    )
    assert len(records) == 2
    first, second = records
    assert first.metrics.feature_extraction_ms >= 0
    assert first.metrics.router_decision_ms >= 0
    assert first.metrics.model_only_latency_ms == 7.0
    assert first.metrics.gateway_elapsed_ms == 12.0
    assert first.metrics.evaluator_latency_ms >= 0
    assert first.metrics.pipeline_end_to_end_ms >= first.metrics.gateway_elapsed_ms
    assert second.metrics.scheduling_queue_ms >= 15.0
    assert second.metrics.pipeline_end_to_end_ms > second.metrics.scheduling_queue_ms
    assert not hasattr(second.metrics, "end_to_end_latency_ms")


def test_provider_failure_is_persisted_and_manifest_mismatch_fails_closed(tmp_path: Path):
    class FailedGateway:
        provider = "mock"

        async def complete(self, request):
            raise GatewayError(ErrorCategory.TIMEOUT, "timeout")

    config = BenchmarkConfig(
        mode="mock",
        provider="mock",
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        max_retries=0,
    )
    row = sample(1).model_copy(update={"assigned_split": "validation"})
    _, results = asyncio.run(
        run_benchmark(
            config, [row], _manifest([row]), tmp_path / "run", "validation", FailedGateway()
        )
    )
    assert len(results) == 2
    assert all(result.execution_status == "provider_failure" for result in results)
    assert all(result.cost.calculated_usd is None for result in results)

    store = RunStore(tmp_path / "store.sqlite3")
    runtime_manifest: RunManifest = create_run_manifest(
        config, [row], _manifest([row]), "validation"
    )
    store.ensure_run(runtime_manifest, _manifest_key(runtime_manifest))
    with pytest.raises(ValueError):
        store.ensure_run(runtime_manifest, "different-manifest")


def test_benchmark_rejects_records_not_matching_source_manifest(tmp_path: Path):
    config = BenchmarkConfig(
        small_model_identifier="small",
        strong_model_identifier="strong",
        max_samples=1,
        max_requests=2,
        max_retries=0,
    )
    row = sample(1).model_copy(update={"assigned_split": "validation"})
    source = _manifest([row]).model_copy(update={"dataset_content_sha256": "f" * 64})
    with pytest.raises(ValueError, match="dataset manifest content hash"):
        asyncio.run(run_benchmark(config, [row], source, tmp_path / "bad-source", "validation"))
