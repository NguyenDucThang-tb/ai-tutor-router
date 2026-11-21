from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.errors import ErrorCategory, GatewayError
from app.gateway import MockGateway
from datasets.provenance import sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord
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
from experiments.records import ExperimentRecord, RunManifest
from experiments.replay import replay_records
from experiments.runner import _manifest_key, create_run_manifest, run_benchmark
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
    with pytest.raises(ValueError):
        BenchmarkConfig.model_validate(data)
    config = BenchmarkConfig.model_validate(
        {**data, "allow_real_provider": True, "experiment_approval_id": "approved-1"}
    )
    assert config.mode == "real"


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
        task_type="word_problem",
        provenance={"source_hash": hashlib.sha256(b"line-1").hexdigest()},
    )


def test_replay_matches_source_hash_and_re_evaluates(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    replay = replay_records(path, [sample(1)], "run-new")
    assert replay[0].execution_mode == "replay"
    assert replay[0].run_id == "run-new"
    assert replay[0].evaluation.is_correct is True


def test_metrics_separate_unknown_cost_and_report_intervals():
    result = summarize_records([_record()])
    row = result["groups"]["validation:small"]
    assert row["accuracy"] == 1
    assert row["accuracy_wilson_95"]["lower"] < 1
    assert row["known_calculated_cost_usd"] is None
    assert row["cost_is_complete"] is False


def test_unsupported_output_does_not_enter_accuracy_denominator():
    unknown = _record().model_copy(update={"evaluation": evaluate_answer("MOCK RESPONSE", "2")})
    row = summarize_records([unknown])["groups"]["validation:small"]
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


def _manifest() -> DatasetManifest:
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
        assigned_split_counts={"train": 0, "validation": 1, "test": 0},
        dataset_content_sha256="a" * 64,
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
    _, first = asyncio.run(run_benchmark(config, rows, _manifest(), run_dir, "validation", gateway))
    _, second = asyncio.run(
        run_benchmark(config, rows, _manifest(), run_dir, "validation", gateway)
    )
    assert len(first) == len(second) == 2
    assert gateway.calls == 2
    assert {row.model_tier for row in first} == {"small", "strong"}
    assert all(row.execution_mode == "mock" for row in first)


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
        run_benchmark(config, [row], _manifest(), tmp_path / "run", "validation", FailedGateway())
    )
    assert len(results) == 2
    assert all(result.execution_status == "provider_failure" for result in results)
    assert all(result.cost.calculated_usd is None for result in results)

    store = RunStore(tmp_path / "store.sqlite3")
    runtime_manifest: RunManifest = create_run_manifest(config, [row], _manifest(), "validation")
    store.ensure_run(runtime_manifest, _manifest_key(runtime_manifest))
    with pytest.raises(ValueError):
        store.ensure_run(runtime_manifest, "different-manifest")
