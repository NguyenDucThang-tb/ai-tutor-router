from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.gateway import CostInfo, ExecutionAttempt
from datasets import gsm8k
from datasets.provenance import sha256_bytes, sha256_records
from datasets.schemas import DatasetManifest, DatasetRecord, SourceFileManifest
from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from evaluation.metrics import paired_statistics
from evaluation.rubrics import phase3a_review_sample
from experiments.benchmark import main as benchmark_main
from experiments.collection_plan import create_plan, seal_plan
from experiments.config import BenchmarkConfig
from experiments.model_catalog import PAIRS
from experiments.preflight import main as preflight_main
from experiments.preflight import make_report
from experiments.pricing import estimate_usd, hard_budget_bound_supported
from experiments.prompts import PROMPT_SHA256, PROMPT_VERSION
from experiments.records import ExperimentRecord
from experiments.runner import run_benchmark, run_id_for
from experiments.training_data import export_training_data


def _dataset(tmp_path: Path, count: int = 12, selected_split: str = "validation"):
    rows = [
        DatasetRecord(
            sample_id=f"row-{index}",
            dataset="gsm8k",
            dataset_revision="rev1",
            original_split="train",
            assigned_split=selected_split if index < count else "test",
            question=("How many items? " + "number " * (index % 4 + 1)),
            reference_answer="1",
            source_id=str(index),
            source_hash=hashlib.sha256(f"{index}".encode()).hexdigest(),
        )
        for index in range(count + 2)
    ]
    directory = tmp_path / "processed"
    directory.mkdir()
    files = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row.assigned_split == split]
        content = "".join(row.model_dump_json() + "\n" for row in selected).encode()
        (directory / f"{split}.jsonl").write_bytes(content)
        files[split] = sha256_bytes(content)
    manifest = DatasetManifest(
        dataset="gsm8k",
        source_organization="fixture",
        source_url="fixture://source",
        revision="rev1",
        license="CC0",
        access_note="fixture",
        downloaded_at_utc=datetime.now(UTC),
        raw_files={},
        transformation_history=["fixture"],
        seed=42,
        train_fraction=0.9,
        near_duplicate_threshold=0.9,
        assigned_split_counts={
            "train": sum(r.assigned_split == "train" for r in rows),
            "validation": sum(r.assigned_split == "validation" for r in rows),
            "test": sum(r.assigned_split == "test" for r in rows),
        },
        dataset_content_sha256=sha256_records(rows),
        processed_file_sha256=files,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    return rows, manifest, directory, manifest_path


def _record(sample_id: str, tier: str, correct: bool | None, split: str = "validation"):
    record = ExperimentRecord(
        run_id="run-a",
        sample_id=sample_id,
        split=split,
        model_tier=tier,
        requested_model_identifier=f"{tier}-model",
        prompt_version="test-v1",
        prompt_sha256="a" * 64,
        provider="mock",
        execution_mode="mock",
        execution_status="success",
        model_output="FINAL ANSWER: 1",
        evaluation=evaluate_answer("FINAL ANSWER: 1", "1")
        if correct is True
        else (
            evaluate_answer("FINAL ANSWER: 2", "1")
            if correct is False
            else evaluate_answer("no answer", "1")
        ),
        evaluation_complete=True,
        task_type="math",
    )
    return record


def test_model_catalog_and_decimal_pricing_have_provenance():
    assert len(PAIRS) >= 3
    assert estimate_usd("openai/gpt-4.1-mini-2025-04-14", 1_000_000, 1_000_000) == 2
    assert estimate_usd("unknown/model", 100, 100) is None
    assert not hard_budget_bound_supported("openai/gpt-4.1-mini-2025-04-14")


def test_collection_plan_is_deterministic_stratified_and_rejects_test(tmp_path):
    rows, manifest, _, _ = _dataset(tmp_path, count=12)
    first = create_plan(rows, "validation", 10, 42)
    second = create_plan(rows, "validation", 10, 42)
    assert first == second
    assert first["selected_count"] == 10
    assert first["stratum_counts"]
    assert set(first["sample_ids"]) != {
        row.sample_id for row in sorted(rows, key=lambda row: row.sample_id)[:10]
    }
    assert first["dataset_content_sha256"] == manifest.dataset_content_sha256
    assert first["plan_sha256"]
    with pytest.raises(ValueError, match="train or validation"):
        create_plan(rows, "test", 1)


def test_preflight_is_offline_and_blocks_unknown_hard_spend_bound(tmp_path):
    _, _, dataset_dir, manifest_path = _dataset(tmp_path)
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="openai/gpt-4.1-mini-2025-04-14",
        strong_model_identifier="openai/gpt-4.1-2025-04-14",
        max_samples=1,
        max_requests=2,
        max_run_budget_usd=1,
        per_request_budget_usd=0.5,
    )
    report = make_report(
        config,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        environment={},
    )
    assert report["status"] == "BLOCKED"
    assert report["provider_calls_made"] == 0
    assert report["cost_projection_usd"]["hard_upper_bound"] is None
    assert any(
        check["check"] == "hard_spend_bound" and check["status"] == "BLOCKED"
        for check in report["checks"]
    )


def test_preflight_uses_exact_collection_plan_samples_and_hash(tmp_path):
    rows, _, dataset_dir, manifest_path = _dataset(tmp_path, count=12)
    plan = create_plan(rows, "validation", 5, seed=17)
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="openai/gpt-4.1-mini-2025-04-14",
        strong_model_identifier="openai/gpt-4.1-2025-04-14",
        max_samples=5,
        max_requests=10,
        max_run_budget_usd=1,
        per_request_budget_usd=0.5,
    )
    report = make_report(
        config,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        environment={},
        collection_plan=plan,
    )
    assert report["selected_sample_ids"] == plan["sample_ids"]
    assert report["selected_sample_ids_sha256"] == plan["sample_ids_sha256"]
    assert report["collection_plan_sha256"] == plan["plan_sha256"]
    assert report["split"] == "validation"


def test_preflight_train_split_is_explicit_and_must_match_plan(tmp_path, capsys):
    rows, _, dataset_dir, manifest_path = _dataset(tmp_path, count=12, selected_split="train")
    plan = create_plan(rows, "train", 5, seed=31)
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="openai/gpt-4.1-mini-2025-04-14",
        strong_model_identifier="openai/gpt-4.1-2025-04-14",
        max_samples=5,
        max_requests=10,
        max_run_budget_usd=1,
        per_request_budget_usd=0.5,
    )
    report = make_report(
        config,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        environment={},
        collection_plan=plan,
        split="train",
    )
    split_check = next(check for check in report["checks"] if check["check"] == "split_policy")
    assert report["split"] == "train"
    assert report["selected_sample_ids"] == plan["sample_ids"]
    assert report["selected_sample_ids_sha256"] == plan["sample_ids_sha256"]
    assert report["collection_plan_sha256"] == plan["plan_sha256"]
    assert "train-only" in split_check["detail"]

    config_path = tmp_path / "real.yaml"
    config_path.write_text(
        "mode: real\nprovider: litellm\nsmall_model_identifier: "
        "openai/gpt-4.1-mini-2025-04-14\nstrong_model_identifier: "
        "openai/gpt-4.1-2025-04-14\nmax_samples: 5\nmax_requests: 10\n"
        "max_run_budget_usd: 1\nper_request_budget_usd: 0.5\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "train-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    cli_report_path = tmp_path / "train-preflight.json"
    assert (
        preflight_main(
            [
                "--config",
                str(config_path),
                "--dataset-dir",
                str(dataset_dir),
                "--manifest",
                str(manifest_path),
                "--collection-plan",
                str(plan_path),
                "--split",
                "train",
                "--output",
                str(cli_report_path),
            ]
        )
        == 2
    )
    capsys.readouterr()
    cli_report = json.loads(cli_report_path.read_text())
    assert cli_report["split"] == "train"
    assert cli_report["selected_sample_ids"] == plan["sample_ids"]

    mismatched = make_report(
        config,
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        environment={},
        collection_plan=plan,
        split="validation",
    )
    assert mismatched["status"] == "FAIL"
    assert mismatched["split"] is None
    assert mismatched["selected_sample_ids"] == []
    assert any(
        check["status"] == "FAIL" and "does not match requested split" in check["detail"]
        for check in mismatched["checks"]
    )


def test_runner_executes_exact_planned_ids_as_paired_small_and_strong(tmp_path):
    from app.gateway import MockGateway

    class CapturingGateway(MockGateway):
        def __init__(self):
            super().__init__("FINAL ANSWER: 1")
            self.calls = []

        async def complete(self, request):
            self.calls.append((request.model_identifier, request.user_content))
            return await super().complete(request)

    rows, manifest, _, _ = _dataset(tmp_path, count=12)
    plan = create_plan(rows, "validation", 4, seed=17)
    config = BenchmarkConfig(
        small_model_identifier="mock-small",
        strong_model_identifier="mock-strong",
        max_samples=4,
        max_requests=8,
        concurrency=2,
    )
    gateway = CapturingGateway()
    run_manifest, executions = asyncio.run(
        run_benchmark(
            config,
            rows,
            manifest,
            tmp_path / "planned-run",
            "validation",
            gateway,
            collection_plan=plan,
        )
    )
    expected = set(plan["sample_ids"])
    assert {record.sample_id for record in executions} == expected
    assert len(executions) == 2 * len(expected)
    assert {
        tier: {r.sample_id for r in executions if r.model_tier == tier}
        for tier in ("small", "strong")
    } == {"small": expected, "strong": expected}
    assert {question for _, question in gateway.calls} == {
        next(row.question for row in rows if row.sample_id == sample_id) for sample_id in expected
    }
    assert run_manifest.collection_plan_sample_ids == plan["sample_ids"]
    assert run_manifest.collection_plan_sha256 == plan["plan_sha256"]
    assert run_manifest.source_dataset_content_sha256 == manifest.dataset_content_sha256
    assert all(
        record.provenance["collection_plan_sha256"] == plan["plan_sha256"] for record in executions
    )


def test_benchmark_and_preflight_clis_accept_same_versioned_plan(tmp_path, capsys):
    rows, _, dataset_dir, manifest_path = _dataset(tmp_path, count=12)
    plan = create_plan(rows, "validation", 3, seed=27)
    plan_path = tmp_path / "collection-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    mock_config = tmp_path / "mock.yaml"
    mock_config.write_text(
        "mode: mock\nprovider: mock\nsmall_model_identifier: mock-small\n"
        "strong_model_identifier: mock-strong\nmax_samples: 3\nmax_requests: 6\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    assert (
        benchmark_main(
            [
                "run",
                "--config",
                str(mock_config),
                "--dataset-dir",
                str(dataset_dir),
                "--manifest",
                str(manifest_path),
                "--collection-plan",
                str(plan_path),
                "--output-dir",
                str(runs_dir),
            ]
        )
        == 0
    )
    benchmark_result = json.loads(capsys.readouterr().out)
    run_dir = Path(benchmark_result["run_dir"])
    run_records = [
        json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    benchmark_ids = sorted({record["sample_id"] for record in run_records})

    real_config = tmp_path / "real.yaml"
    real_config.write_text(
        "mode: real\nprovider: litellm\nsmall_model_identifier: "
        "openai/gpt-4.1-mini-2025-04-14\nstrong_model_identifier: "
        "openai/gpt-4.1-2025-04-14\nmax_samples: 3\nmax_requests: 6\n"
        "max_run_budget_usd: 1\nper_request_budget_usd: 0.5\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "preflight.json"
    assert (
        preflight_main(
            [
                "--config",
                str(real_config),
                "--dataset-dir",
                str(dataset_dir),
                "--manifest",
                str(manifest_path),
                "--collection-plan",
                str(plan_path),
                "--output",
                str(report_path),
            ]
        )
        == 2
    )  # The existing paid spend gate remains BLOCKED.
    capsys.readouterr()
    report = json.loads(report_path.read_text())
    assert report["selected_sample_ids"] == plan["sample_ids"]
    assert benchmark_ids == sorted(report["selected_sample_ids"])
    assert report["selected_sample_ids_sha256"] == plan["sample_ids_sha256"]


def test_changed_plan_gets_distinct_checkpoint_identity_and_request_cap_fails(tmp_path):
    rows, manifest, _, _ = _dataset(tmp_path, count=12)
    first = create_plan(rows, "validation", 4, seed=17)
    changed = create_plan(rows, "validation", 4, seed=18)
    config = BenchmarkConfig(
        small_model_identifier="mock-small",
        strong_model_identifier="mock-strong",
        max_samples=4,
        max_requests=8,
    )
    selected_first = [
        next(row for row in rows if row.sample_id == sid) for sid in first["sample_ids"]
    ]
    selected_changed = [
        next(row for row in rows if row.sample_id == sid) for sid in changed["sample_ids"]
    ]
    assert run_id_for(config, selected_first, first["plan_sha256"]) != run_id_for(
        config, selected_changed, changed["plan_sha256"]
    )
    too_small = config.model_copy(update={"max_requests": 6})
    with pytest.raises(ValueError, match="complete collection plan"):
        asyncio.run(
            run_benchmark(
                too_small,
                rows,
                manifest,
                tmp_path / "cap-run",
                "validation",
                collection_plan=first,
            )
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("checksum", "dataset checksum"),
        ("revision", "dataset revision"),
        ("sample_count", "sample count"),
        ("plan_hash", "plan hash mismatch"),
        ("missing", "missing"),
        ("duplicate", "duplicate"),
        ("cross_split", "does not match requested split"),
        ("official_test", "official test"),
    ],
)
def test_collection_plan_rejects_invalid_dataset_binding(tmp_path, mutation, match):
    rows, _, _, _ = _dataset(tmp_path, count=12)
    plan = create_plan(rows, "validation", 3, seed=11)
    plan = dict(plan)
    if mutation == "checksum":
        plan["dataset_content_sha256"] = "0" * 64
    elif mutation == "revision":
        plan["dataset_revision"] = "other-revision"
    elif mutation == "sample_count":
        plan["selected_count"] -= 1
    elif mutation == "plan_hash":
        pass
    elif mutation == "missing":
        plan["sample_ids"][0] = "not-in-dataset"
        plan["sample_ids_sha256"] = hashlib.sha256(
            "\n".join(plan["sample_ids"]).encode()
        ).hexdigest()
    elif mutation == "duplicate":
        plan["sample_ids"][1] = plan["sample_ids"][0]
    elif mutation == "cross_split":
        plan["split"] = "train"
        plan["sample_ids_sha256"] = hashlib.sha256(
            "\n".join(plan["sample_ids"]).encode()
        ).hexdigest()
    elif mutation == "official_test":
        plan["sample_ids"][0] = next(row.sample_id for row in rows if row.assigned_split == "test")
        plan["sample_ids"].sort()
        plan["sample_ids_sha256"] = hashlib.sha256(
            "\n".join(plan["sample_ids"]).encode()
        ).hexdigest()
    plan = seal_plan(plan)
    if mutation == "plan_hash":
        plan["plan_sha256"] = "0" * 64
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="openai/gpt-4.1-mini-2025-04-14",
        strong_model_identifier="openai/gpt-4.1-2025-04-14",
        max_samples=3,
        max_requests=6,
        max_run_budget_usd=1,
        per_request_budget_usd=0.5,
    )
    report = make_report(
        config,
        dataset_dir=tmp_path / "processed",
        manifest_path=tmp_path / "manifest.json",
        environment={},
        collection_plan=plan,
    )
    assert report["status"] in {"FAIL", "BLOCKED"}
    assert any(check["status"] == "FAIL" and match in check["detail"] for check in report["checks"])


def test_paid_runner_refuses_litellm_before_dataset_or_provider_invocation(tmp_path):
    config = BenchmarkConfig(
        mode="real",
        provider="litellm",
        small_model_identifier="openai/gpt-4.1-mini-2025-04-14",
        strong_model_identifier="openai/gpt-4.1-2025-04-14",
        allow_real_provider=True,
        experiment_approval_id="approval",
    )
    with pytest.raises(ValueError, match="paid invocation blocked"):
        asyncio.run(run_benchmark(config, [], None, tmp_path / "run", "validation"))


def test_paired_bootstrap_keeps_incomplete_unknown_and_failures_separate():
    records = [
        _record("1", "small", True),
        _record("1", "strong", False),
        _record("2", "small", False),
        _record("2", "strong", True),
        _record("3", "small", None),
        _record("3", "strong", True),
        _record("4", "small", True),
    ]
    # The first assertion confirms the analysis refuses mock/replay evidence.
    assert paired_statistics(records)["status"] == "NOT_COMPUTED_NONREAL_RECORDS"
    records = [
        record.model_copy(
            update={"execution_mode": "real", "provider": "litellm", "scientific_eligible": True}
        )
        for record in records
    ]
    records[5] = records[5].model_copy(update={"execution_status": "provider_failure"})
    stats = paired_statistics(records, seed=42, bootstrap_samples=100)
    assert stats["paired_determinate_n"] == 2
    assert stats["incomplete_pairs"] == 1
    assert stats["execution_failure_pairs"] == 1
    assert stats["small_minus_strong_accuracy"] == 0
    assert "Oracle upper bound" in stats["oracle_label"]


def test_review_sample_balances_statuses_and_never_marks_human_review():
    records = [
        _record("correct", "small", True),
        _record("incorrect", "small", False),
        _record("unknown", "small", None),
    ]
    queue = phase3a_review_sample(records, per_status=1, seed=8)
    assert {row["automatic_status"] for row in queue} == {"correct", "incorrect", "unparseable"}
    assert all(row["reviewed"] is False and row["reviewer_id"] is None for row in queue)


def test_training_export_excludes_mock_test_and_does_not_leak_answers(tmp_path):
    rows, manifest, _, _ = _dataset(tmp_path)
    mock = _record(rows[0].sample_id, "small", True)
    test = _record(rows[-1].sample_id, "small", True, split="test")
    result = export_training_data([mock, test], rows, manifest, tmp_path / "out")
    assert result["excluded_official_test_records"] == 1
    assert not result["scientific_training_data_available"]
    assert (tmp_path / "out" / "train.jsonl").read_text() == ""
    unknown = (
        json.loads((tmp_path / "out" / "unknown.jsonl").read_text().splitlines()[0])
        if (tmp_path / "out" / "unknown.jsonl").read_text()
        else {}
    )
    assert "model_output" not in unknown and "reference_answer" not in unknown


def test_training_export_separates_verified_real_labels_from_query_features(tmp_path, monkeypatch):
    rows, manifest, _, _ = _dataset(tmp_path)
    plan = create_plan(rows, "validation", 12, seed=23)
    raw_hashes = {"train": "a" * 64, "test": "b" * 64}
    raw_sizes = {"train": 11, "test": 13}
    counts = {
        "train": sum(row.assigned_split == "train" for row in rows),
        "validation": sum(row.assigned_split == "validation" for row in rows),
        "test": sum(row.assigned_split == "test" for row in rows),
    }
    for name, value in {
        "SOURCE_REVISION": "rev1",
        "SOURCE_REPOSITORY": "fixture://source",
        "SOURCE_ORGANIZATION": "fixture",
        "LICENSE": "CC0",
        "VERIFIED_RAW_FILE_SHA256": raw_hashes,
        "VERIFIED_RAW_FILE_SIZES": raw_sizes,
        "VERIFIED_PREPARED_CONTENT_SHA256": manifest.dataset_content_sha256,
        "VERIFIED_SPLIT_COUNTS": counts,
    }.items():
        monkeypatch.setattr(gsm8k, name, value)
    manifest = manifest.model_copy(
        update={
            "raw_files": {
                split: SourceFileManifest(
                    original_split=split,
                    source_url=gsm8k.raw_file_url(split),
                    sha256=raw_hashes[split],
                    size_bytes=raw_sizes[split],
                    downloaded_at_utc=datetime.now(UTC),
                )
                for split in ("train", "test")
            }
        }
    )
    sample = rows[0]
    output = "FINAL ANSWER: 1"
    model_id = "openai/gpt-4.1-mini-2025-04-14"
    native_id = model_id.split("/", 1)[-1]
    source = ExperimentRecord(
        run_id="verified-run",
        sample_id=sample.sample_id,
        split="validation",
        model_tier="small",
        requested_model_identifier=model_id,
        executed_model_identifier=native_id,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=PROMPT_SHA256,
        evaluator_version=EVALUATOR_VERSION,
        provider="litellm",
        execution_mode="real",
        execution_status="success",
        model_output=output,
        evaluation=evaluate_answer(output, sample.reference_answer),
        evaluation_complete=True,
        scientific_eligible=True,
        attempts=[
            ExecutionAttempt(
                request_id="request-1",
                attempt_number=1,
                model_identifier=native_id,
                provider="litellm",
                fallback=False,
                latency_ms=10,
                status="success",
                cost=CostInfo(),
            )
        ],
        task_type=sample.task_type,
        provenance={
            "dataset": sample.dataset,
            "dataset_revision": sample.dataset_revision,
            "source_hash": sample.source_hash,
            "sample_is_synthetic": False,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "collection_plan_sha256": plan["plan_sha256"],
            "selection_policy": "versioned_collection_plan_v1",
        },
    )
    result = export_training_data(
        [source], rows, manifest, tmp_path / "eligible", collection_plan=plan
    )
    assert result["counts"]["validation"] == 1
    exported = json.loads((tmp_path / "eligible" / "validation.jsonl").read_text().splitlines()[0])
    assert exported["small_model_sufficient"] == 1
    assert exported["label_status"] == "determinate"
    assert "model_output" not in exported and "reference_answer" not in exported
    assert "small_model_sufficient" not in exported["query_features"]
    assert exported["source_collection_plan_sha256"] == plan["plan_sha256"]
    assert result["collection_plan_sha256"] == plan["plan_sha256"]

    corrupted = source.model_copy(
        update={"evaluation": evaluate_answer("FINAL ANSWER: 2", sample.reference_answer)}
    )
    rejected = export_training_data(
        [corrupted], rows, manifest, tmp_path / "bad-evaluation", collection_plan=plan
    )
    assert rejected["counts"]["validation"] == 0
    assert rejected["excluded_records"]["unverified_or_ineligible_execution"] == 1
