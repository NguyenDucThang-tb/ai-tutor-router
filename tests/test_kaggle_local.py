import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.kaggle_local import (
    LOCAL_RECORD_SCHEMA,
    evaluate_records,
    execute_samples,
    model_metrics,
    notebook_scientific_status,
    paired_metrics,
    paired_rows,
    read_records,
    sample_ids_sha256,
    write_paired_csv,
    write_records,
)


def sample(sample_id: str, reference: str = "7"):
    return SimpleNamespace(sample_id=sample_id, reference_answer=reference, question="What is 3+4?")


def record(sample_id: str, *, correct: bool | None, status: str = "completed"):
    return {
        "schema_version": LOCAL_RECORD_SCHEMA,
        "provider": "local_transformers",
        "scientific_eligible": False,
        "sample_id": sample_id,
        "status": status,
        "model_output": f"FINAL ANSWER: {sample_id}",
        "reference_answer": "7",
        "inference_latency_s": 1.0,
        "output_tokens": 5,
        "evaluation": (
            {"status": "correct" if correct else "incorrect", "is_correct": correct}
            if status == "completed" and correct is not None
            else None
        ),
    }


def test_disabled_execution_never_invokes_model(tmp_path):
    called = []
    result = execute_samples(
        [sample("a"), sample("b")],
        tmp_path / "checkpoint.jsonl",
        model_id="Qwen/example",
        model_revision="a" * 40,
        run_enabled=False,
        infer_one=lambda row: called.append(row.sample_id),
        provenance={},
    )
    assert called == []
    assert result == []
    assert not (tmp_path / "checkpoint.jsonl").exists()


def test_execution_checkpoints_each_row_and_resume_skips_existing(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    calls = []

    def infer(row):
        calls.append(row.sample_id)
        return {"status": "completed", "model_output": "FINAL ANSWER: 7"}

    rows = [sample("a"), sample("b")]
    first = execute_samples(
        rows,
        path,
        model_id="Qwen/example",
        model_revision="a" * 40,
        run_enabled=True,
        infer_one=infer,
        provenance={"run_id": "r1"},
    )
    resumed = execute_samples(
        rows,
        path,
        model_id="Qwen/example",
        model_revision="a" * 40,
        run_enabled=True,
        infer_one=infer,
        provenance={"run_id": "r1"},
    )
    assert calls == ["a", "b"]
    assert [row["sample_id"] for row in resumed] == ["a", "b"]
    assert len(read_records(path)) == 2
    assert first[0]["provider"] == "local_transformers"
    assert first[0]["scientific_eligible"] is False


def test_failed_outcome_is_durable_and_not_automatically_retried(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    calls = []

    def fail(_row):
        calls.append(1)
        raise RuntimeError("synthetic failure")

    for _ in range(2):
        results = execute_samples(
            [sample("a")],
            path,
            model_id="Qwen/example",
            model_revision="b" * 40,
            run_enabled=True,
            infer_one=fail,
            provenance={},
        )
    assert calls == [1]
    assert results[0]["status"] == "failed"
    assert results[0]["error_message"] == "synthetic failure"


def test_resume_rejects_changed_model_or_dataset_provenance(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    execute_samples(
        [sample("a")],
        path,
        model_id="Qwen/example",
        model_revision="c" * 40,
        run_enabled=True,
        infer_one=lambda _: {"status": "completed", "model_output": "FINAL ANSWER: 7"},
        provenance={"collection_plan_sha256": "d" * 64},
    )
    with pytest.raises(ValueError, match="provenance"):
        execute_samples(
            [sample("a")],
            path,
            model_id="Qwen/example",
            model_revision="e" * 40,
            run_enabled=True,
            infer_one=lambda _: pytest.fail("changed model must not be invoked"),
            provenance={"collection_plan_sha256": "d" * 64},
        )
    with pytest.raises(ValueError, match="provenance"):
        execute_samples(
            [sample("a")],
            path,
            model_id="Qwen/example",
            model_revision="c" * 40,
            run_enabled=True,
            infer_one=lambda _: pytest.fail("changed plan must not be invoked"),
            provenance={"collection_plan_sha256": "f" * 64},
        )


def test_local_jsonl_rejects_provider_or_eligible_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    bad = record("a", correct=True)
    bad["provider"] = "litellm"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="separate from provider"):
        read_records(path)


def test_inference_callback_cannot_promote_a_local_record(tmp_path):
    output = execute_samples(
        [sample("a")],
        tmp_path / "checkpoint.jsonl",
        model_id="Qwen/example",
        model_revision="f" * 40,
        run_enabled=True,
        infer_one=lambda _: {
            "provider": "litellm",
            "scientific_eligible": True,
            "model_output": "FINAL ANSWER: 7",
        },
        provenance={},
    )
    assert output[0]["status"] == "failed"
    assert output[0]["provider"] == "local_transformers"
    assert output[0]["scientific_eligible"] is False
    assert output[0]["model_output"] is None


def test_evaluator_is_reused_only_for_successful_outputs():
    rows = [record("a", correct=None), record("b", correct=None, status="failed")]
    calls = []

    def evaluator(output, reference):
        calls.append((output, reference))
        return SimpleNamespace(model_dump=lambda **_: {"status": "correct", "is_correct": True})

    evaluated = evaluate_records(rows, evaluator)
    assert len(calls) == 1
    assert evaluated[0]["evaluation"]["is_correct"] is True
    assert evaluated[1]["evaluation"] is None
    assert evaluated[1]["status"] == "failed"


def test_paired_metrics_keep_unknowns_and_explicit_denominators(tmp_path):
    samples = [sample("a"), sample("b")]
    small = [record("a", correct=True), record("b", correct=None, status="failed")]
    strong = [record("a", correct=False)]
    pairs = paired_rows(samples, small, strong)
    assert [row["paired_outcome"] for row in pairs] == [
        "small_only_correct",
        "missing_generation",
    ]
    metrics = model_metrics(small)
    assert metrics["accuracy"] == 1.0
    assert metrics["accuracy_denominator"] == 1
    assert metrics["failed_samples"] == 1
    assert metrics["mean_inference_latency_s"] == 1.0
    path = tmp_path / "paired.csv"
    write_paired_csv(path, pairs)
    assert "missing_generation" in path.read_text()


def output_record(
    sample_id, correct=True, status="completed", model_id="Qwen/test", revision="a" * 40
):
    return {
        "sample_id": sample_id,
        "status": status,
        "model_id": model_id,
        "model_revision": revision,
        "model_output": "FINAL ANSWER: 7" if status == "completed" else None,
        "evaluation": (
            {"status": "correct" if correct else "incorrect", "is_correct": correct}
            if status == "completed" and correct is not None
            else None
        ),
    }


def test_pair_accounting_with_zero_inference():
    result = paired_metrics([sample("a"), sample("b")], [], [])
    assert result["planned_pair_count"] == 2
    assert result["completed_generation_pair_count"] == 0
    assert result["determinate_pair_count"] == 0
    assert result["unknown_pair_count"] == 0
    assert result["missing_pair_count"] == 2
    assert result["paired_outcome_counts"] == {"missing_generation": 2}


def test_zero_inference_exports_planned_pairs_without_claiming_completion(tmp_path):
    samples = [sample(f"s{i}") for i in range(10)]
    small_path = tmp_path / "small_results.jsonl"
    strong_path = tmp_path / "strong_results.jsonl"
    calls = []
    small = execute_samples(
        samples,
        small_path,
        model_id="Qwen/small",
        model_revision="1" * 40,
        run_enabled=False,
        infer_one=lambda row: calls.append(row.sample_id),
        provenance={},
    )
    strong = execute_samples(
        samples,
        strong_path,
        model_id="Qwen/strong",
        model_revision="2" * 40,
        run_enabled=False,
        infer_one=lambda row: calls.append(row.sample_id),
        provenance={},
    )
    metrics = paired_metrics(samples, small, strong)
    status = notebook_scientific_status(
        samples,
        [],
        [],
        [],
        [],
        small_model_id="Qwen/small",
        small_model_revision="1" * 40,
        strong_model_id="Qwen/strong",
        strong_model_revision="2" * 40,
        expected_provenance={},
        required_evidence=False,
    )
    write_paired_csv(tmp_path / "paired.csv", paired_rows(samples, small, strong))
    write_records(small_path, small)
    write_records(strong_path, strong)
    output_metrics = {**metrics, "scientific_status": status}
    (tmp_path / "metrics.json").write_text(json.dumps(output_metrics))
    assert calls == []
    assert metrics["planned_pair_count"] == 10
    assert metrics["completed_generation_pair_count"] == 0
    assert metrics["missing_pair_count"] == 10
    assert sum(metrics["paired_outcome_counts"].values()) == 10
    assert status == "NOTEBOOK_READY_FOR_KAGGLE"
    assert read_records(small_path) == []
    assert json.loads((tmp_path / "metrics.json").read_text())["scientific_status"] == (
        "NOTEBOOK_READY_FOR_KAGGLE"
    )


def test_pair_accounting_with_only_one_model_completed():
    result = paired_metrics([sample("a")], [output_record("a")], [])
    assert result["completed_generation_pair_count"] == 0
    assert result["missing_pair_count"] == 1
    assert result["paired_outcome_counts"] == {"missing_generation": 1}


def test_pair_accounting_with_both_models_completed():
    result = paired_metrics(
        [sample("a")],
        [output_record("a", correct=True)],
        [output_record("a", correct=False, model_id="Qwen/strong", revision="b" * 40)],
    )
    assert result["completed_generation_pair_count"] == 1
    assert result["determinate_pair_count"] == 1
    assert result["paired_outcome_counts"] == {"small_only_correct": 1}


def test_pair_accounting_with_one_failed_model():
    result = paired_metrics(
        [sample("a")],
        [output_record("a", correct=True)],
        [output_record("a", status="failed", model_id="Qwen/strong", revision="b" * 40)],
    )
    assert result["completed_generation_pair_count"] == 0
    assert result["missing_pair_count"] == 1
    assert result["paired_outcome_counts"] == {"missing_generation": 1}


def test_pair_accounting_with_unknown_evaluator_result():
    result = paired_metrics(
        [sample("a")],
        [output_record("a", correct=None)],
        [output_record("a", correct=True, model_id="Qwen/strong", revision="b" * 40)],
    )
    assert result["completed_generation_pair_count"] == 1
    assert result["determinate_pair_count"] == 0
    assert result["unknown_pair_count"] == 1
    assert result["missing_pair_count"] == 0
    assert result["paired_outcome_counts"] == {"unknown_evaluation": 1}


def test_pair_accounting_all_ten_planned_pairs_completed():
    samples = [sample(f"s{i}") for i in range(10)]
    small = [output_record(row.sample_id, correct=True) for row in samples]
    strong = [
        output_record(row.sample_id, correct=True, model_id="Qwen/strong", revision="b" * 40)
        for row in samples
    ]
    result = paired_metrics(samples, small, strong)
    assert result["planned_pair_count"] == 10
    assert result["completed_generation_pair_count"] == 10
    assert result["determinate_pair_count"] == 10
    assert result["unknown_pair_count"] == 0
    assert result["missing_pair_count"] == 0
    assert result["paired_outcome_counts"] == {"both_correct": 10}
    assert sum(result["paired_outcome_counts"].values()) == result["planned_pair_count"]


def test_throughput_uses_only_records_with_valid_latency_and_token_count():
    records = [
        {**record("a", correct=True), "inference_latency_s": 2.0, "output_tokens": 10},
        {**record("b", correct=True), "inference_latency_s": None, "output_tokens": 20},
        {**record("c", correct=True), "inference_latency_s": 4.0, "output_tokens": None},
        {**record("d", correct=True), "inference_latency_s": 0.0, "output_tokens": 30},
    ]
    metrics = model_metrics(records)
    assert metrics["throughput_sample_count"] == 1
    assert metrics["tokens_per_second"] == 5.0
    assert metrics["missing_latency_count"] == 2
    assert metrics["missing_token_count"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inference_latency_s", float("nan")),
        ("inference_latency_s", float("inf")),
        ("inference_latency_s", -0.1),
        ("output_tokens", float("nan")),
        ("output_tokens", float("inf")),
        ("output_tokens", -1),
        ("output_tokens", 1.5),
        ("peak_gpu_allocated_bytes", float("nan")),
        ("peak_gpu_reserved_bytes", -1),
    ],
)
def test_model_metrics_reject_nonfinite_or_negative_values(field, value):
    row = {**record("a", correct=True), "inference_latency_s": 1.0, "output_tokens": 1}
    row[field] = value
    with pytest.raises(ValueError, match=field):
        model_metrics([row])


def status_record(sample_id, model_id, revision, provenance, *, status="completed", evaluated=True):
    return {
        "schema_version": LOCAL_RECORD_SCHEMA,
        "provider": "local_transformers",
        "scientific_eligible": False,
        "sample_id": sample_id,
        "model_id": model_id,
        "model_revision": revision,
        "status": status,
        "model_output": "FINAL ANSWER: 7" if status == "completed" else None,
        "evaluation": {"status": "correct", "is_correct": True} if evaluated else None,
        **provenance,
    }


def status_fixture(sample_ids):
    samples = [sample(sample_id) for sample_id in sample_ids]
    expected = {
        "dataset_revision": "dataset-revision",
        "dataset_content_sha256": "d" * 64,
        "collection_plan_sha256": "e" * 64,
        "prompt_version": "math-direct-v1",
        "original_prompt_sha256": "f" * 64,
    }
    return samples, expected


def test_scientific_status_requires_actual_smoke_outputs():
    samples, expected = status_fixture(["a", "b"])
    small_id, strong_id = "Qwen/small", "Qwen/strong"
    small_rev, strong_rev = "1" * 40, "2" * 40
    small_smoke = [
        status_record(samples[0].sample_id, small_id, small_rev, {**expected, "run_mode": "smoke"})
    ]
    kwargs = {
        "small_model_id": small_id,
        "small_model_revision": small_rev,
        "strong_model_id": strong_id,
        "strong_model_revision": strong_rev,
        "expected_provenance": expected,
        "required_evidence": True,
    }
    assert notebook_scientific_status(samples, [], [], small_smoke, [], **kwargs) == (
        "NOTEBOOK_READY_FOR_KAGGLE"
    )
    strong_smoke = [
        status_record(
            samples[0].sample_id, strong_id, strong_rev, {**expected, "run_mode": "smoke"}
        )
    ]
    assert (
        notebook_scientific_status(samples, [], [], small_smoke, strong_smoke, **kwargs)
        == "KAGGLE_SMOKE_PASSED"
    )


def test_scientific_status_requires_all_exact_full_outputs_and_evaluations():
    samples, expected = status_fixture([f"s{i}" for i in range(10)])
    small_id, strong_id = "Qwen/small", "Qwen/strong"
    small_rev, strong_rev = "1" * 40, "2" * 40
    small = [
        status_record(row.sample_id, small_id, small_rev, {**expected, "run_mode": "full"})
        for row in samples
    ]
    strong = [
        status_record(row.sample_id, strong_id, strong_rev, {**expected, "run_mode": "full"})
        for row in samples
    ]
    kwargs = {
        "small_model_id": small_id,
        "small_model_revision": small_rev,
        "strong_model_id": strong_id,
        "strong_model_revision": strong_rev,
        "expected_provenance": expected,
        "required_evidence": True,
        "full_run_marker": {
            "schema_version": "kaggle-full-run-start-v1",
            "status": "started",
            "sample_ids": [row.sample_id for row in samples],
            "dataset_revision": expected["dataset_revision"],
            "dataset_content_sha256": expected["dataset_content_sha256"],
            "collection_plan_sha256": expected["collection_plan_sha256"],
            "prompt_version": expected["prompt_version"],
            "original_prompt_sha256": expected["original_prompt_sha256"],
            "small_model_id": small_id,
            "small_model_revision": small_rev,
            "strong_model_id": strong_id,
            "strong_model_revision": strong_rev,
        },
    }
    assert notebook_scientific_status(samples, small, strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_COMPLETED"
    )
    assert notebook_scientific_status(samples, [], [], [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )
    assert notebook_scientific_status(samples, small[:-1], strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )
    wrong_revision = [dict(row, model_revision="9" * 40) for row in small]
    assert notebook_scientific_status(samples, wrong_revision, strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )
    unknown_eval = [dict(row, evaluation=None) for row in small]
    assert notebook_scientific_status(samples, unknown_eval, strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )
    wrong_plan = [dict(row, collection_plan_sha256="9" * 64) for row in small]
    assert notebook_scientific_status(samples, wrong_plan, strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )
    failed = [
        *small[:-1],
        status_record("s9", small_id, small_rev, {**expected, "run_mode": "full"}, status="failed"),
    ]
    assert notebook_scientific_status(samples, failed, strong, [], [], **kwargs) == (
        "KAGGLE_PILOT_INCOMPLETE"
    )


def test_serialized_evaluated_outputs_are_jsonl(tmp_path):
    path = tmp_path / "small_results.jsonl"
    rows = [record("gsm8k-train-000279", correct=True)]
    write_records(path, rows)
    loaded = read_records(path)
    assert loaded[0]["evaluation"]["status"] == "correct"
    assert sample_ids_sha256(["x", "y"]) == sample_ids_sha256(["x", "y"])


def test_notebook_is_valid_json_and_has_sixteen_ordered_sections():
    notebook_path = Path(__file__).parents[1] / "experiments/notebooks/kaggle_t4_qwen3_gsm8k.ipynb"
    notebook = json.loads(notebook_path.read_text("utf-8"))
    headings = [
        "CELL 1 — Introduction",
        "CELL 2 — Environment setup",
        "CELL 3 — GPU and VRAM inspection",
        "CELL 4 — Reproducibility",
        "CELL 5 — Dataset loading",
        "CELL 6 — Frozen prompting protocol",
        "CELL 7 — Generation configuration",
        "CELL 8 — Small-model inference",
        "CELL 9 — GPU cleanup",
        "CELL 10 — Strong-model inference",
        "CELL 11 — Evaluate GSM8K correctness",
        "CELL 12 — Metrics",
        "CELL 13 — Visualization",
        "CELL 14 — Error analysis",
        "CELL 15 — Export outputs",
        "CELL 16 — Final summary",
    ]
    markdown = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    ]
    assert [
        heading for heading in headings if any(heading in cell for cell in markdown)
    ] == headings
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_FULL_PILOT = False" in source
    assert "official test" in source.lower()
    assert "enable_thinking=False" in source
    assert "do_sample=False" in source
    assert "paired_metrics(SAMPLES, small_evaluated, strong_evaluated)" in source
    assert "notebook_scientific_status(" in source
    assert '"scientific_status": SCIENTIFIC_STATUS' in source
    assert '"scientific_status": "NOTEBOOK_READY_FOR_KAGGLE"' not in source
