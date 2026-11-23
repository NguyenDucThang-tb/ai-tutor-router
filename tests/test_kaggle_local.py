import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.kaggle_local import (
    LOCAL_RECORD_SCHEMA,
    evaluate_records,
    execute_samples,
    model_metrics,
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
        "at_least_one_unknown",
    ]
    metrics = model_metrics(small)
    assert metrics["accuracy"] == 1.0
    assert metrics["accuracy_denominator"] == 1
    assert metrics["failed_samples"] == 1
    assert metrics["mean_inference_latency_s"] == 1.0
    path = tmp_path / "paired.csv"
    write_paired_csv(path, pairs)
    assert "at_least_one_unknown" in path.read_text()


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
