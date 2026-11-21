"""Offline replay of prior successful outputs against the current dataset reference."""

from __future__ import annotations

from pathlib import Path

from datasets.schemas import DatasetRecord
from evaluation.evaluator import evaluate_answer
from experiments.records import ExperimentRecord


def replay_records(
    source_path: Path, records: list[DatasetRecord], run_id: str
) -> list[ExperimentRecord]:
    expected = {record.sample_id: record for record in records}
    output = []
    for line_number, line in enumerate(source_path.read_text("utf-8").splitlines(), 1):
        previous = ExperimentRecord.model_validate_json(line)
        sample = expected.get(previous.sample_id)
        if sample is None:
            raise ValueError(f"replay record at line {line_number} is absent from dataset")
        if previous.provenance.get("source_hash") != sample.source_hash:
            raise ValueError(f"replay dataset source hash mismatch for {sample.sample_id}")
        if previous.execution_status != "success" or previous.model_output is None:
            continue
        output.append(
            previous.model_copy(
                update={
                    "run_id": run_id,
                    "execution_mode": "replay",
                    "evaluation": evaluate_answer(previous.model_output, sample.reference_answer),
                }
            )
        )
    return output


def write_replay(path: Path, records: list[ExperimentRecord]) -> None:
    content = "".join(record.model_dump_json() + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
