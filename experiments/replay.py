"""Offline replay retaining failures and validating available provenance."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from datasets.schemas import DatasetRecord
from evaluation.evaluator import EVALUATOR_VERSION, evaluate_answer
from experiments.records import ExperimentRecord


def replay_records(
    source_path: Path, records: list[DatasetRecord], run_id: str
) -> list[ExperimentRecord]:
    expected = {record.sample_id: record for record in records}
    output: list[ExperimentRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(source_path.read_text("utf-8").splitlines(), 1):
        previous = ExperimentRecord.model_validate_json(line)
        key = (previous.sample_id, previous.split, previous.model_tier)
        if key in seen:
            raise ValueError(f"duplicate replay execution at line {line_number}: {key}")
        seen.add(key)
        sample = expected.get(previous.sample_id)
        if sample is None:
            raise ValueError(f"replay record at line {line_number} is absent from dataset")
        if previous.split != sample.assigned_split:
            raise ValueError(f"replay split mismatch for {sample.sample_id}")
        if previous.provenance.get("source_hash") != sample.source_hash:
            raise ValueError(f"replay dataset source hash mismatch for {sample.sample_id}")
        if previous.provenance.get("dataset_revision") != sample.dataset_revision:
            raise ValueError(f"replay dataset revision mismatch for {sample.sample_id}")

        provenance = dict(previous.provenance)
        reasons = _execution_provenance_gaps(previous)
        provenance.update(
            {
                "replayed_from_run_id": previous.run_id,
                "replayed_from_execution_mode": previous.execution_mode,
                "replay_provenance_status": "verified" if not reasons else "incomplete",
                "replay_provenance_gaps": ";".join(reasons) if reasons else None,
            }
        )
        evaluation = previous.evaluation
        evaluation_complete = previous.evaluation_complete
        if previous.execution_status == "success" and previous.model_output is not None:
            evaluation = evaluate_answer(previous.model_output, sample.reference_answer)
            evaluation_complete = True
        else:
            evaluation = None
            evaluation_complete = False
        output.append(
            previous.model_copy(
                update={
                    "run_id": run_id,
                    "execution_mode": "replay",
                    "evaluation": evaluation,
                    "evaluation_complete": evaluation_complete,
                    "scientific_eligible": False,
                    "provenance": provenance,
                    "evaluator_version": EVALUATOR_VERSION,
                }
            )
        )
    return output


def _execution_provenance_gaps(record: ExperimentRecord) -> list[str]:
    gaps = []
    if not record.provider or not record.requested_model_identifier:
        gaps.append("provider_or_requested_model_missing")
    if not record.prompt_version or not re.fullmatch(r"[0-9a-f]{64}", record.prompt_sha256):
        gaps.append("prompt_provenance_missing")
    if not record.provenance.get("commit_sha"):
        gaps.append("commit_sha_missing")
    if record.execution_status == "success" and record.model_output is not None:
        output_hash = record.provenance.get("output_sha256")
        actual = hashlib.sha256(record.model_output.encode()).hexdigest()
        if output_hash is not None and output_hash != actual:
            raise ValueError(f"replay model output hash mismatch for {record.sample_id}")
        if output_hash is None:
            gaps.append("output_hash_missing")
        if not record.attempts:
            gaps.append("attempt_provenance_missing")
        elif record.executed_model_identifier not in {
            attempt.model_identifier for attempt in record.attempts
        }:
            gaps.append("executed_model_not_in_attempts")
        elif not any(
            attempt.status == "success"
            and attempt.request_id == record.provenance.get("request_id")
            and attempt.model_identifier == record.executed_model_identifier
            and attempt.provider == record.provider
            for attempt in record.attempts
        ):
            gaps.append("successful_attempt_provenance_mismatch")
    elif (
        record.execution_status in {"provider_failure", "outcome_unknown"}
        and not record.attempts
        and not record.provenance.get("request_id")
    ):
        gaps.append("failed_execution_request_provenance_missing")
    if record.execution_status == "not_started":
        gaps.append("execution_not_started")
    if record.execution_status == "interrupted" and not record.attempts:
        gaps.append("interrupted_attempt_provenance_missing")
    if record.execution_status in {"provider_failure", "outcome_unknown"}:
        request_id = record.provenance.get("request_id")
        if (
            request_id
            and record.attempts
            and not any(attempt.request_id == request_id for attempt in record.attempts)
        ):
            gaps.append("failed_attempt_request_id_mismatch")
        if record.attempts and any(
            attempt.provider not in {record.provider, "unknown"} for attempt in record.attempts
        ):
            gaps.append("failed_attempt_provider_mismatch")
    return gaps


def write_replay(path: Path, records: list[ExperimentRecord]) -> None:
    content = "".join(record.model_dump_json() + "\n" for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
