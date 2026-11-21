"""Human review queue and adjudication validation for ambiguous model answers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_tier: Literal["small", "strong"]
    label: Literal["correct", "incorrect", "unparseable", "unsupported", "ambiguous"]
    rationale: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    reviewed_at_utc: str = Field(min_length=1)
    reviewed: bool = False
    agreement_with_second_reviewer: bool | None = None
    adjudication_id: str | None = None


class TutoringRubric(BaseModel):
    """Human annotation fields; no automated judge is wired to this schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    model_tier: Literal["small", "strong"]
    mathematical_clarity: int | None = Field(default=None, ge=1, le=5)
    logical_organization: int | None = Field(default=None, ge=1, le=5)
    educational_usefulness: int | None = Field(default=None, ge=1, le=5)
    age_appropriateness: int | None = Field(default=None, ge=1, le=5)
    student_mistake_handling: int | None = Field(default=None, ge=1, le=5)
    reviewer_id: str = Field(min_length=1)
    reviewed_at_utc: str = Field(min_length=1)
    free_text_rationale: str = ""
    agreement_with_second_reviewer: bool | None = None
    adjudication_id: str | None = None


def review_queue(records) -> list[dict]:
    """Return unlabeled candidate rows; automated output never marks human review."""
    rows = []
    for record in records:
        if record.execution_status != "success" or record.evaluation is None:
            continue
        if record.evaluation.status in {"unsupported", "ambiguous", "unparseable"}:
            rows.append(
                {
                    "run_id": record.run_id,
                    "sample_id": record.sample_id,
                    "model_tier": record.model_tier,
                    "model_output": record.model_output,
                    "reference_answer": record.evaluation.reference_answer,
                    "automatic_status": record.evaluation.status,
                    "reviewed": False,
                }
            )
    return rows


def correctness_sufficiency_label(record, *, frozen_criterion: str) -> int | None:
    """Produce a train/validation-only correctness label; mock/replay are ineligible."""
    if record.split == "test":
        raise ValueError("held-out test records cannot be used to prepare training labels")
    if record.execution_mode != "real":
        return None
    if frozen_criterion != "correctness-based-sufficiency-v1":
        raise ValueError("unknown or unfrozen sufficiency criterion")
    if record.model_tier != "small" or record.execution_status != "success":
        return None
    if record.evaluation is None or record.evaluation.is_correct is None:
        return None
    return int(record.evaluation.is_correct)
