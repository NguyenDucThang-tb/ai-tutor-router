from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class EvaluationStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSEABLE = "unparseable"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_version: str = "numeric-symbolic-v2"
    status: EvaluationStatus
    is_correct: bool | None
    extracted_answer: str | None = None
    reference_answer: str | None = None
    reason: str | None = None
