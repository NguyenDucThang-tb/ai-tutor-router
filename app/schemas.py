from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.gateway import CostInfo, ExecutionAttempt, UsageInfo
from app.routing import RouteDecision


class TutorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=12000)
    task_type: Literal["math", "algebra", "word_problem", "verification", "guided"] = "math"
    mode: Literal["auto", "direct", "guided", "verify"] = "auto"
    student_level: Literal["middle_school", "high_school", "undergraduate", "unspecified"] = (
        "unspecified"
    )


class TutorResponse(BaseModel):
    request_id: str
    answer: str
    selected_tier: Literal["small", "strong"]
    selected_model_id: str
    executed_model_id: str
    router: RouteDecision
    provider: Literal["mock", "litellm"]
    router_latency_ms: float
    gateway_latency_ms: float
    total_latency_ms: float
    usage: UsageInfo | None
    cost: CostInfo
    attempts: list[ExecutionAttempt]
    prompt_version: str = "tutor-v1"


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    attempts: list[ExecutionAttempt] | None = None
