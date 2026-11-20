from typing import Literal

from pydantic import BaseModel, Field


class TutorRequest(BaseModel):
    question: str = Field(min_length=1, max_length=12000)
    task_type: Literal["math", "algebra", "word_problem", "verification", "guided"] = "math"
    mode: Literal["auto", "direct", "guided", "verify"] = "auto"
    student_level: Literal["middle_school", "high_school", "undergraduate", "unspecified"] = "unspecified"


class RouteDecision(BaseModel):
    selected_strategy: Literal["small", "strong"]
    router_name: str
    quality_estimate: float | None = None
    estimated_cost_usd: float | None = None
    estimated_latency_ms: float | None = None
    reason_code: str


class TutorResponse(BaseModel):
    request_id: str
    answer: str
    route: RouteDecision
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    latency_ms: float
    prompt_version: str = "tutor-v1"
