from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "real", "replay"] = "mock"
    provider: Literal["mock", "litellm"] = "mock"
    small_model_identifier: str = Field(min_length=1)
    strong_model_identifier: str = Field(min_length=1)
    max_samples: int = Field(default=2, ge=1)
    max_requests: int = Field(default=4, ge=2)
    concurrency: int = Field(default=2, ge=1, le=128)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    execution_lease_seconds: float = Field(default=900, gt=0, le=7200)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    seed: int = 42
    max_retries: int = Field(default=0, ge=0, le=3)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=5)
    fallback_policy: Literal["disabled", "operational"] = "disabled"
    small_fallback_identifier: str | None = None
    strong_fallback_identifier: str | None = None
    per_request_budget_usd: float | None = Field(default=None, ge=0)
    provider_region: str | None = None
    pricing_source: str | None = None
    pricing_effective_date: str | None = None
    allow_real_provider: bool = False
    experiment_approval_id: str | None = None
    allow_paid_retries: bool = False
    paid_retry_approval_id: str | None = None
    retry_interrupted: bool = False
    allow_uncertain_reissue: bool = False
    uncertain_reissue_approval_id: str | None = None
    retry_failed: bool = False
    rerun_completed: bool = False
    router_strategy: Literal["always_small", "always_strong", "random", "rule"] = "rule"

    @model_validator(mode="after")
    def validate_mode_and_models(self):
        if self.small_model_identifier == self.strong_model_identifier:
            raise ValueError("small and strong model identifiers must differ")
        if self.mode == "mock" and self.provider != "mock":
            raise ValueError("mock execution mode requires the mock provider")
        if self.mode == "real":
            if self.provider != "litellm":
                raise ValueError("real execution mode requires provider=litellm")
            if not self.allow_real_provider or not self.experiment_approval_id:
                raise ValueError(
                    "real mode requires allow_real_provider and experiment_approval_id"
                )
            if self.allow_uncertain_reissue and not self.uncertain_reissue_approval_id:
                raise ValueError("uncertain paid reissue requires uncertain_reissue_approval_id")
            if (self.max_retries > 0 or self.retry_failed or self.rerun_completed) and (
                not self.allow_paid_retries or not self.paid_retry_approval_id
            ):
                raise ValueError(
                    "real paid retries/reruns require allow_paid_retries and paid_retry_approval_id"
                )
        elif self.uncertain_reissue_approval_id is not None:
            raise ValueError("uncertain reissue approval is only valid in real mode")
        if self.mode != "real" and self.paid_retry_approval_id is not None:
            raise ValueError("paid retry approval is only valid in real mode")
        if self.fallback_policy == "disabled" and (
            self.small_fallback_identifier or self.strong_fallback_identifier
        ):
            raise ValueError("fallback identifiers require fallback_policy=operational")
        if self.fallback_policy == "operational":
            for primary, fallback in (
                (self.small_model_identifier, self.small_fallback_identifier),
                (self.strong_model_identifier, self.strong_fallback_identifier),
            ):
                if fallback and fallback == primary:
                    raise ValueError("fallback model must differ from its primary model")
        if self.per_request_budget_usd is not None and self.mode != "real":
            raise ValueError("per-request provider budget is only valid in real mode")
        return self


def load_config(path: Path) -> BenchmarkConfig:
    data = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("benchmark config must be a YAML mapping")
    return BenchmarkConfig.model_validate(data)
