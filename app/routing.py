"""Deterministic routing baselines, independent of the API and provider gateway."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.features import QueryFeatures

Tier = Literal["small", "strong"]
RouterName = Literal["always_small", "always_strong", "random", "rule"]


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    router_strategy: RouterName
    selected_tier: Tier
    selected_model_id: str
    reason_code: str
    decision_metadata: dict[str, Any] = Field(default_factory=dict)
    router_latency_ms: float


@dataclass(frozen=True)
class RouterConfig:
    small_model_id: str
    strong_model_id: str
    random_small_probability: float = 0.5
    random_seed: str = "42"
    max_short_query_characters: int = 160
    max_short_query_tokens: int = 30
    max_math_symbols_for_small: int = 3


class BaseRouter(ABC):
    name: RouterName

    def __init__(self, config: RouterConfig):
        self.config = config

    @abstractmethod
    def decide(self, features: QueryFeatures, request_id: str) -> RouteDecision:
        """Choose a model deterministically for the supplied request ID."""

    def _decision(
        self, tier: Tier, request_id: str, reason: str, metadata: dict[str, Any], started: float
    ) -> RouteDecision:
        model_id = self.config.small_model_id if tier == "small" else self.config.strong_model_id
        return RouteDecision(
            router_strategy=self.name,
            selected_tier=tier,
            selected_model_id=model_id,
            reason_code=reason,
            decision_metadata={**metadata, "request_id": request_id},
            router_latency_ms=(time.perf_counter() - started) * 1000,
        )


class AlwaysSmallRouter(BaseRouter):
    name: RouterName = "always_small"

    def decide(self, features: QueryFeatures, request_id: str) -> RouteDecision:
        started = time.perf_counter()
        return self._decision("small", request_id, "ALWAYS_SMALL", {}, started)


class AlwaysStrongRouter(BaseRouter):
    name: RouterName = "always_strong"

    def decide(self, features: QueryFeatures, request_id: str) -> RouteDecision:
        started = time.perf_counter()
        return self._decision("strong", request_id, "ALWAYS_STRONG", {}, started)


class RandomRouter(BaseRouter):
    name: RouterName = "random"

    def __init__(self, config: RouterConfig):
        super().__init__(config)
        if not 0.0 <= config.random_small_probability <= 1.0:
            raise ValueError("random_small_probability must be between 0 and 1")

    def decide(self, features: QueryFeatures, request_id: str) -> RouteDecision:
        started = time.perf_counter()
        material = f"{self.config.random_seed}\0{request_id}".encode()
        value = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") / 2**64
        tier: Tier = "small" if value < self.config.random_small_probability else "strong"
        return self._decision(
            tier,
            request_id,
            "SEEDED_REQUEST_HASH",
            {
                "small_probability": self.config.random_small_probability,
                "seed": self.config.random_seed,
                "draw": value,
            },
            started,
        )


class RuleBasedRouter(BaseRouter):
    name: RouterName = "rule"

    def decide(self, features: QueryFeatures, request_id: str) -> RouteDecision:
        started = time.perf_counter()
        cues = {
            "long_query": features.char_count > self.config.max_short_query_characters
            or features.token_count > self.config.max_short_query_tokens,
            "proof_request": features.has_proof_request,
            "solution_verification": features.has_verification_intent,
            "many_math_symbols": features.math_symbol_count
            > self.config.max_math_symbols_for_small,
            "multi_step_request": features.requested_steps,
            "advanced_task_type": features.task_type in {"word_problem", "guided"},
        }
        # Transparent conservative rules. This is a baseline and emits no quality estimate.
        complex_query = any(cues.values())
        tier: Tier = "strong" if complex_query else "small"
        reason = "RULE_COMPLEXITY_CUE" if complex_query else "RULE_SHORT_DIRECT_QUERY"
        return self._decision(tier, request_id, reason, {"matched_cues": cues}, started)


ROUTERS: dict[str, type[BaseRouter]] = {
    "always_small": AlwaysSmallRouter,
    "always_strong": AlwaysStrongRouter,
    "random": RandomRouter,
    "rule": RuleBasedRouter,
}


def build_router(name: str, config: RouterConfig) -> BaseRouter:
    try:
        router_type = ROUTERS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported router strategy: {name}") from exc
    return router_type(config)
