from dataclasses import dataclass
import random

from app.features import QueryFeatures
from app.schemas import RouteDecision


@dataclass(frozen=True)
class RoutingConfig:
    quality_threshold: float = 0.80
    cost_lambda: float = 0.10
    latency_lambda: float = 0.10
    small_cost_usd: float = 0.0002
    strong_cost_usd: float = 0.002
    small_latency_ms: float = 900
    strong_latency_ms: float = 1800


class Router:
    def __init__(self, name: str = "rule", config: RoutingConfig = RoutingConfig(), seed: int = 42):
        self.name, self.config, self.rng = name, config, random.Random(seed)

    def decide(self, f: QueryFeatures) -> RouteDecision:
        c = self.config
        if self.name == "always_small":
            return self._decision("small", 1.0, "ALWAYS_SMALL")
        if self.name == "always_strong":
            return self._decision("strong", 1.0, "ALWAYS_STRONG")
        if self.name == "random":
            tier = "small" if self.rng.random() < 0.5 else "strong"
            return self._decision(tier, 0.5, "SEEDED_RANDOM")
        # An interpretable rule baseline; complexity cues are combined, not keyword-only.
        difficult = f.has_proof_request or f.has_verification_intent or f.token_count > 45 or f.char_count > 280
        qhat = 0.55 if difficult else 0.88
        if self.name in {"quality_aware", "ml"}:
            # Prior-only fallback until a trained estimator artifact is supplied.
            # Never present this as a learned prediction.
            qhat = 0.52 if difficult else 0.84
        quality_ok = qhat >= c.quality_threshold
        # Explicitly scaled unit costs; these lambdas are configurable policy weights, not learned.
        small_utility = qhat - c.cost_lambda * c.small_cost_usd - c.latency_lambda * c.small_latency_ms / 10000
        strong_utility = 1.0 - c.cost_lambda * c.strong_cost_usd - c.latency_lambda * c.strong_latency_ms / 10000
        choose_small = quality_ok and (self.name != "utility" or small_utility >= strong_utility)
        tier = "small" if choose_small else "strong"
        reason = "SMALL_MODEL_EXPECTED_SUFFICIENT" if choose_small else "QUALITY_CONSTRAINT_OR_COMPLEXITY"
        return self._decision(tier, qhat, reason)

    def _decision(self, tier: str, q: float | None, reason: str) -> RouteDecision:
        c = self.config
        return RouteDecision(selected_strategy=tier, router_name=self.name, quality_estimate=q,
            estimated_cost_usd=c.small_cost_usd if tier == "small" else c.strong_cost_usd,
            estimated_latency_ms=c.small_latency_ms if tier == "small" else c.strong_latency_ms,
            reason_code=reason)
