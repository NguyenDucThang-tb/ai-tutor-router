import random

import pytest

from app.features import extract_features
from app.routing import RouterConfig, build_router


@pytest.fixture
def config():
    return RouterConfig("tier-small", "tier-strong", random_seed="seed-A")


def test_always_small_and_strong_ignore_features(config):
    features = extract_features("Prove a long theorem step by step", "verification")
    assert (
        build_router("always_small", config).decide(features, "r1").selected_model_id
        == "tier-small"
    )
    assert (
        build_router("always_strong", config).decide(features, "r1").selected_model_id
        == "tier-strong"
    )


def test_random_replays_by_request_id_without_shared_rng(config):
    global_rng = random.getstate()
    router = build_router("random", config)
    features = extract_features("solve x = 1")
    first = router.decide(features, "request-42")
    for _ in range(10):
        router.decide(features, f"other-{_}")
    replay = router.decide(features, "request-42")
    assert first.selected_model_id == replay.selected_model_id
    assert first.decision_metadata == replay.decision_metadata
    assert random.getstate() == global_rng


@pytest.mark.parametrize(("probability", "expected"), [(0.0, "tier-strong"), (1.0, "tier-small")])
def test_random_probability_boundaries(config, probability, expected):
    config = RouterConfig(**{**config.__dict__, "random_small_probability": probability})
    assert (
        build_router("random", config).decide(extract_features("x"), "r").selected_model_id
        == expected
    )


def test_random_rejects_invalid_probability(config):
    config = RouterConfig(**{**config.__dict__, "random_small_probability": 1.1})
    with pytest.raises(ValueError):
        build_router("random", config)


@pytest.mark.parametrize(
    "question,task,tier",
    [
        ("2 + 2?", "math", "small"),
        ("x" * 161, "math", "strong"),
        ("Check my solution: x=4", "verification", "strong"),
        ("Prove that x equals 2", "math", "strong"),
        ("Give a step-by-step solution", "math", "strong"),
        ("A train leaves at noon; how far after 3 hours?", "word_problem", "strong"),
        ("x+y=z and a+b=c", "math", "strong"),
    ],
)
def test_rule_router_cases(config, question, task, tier):
    decision = build_router("rule", config).decide(extract_features(question, task), "case")
    assert decision.selected_tier == tier
    assert decision.selected_model_id == ("tier-small" if tier == "small" else "tier-strong")
    assert decision.reason_code in {"RULE_COMPLEXITY_CUE", "RULE_SHORT_DIRECT_QUERY"}


def test_rule_length_boundary_is_inclusive(config):
    question = "x" * config.max_short_query_characters
    assert (
        build_router("rule", config).decide(extract_features(question), "a").selected_tier
        == "small"
    )
    question += "x"
    assert (
        build_router("rule", config).decide(extract_features(question), "b").selected_tier
        == "strong"
    )


def test_decision_has_no_quality_probability(config):
    decision = build_router("rule", config).decide(extract_features("2+2"), "id-1")
    assert "quality_estimate" not in decision.model_dump()
    assert decision.decision_metadata["request_id"] == "id-1"
    assert decision.router_latency_ms >= 0


def test_invalid_router_name_rejected(config):
    with pytest.raises(ValueError, match="Unsupported router"):
        build_router("quality_aware", config)


def test_explicit_step_count_feature():
    f = extract_features("Explain this in 3 steps", "math")
    assert f.requested_steps is True
    assert f.requested_step_count == 3
