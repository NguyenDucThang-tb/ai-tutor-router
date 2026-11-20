from app.features import extract_features
from app.gateway import ModelGateway
from app.routing import Router


def test_complex_rule_selects_strong():
    f = extract_features("Prove that this identity is true, step by step", "math")
    assert Router("rule").decide(f).selected_strategy == "strong"


def test_simple_rule_selects_small():
    assert Router("rule").decide(extract_features("What is 2 + 2?", "math")).selected_strategy == "small"


def test_seeded_random_reproducible():
    a, b = Router("random", seed=17), Router("random", seed=17)
    f = extract_features("Solve 3x=9")
    assert [a.decide(f).selected_strategy for _ in range(8)] == [b.decide(f).selected_strategy for _ in range(8)]


def test_mock_gateway_is_explicitly_mock():
    result = ModelGateway().complete("small", "Question:\n2+2")
    assert result.model == "mock-small"
    assert result.cost_usd == 0.0
    assert "MOCK" in result.text
