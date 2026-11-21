"""Decimal price projections. These estimates are not provider billing guarantees."""

from decimal import Decimal

from experiments.model_catalog import resolve_model


def estimate_usd(model_id: str, input_tokens: int, output_tokens: int) -> Decimal | None:
    model = resolve_model(model_id)
    if model is None or input_tokens < 0 or output_tokens < 0:
        return None
    return (
        Decimal(input_tokens) * model.input_usd_per_million
        + Decimal(output_tokens) * model.output_usd_per_million
    ) / Decimal(1_000_000)


def hard_budget_bound_supported(model_id: str) -> bool:
    """No provider presently has verified local token counting plus hard billing caps."""
    return False
