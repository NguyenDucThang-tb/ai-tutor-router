"""Research candidates with source-backed identity and prices, never availability claims."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class CandidateModel:
    provider: str
    model_id: str
    tier: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    pricing_url: str
    identity_url: str
    retrieved: date
    pricing_effective_date: str | None = None
    maximum_output_tokens: int | None = None
    context_tokens: int | None = None
    notes: str = ""
    status: str = "documented_current"


# Prices are USD per 1M uncached text tokens, checked against official pages 2026-11-22.
# Missing limits and account access are explicitly not inferred from a pricing page.
CANDIDATES = {
    "openai_gpt41_mini": CandidateModel(
        "openai",
        "openai/gpt-4.1-mini-2025-04-14",
        "small",
        Decimal("0.40"),
        Decimal("1.60"),
        "https://developers.openai.com/api/docs/pricing",
        "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
        date(2026, 11, 22),
        notes="Official model page marks this dated ID deprecated; not eligible for new pilot.",
        status="deprecated",
    ),
    "openai_gpt41": CandidateModel(
        "openai",
        "openai/gpt-4.1-2025-04-14",
        "strong",
        Decimal("2.00"),
        Decimal("8.00"),
        "https://developers.openai.com/api/docs/pricing",
        "https://developers.openai.com/api/docs/models/gpt-4.1",
        date(2026, 11, 22),
        notes="Official model page marks this dated ID deprecated; not eligible for new pilot.",
        status="deprecated",
    ),
    "anthropic_haiku55": CandidateModel(
        "anthropic",
        "anthropic/claude-haiku-5-5",
        "small",
        Decimal("0.10"),
        Decimal("0.50"),
        "https://platform.claude.com/docs/en/about-claude/pricing",
        "https://platform.claude.com/docs/en/about-claude/model-deprecations",
        date(2026, 11, 22),
        notes="Base pricing is for prompts up to 100k tokens; higher tier pricing applies above.",
    ),
    "anthropic_sonnet55": CandidateModel(
        "anthropic",
        "anthropic/claude-sonnet-5-5",
        "strong",
        Decimal("2.00"),
        Decimal("10.00"),
        "https://platform.claude.com/docs/en/about-claude/pricing",
        "https://platform.claude.com/docs/en/about-claude/model-deprecations",
        date(2026, 11, 22),
        notes="Base uncached text-token rates; account and region availability unverified.",
    ),
    "gemini_flash_lite": CandidateModel(
        "gemini",
        "gemini/gemini-3.5-flash-lite",
        "small",
        Decimal("0.30"),
        Decimal("2.50"),
        "https://ai.google.dev/gemini-api/docs/pricing",
        "https://ai.google.dev/gemini-api/docs/models",
        date(2026, 11, 22),
        notes="Paid standard tier through 2026-12-31; input/output thinking; free-tier data-use excluded.",
    ),
    "gemini_flash": CandidateModel(
        "gemini",
        "gemini/gemini-3.8-flash",
        "strong",
        Decimal("0.75"),
        Decimal("3.75"),
        "https://ai.google.dev/gemini-api/docs/pricing",
        "https://ai.google.dev/gemini-api/docs/models",
        date(2026, 11, 22),
        notes="Paid standard tier through 2026-12-31; thinking tokens billed at output rate.",
    ),
}

PAIRS = {
    "openai_gpt41": (CANDIDATES["openai_gpt41_mini"], CANDIDATES["openai_gpt41"]),
    "anthropic_55": (CANDIDATES["anthropic_haiku55"], CANDIDATES["anthropic_sonnet55"]),
    "gemini_35": (CANDIDATES["gemini_flash_lite"], CANDIDATES["gemini_flash"]),
}


def resolve_model(model_id: str) -> CandidateModel | None:
    return next((model for model in CANDIDATES.values() if model.model_id == model_id), None)
