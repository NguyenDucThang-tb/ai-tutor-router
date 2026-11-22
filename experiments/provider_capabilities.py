"""Versioned, conservative provider capability inventory.

Public documentation can verify model metadata, but not the user's account access,
project configuration, effective price tier, or exclusive billing scope. Accordingly,
no shipped entry currently authorizes paid execution.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal


@dataclass(frozen=True)
class ProviderCapability:
    provider: str
    model_id: str
    identity_url: str
    pricing_url: str
    verified_on: date
    input_token_method: str | None
    output_token_bound: int | None
    billing_control: str
    pricing_version: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal | None
    max_price_age_days: int = 7
    all_billable_categories_supported: bool = False
    account_access: str = "unverified"
    supports_provable_run_bound: bool = False
    status: str = "documented_current"
    limitation: str = "Account scope and effective billing controls are not verified."


VERIFICATION_DATE = date(2026, 11, 22)
GOOGLE_PRICE = "https://ai.google.dev/gemini-api/docs/pricing"
GOOGLE_BILLING = "https://ai.google.dev/gemini-api/docs/billing"

CAPABILITIES = {
    "openai/gpt-6-luna": ProviderCapability(
        provider="openai",
        model_id="openai/gpt-6-luna",
        identity_url="https://developers.openai.com/api/docs/models/gpt-6-luna",
        pricing_url="https://developers.openai.com/api/docs/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method=None,
        output_token_bound=128000,
        billing_control="https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform",
        pricing_version="openai-gpt-6-luna-2026-11-22-standard",
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.50"),
        cached_input_usd_per_million=Decimal("0.01"),
        limitation=(
            "Exact locked LiteLLM model mapping, reasoning/cache/tool categories, account access, "
            "and external bill bound are not verified."
        ),
    ),
    "openai/gpt-6.2-sol": ProviderCapability(
        provider="openai",
        model_id="openai/gpt-6.2-sol",
        identity_url="https://developers.openai.com/api/docs/models/gpt-6.2-sol",
        pricing_url="https://developers.openai.com/api/docs/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method=None,
        output_token_bound=128000,
        billing_control="https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform",
        pricing_version="openai-gpt-6.2-sol-2026-11-22-standard",
        input_usd_per_million=Decimal("2.00"),
        output_usd_per_million=Decimal("10.00"),
        cached_input_usd_per_million=Decimal("0.10"),
        limitation=(
            "Exact locked LiteLLM model mapping, reasoning/cache/tool categories, account access, "
            "and external bill bound are not verified."
        ),
    ),
    "openai/gpt-4.1-mini-2025-04-14": ProviderCapability(
        provider="openai",
        model_id="openai/gpt-4.1-mini-2025-04-14",
        identity_url="https://developers.openai.com/api/docs/models/gpt-4.1-mini",
        pricing_url="https://developers.openai.com/api/docs/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method=None,
        output_token_bound=32768,
        billing_control="https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform",
        pricing_version="openai-gpt-4.1-mini-2026-11-22-deprecated",
        input_usd_per_million=Decimal("0.40"),
        output_usd_per_million=Decimal("1.60"),
        cached_input_usd_per_million=Decimal("0.10"),
        status="deprecated",
        limitation="The dated model is deprecated; hard spend enforcement may overshoot.",
    ),
    "openai/gpt-4.1-2025-04-14": ProviderCapability(
        provider="openai",
        model_id="openai/gpt-4.1-2025-04-14",
        identity_url="https://developers.openai.com/api/docs/models/gpt-4.1",
        pricing_url="https://developers.openai.com/api/docs/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method=None,
        output_token_bound=32768,
        billing_control="https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform",
        pricing_version="openai-gpt-4.1-2026-11-22-deprecated",
        input_usd_per_million=Decimal("2.00"),
        output_usd_per_million=Decimal("8.00"),
        cached_input_usd_per_million=Decimal("0.50"),
        status="deprecated",
        limitation="The dated model is deprecated; hard spend enforcement may overshoot.",
    ),
    "anthropic/claude-haiku-5-5": ProviderCapability(
        provider="anthropic",
        model_id="anthropic/claude-haiku-5-5",
        identity_url="https://platform.claude.com/docs/en/models/overview",
        pricing_url="https://platform.claude.com/docs/en/about-claude/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method="Anthropic count_tokens endpoint (input estimate)",
        output_token_bound=131072,
        billing_control="https://platform.claude.com/docs/en/api/rate-limits",
        pricing_version="anthropic-claude-haiku-5-5-2026-11-22-tiered",
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.50"),
        cached_input_usd_per_million=Decimal("0.01"),
        limitation="Tiered input/cache prices and account spend scope are not verified.",
    ),
    "anthropic/claude-sonnet-5-5": ProviderCapability(
        provider="anthropic",
        model_id="anthropic/claude-sonnet-5-5",
        identity_url="https://platform.claude.com/docs/en/models/overview",
        pricing_url="https://platform.claude.com/docs/en/about-claude/pricing",
        verified_on=VERIFICATION_DATE,
        input_token_method="Anthropic count_tokens endpoint (input estimate)",
        output_token_bound=131072,
        billing_control="https://platform.claude.com/docs/en/api/rate-limits",
        pricing_version="anthropic-claude-sonnet-5-5-2026-11-22",
        input_usd_per_million=Decimal("2.00"),
        output_usd_per_million=Decimal("10.00"),
        cached_input_usd_per_million=Decimal("0.10"),
        limitation="Cache writes use separate rates and account spend scope is not verified.",
    ),
    "gemini/gemini-3.5-flash-lite": ProviderCapability(
        provider="gemini",
        model_id="gemini/gemini-3.5-flash-lite",
        identity_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite",
        pricing_url=GOOGLE_PRICE,
        verified_on=VERIFICATION_DATE,
        input_token_method="Gemini countTokens endpoint (input estimate; not a bill cap)",
        output_token_bound=65536,
        billing_control=GOOGLE_BILLING,
        pricing_version="gemini-standard-2026-11-22",
        input_usd_per_million=Decimal("0.30"),
        output_usd_per_million=Decimal("2.50"),
        cached_input_usd_per_million=Decimal("0.03"),
        limitation=(
            "Project spend caps are experimental and may overrun during billing-data delay; "
            "account tier caps are shared across linked projects."
        ),
    ),
    "gemini/gemini-3.8-flash": ProviderCapability(
        provider="gemini",
        model_id="gemini/gemini-3.8-flash",
        identity_url="https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
        pricing_url=GOOGLE_PRICE,
        verified_on=VERIFICATION_DATE,
        input_token_method="Gemini countTokens endpoint (input estimate; not a bill cap)",
        output_token_bound=65536,
        billing_control=GOOGLE_BILLING,
        pricing_version="gemini-standard-through-2026-12-31-v2026-11-22",
        input_usd_per_million=Decimal("0.75"),
        output_usd_per_million=Decimal("3.75"),
        cached_input_usd_per_million=Decimal("0.075"),
        limitation=(
            "Project spend caps are experimental and may overrun during billing-data delay; "
            "account tier caps are shared across linked projects."
        ),
    ),
}


def resolve_capability(model_id: str) -> ProviderCapability | None:
    return CAPABILITIES.get(model_id)


def price_provenance_is_current(capability: ProviderCapability, *, today: date) -> bool:
    return today <= capability.verified_on + timedelta(days=capability.max_price_age_days)
