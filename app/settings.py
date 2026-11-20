"""Typed, fail-closed application configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TUTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["development", "test", "production"] = "development"
    provider: Literal["mock", "litellm"] = "mock"
    small_model_identifier: str = Field(default="mock-small", min_length=1)
    strong_model_identifier: str = Field(default="mock-strong", min_length=1)
    litellm_api_key: SecretStr | None = None
    litellm_api_base: str | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=1, ge=0, le=3)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=5)
    fallback_model_identifier: str | None = None
    router_strategy: Literal["always_small", "always_strong", "random", "rule"] = "rule"
    random_small_probability: float = Field(default=0.5, ge=0, le=1)
    random_seed: str = "42"
    rule_max_short_query_characters: int = Field(default=160, ge=1, le=12000)
    rule_max_short_query_tokens: int = Field(default=30, ge=1, le=3000)
    rule_max_math_symbols_for_small: int = Field(default=3, ge=0, le=1000)
    auth_enabled: bool = False
    api_bearer_token: SecretStr | None = None
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=10000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    per_request_budget_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def enforce_secure_configuration(self) -> Settings:
        if self.auth_enabled and (
            not self.api_bearer_token or not self.api_bearer_token.get_secret_value().strip()
        ):
            raise ValueError("TUTOR_API_BEARER_TOKEN is required when TUTOR_AUTH_ENABLED=true")
        if self.environment == "production":
            if (
                not self.auth_enabled
                or not self.api_bearer_token
                or not self.api_bearer_token.get_secret_value().strip()
            ):
                raise ValueError("production requires bearer authentication and a configured token")
            if self.provider == "mock":
                raise ValueError("production mode cannot use the mock provider")
            if self.litellm_api_base and not self.litellm_api_base.startswith("https://"):
                raise ValueError("production LiteLLM API base must use HTTPS")
        if self.provider == "litellm" and (
            not self.litellm_api_key or not self.litellm_api_key.get_secret_value().strip()
        ):
            raise ValueError("TUTOR_LITELLM_API_KEY is required when TUTOR_PROVIDER=litellm")
        if self.fallback_model_identifier and self.fallback_model_identifier in {
            self.small_model_identifier,
            self.strong_model_identifier,
        }:
            raise ValueError("fallback model must differ from both primary model identifiers")
        return self
