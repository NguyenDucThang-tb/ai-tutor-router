import pytest
from pydantic import ValidationError

from app.settings import Settings


def test_development_mock_defaults_are_explicit():
    settings = Settings(_env_file=None, environment="test")
    assert settings.provider == "mock"
    assert settings.auth_enabled is False


def test_auth_enabled_requires_token():
    with pytest.raises(ValidationError, match="TUTOR_API_BEARER_TOKEN"):
        Settings(_env_file=None, environment="test", auth_enabled=True)
    with pytest.raises(ValidationError, match="TUTOR_API_BEARER_TOKEN"):
        Settings(_env_file=None, environment="test", auth_enabled=True, api_bearer_token=" ")


def test_production_fails_closed_without_auth_and_real_provider():
    with pytest.raises(ValidationError, match="production requires bearer authentication"):
        Settings(_env_file=None, environment="production")


def test_production_rejects_mock_even_when_auth_is_configured():
    with pytest.raises(ValidationError, match="cannot use the mock provider"):
        Settings(
            _env_file=None,
            environment="production",
            auth_enabled=True,
            api_bearer_token="secure-token",
        )


def test_litellm_requires_configured_key():
    with pytest.raises(ValidationError, match="TUTOR_LITELLM_API_KEY"):
        Settings(_env_file=None, environment="test", provider="litellm")


def test_invalid_numeric_configuration_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", max_retries=99)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", request_timeout_seconds=0)


def test_valid_production_configuration_accepted():
    config = Settings(
        _env_file=None,
        environment="production",
        provider="litellm",
        auth_enabled=True,
        api_bearer_token="secure-token",
        litellm_api_key="provider-key",
    )
    assert config.environment == "production"


def test_production_rejects_insecure_custom_endpoint():
    with pytest.raises(ValidationError, match="must use HTTPS"):
        Settings(
            _env_file=None,
            environment="production",
            provider="litellm",
            auth_enabled=True,
            api_bearer_token="secure-token",
            litellm_api_key="provider-key",
            litellm_api_base="http://provider.internal",
        )
