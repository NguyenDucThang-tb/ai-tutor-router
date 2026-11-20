from __future__ import annotations

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.errors import ErrorCategory, GatewayError
from app.gateway import Gateway, LiteLLMGateway, MockGateway
from app.middleware import RequestContextMiddleware
from app.observability import configure_logging
from app.rate_limit import InMemoryRateLimiter
from app.routing import RouterConfig, build_router
from app.schemas import ErrorResponse, TutorRequest, TutorResponse
from app.service import TutorService
from app.settings import Settings

ERROR_MESSAGES = {
    ErrorCategory.AUTHENTICATION_FAILURE: (
        502,
        "provider_authentication_failure",
        "Provider authentication failed",
    ),
    ErrorCategory.RATE_LIMIT: (503, "provider_rate_limit", "Provider rate limit reached"),
    ErrorCategory.TIMEOUT: (504, "provider_timeout", "Provider request timed out"),
    ErrorCategory.TEMPORARY_PROVIDER_FAILURE: (
        503,
        "provider_unavailable",
        "Provider is temporarily unavailable",
    ),
    ErrorCategory.INVALID_PROVIDER_RESPONSE: (
        502,
        "invalid_provider_response",
        "Provider returned an invalid response",
    ),
    ErrorCategory.UNSUPPORTED_MODEL: (502, "unsupported_model", "Configured model is unsupported"),
    ErrorCategory.BUDGET_EXCEEDED: (
        402,
        "budget_exceeded",
        "Configured per-request cost limit was exceeded",
    ),
}


def _make_gateway(settings: Settings) -> Gateway:
    if settings.provider == "mock":
        return MockGateway()
    assert settings.litellm_api_key is not None
    return LiteLLMGateway(settings.litellm_api_key.get_secret_value(), settings.litellm_api_base)


def create_app(settings: Settings | None = None, gateway: Gateway | None = None) -> FastAPI:
    config = settings or Settings()
    configure_logging(config.log_level)
    router_config = RouterConfig(
        small_model_id=config.small_model_identifier,
        strong_model_id=config.strong_model_identifier,
        random_small_probability=config.random_small_probability,
        random_seed=config.random_seed,
        max_short_query_characters=config.rule_max_short_query_characters,
        max_short_query_tokens=config.rule_max_short_query_tokens,
        max_math_symbols_for_small=config.rule_max_math_symbols_for_small,
    )
    router = build_router(config.router_strategy, router_config)
    model_gateway = gateway or _make_gateway(config)
    service = TutorService(config, router, model_gateway)
    limiter = InMemoryRateLimiter(config.rate_limit_requests_per_minute)

    app = FastAPI(title="AI Tutor Quality-Aware Router", version="0.2.0")
    app.state.settings = config
    app.state.service = service
    app.state.gateway = model_gateway
    app.state.rate_limiter = limiter
    app.add_middleware(RequestContextMiddleware, limiter=limiter)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "Request body did not match the expected schema",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            },
        )

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError):
        status, code, message = ERROR_MESSAGES[exc.category]
        payload = ErrorResponse(
            error={
                "code": code,
                "message": message,
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
            attempts=list(exc.attempts),
        )
        return JSONResponse(
            status_code=status,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if exc.status_code == 401:
            code, message = "authentication_required", "Valid bearer authentication is required"
        else:
            code, message = "request_rejected", "Request was rejected"
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            },
        )

    async def authentication(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        if not config.auth_enabled:
            if config.environment == "production":
                # Defensive even though Settings rejects this state.
                raise HTTPException(status_code=503, detail="Authentication is misconfigured")
            return
        token = config.api_bearer_token.get_secret_value() if config.api_bearer_token else ""
        supplied = (
            authorization.removeprefix("Bearer ")
            if authorization and authorization.startswith("Bearer ")
            else ""
        )
        if not token or not secrets.compare_digest(token, supplied):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # Do not return exception messages, which can contain provider or secret material.
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            },
        )

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        return {"status": "ready", "provider": model_gateway.provider}

    @app.get("/v1/router/strategies")
    async def strategies():
        return {"strategies": ["always_small", "always_strong", "random", "rule"]}

    @app.post(
        "/v1/tutor/ask",
        response_model=TutorResponse,
        responses={
            401: {"model": ErrorResponse},
            402: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def ask(request: Request, body: TutorRequest, _auth: None = Depends(authentication)):
        return await service.ask(body, request.state.request_id, request.state.request_started_at)

    return app


app = create_app()
