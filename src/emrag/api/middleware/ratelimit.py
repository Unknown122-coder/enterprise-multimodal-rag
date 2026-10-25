"""Rate limiting.

In-process token buckets via slowapi, keyed by credential rather than client IP
so one tenant cannot exhaust another's budget. A single free-tier instance has no
shared state, so this needs no Redis; if the service is ever scaled out, swap the
limiter storage and nothing else changes.

This limits *callers*. Upstream LLM quotas are a separate concern handled by the
quota ledger.
"""

from __future__ import annotations

from fastapi import FastAPI
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from emrag.api.schemas.errors import Problem
from emrag.settings import Settings


def credential_key(request: Request) -> str:
    """Derive a limiter key from the bearer credential, falling back to the client."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token:
            # Key on a prefix only: enough to separate callers, never the secret.
            return f"key:{token[:12]}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def build_limiter(settings: Settings) -> Limiter:
    """Create the limiter with a global default."""
    return Limiter(
        key_func=credential_key,
        default_limits=["300/minute"],
        headers_enabled=True,
        strategy="moving-window",
        enabled=settings.app_env != "test",
    )


def register_rate_limiting(app: FastAPI, limiter: Limiter) -> None:
    """Attach the limiter, its middleware and its error handler."""
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        problem = Problem.build(
            code="RATE_LIMITED",
            status=429,
            title="Too Many Requests",
            detail=f"Rate limit exceeded: {exc.detail}",
            request_id=getattr(request.state, "request_id", None),
            retry_after_s=60,
        )
        return JSONResponse(
            status_code=429,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
            headers={"Retry-After": "60"},
        )
