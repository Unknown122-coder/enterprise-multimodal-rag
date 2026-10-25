"""Global exception handlers.

Every error leaves the API as an RFC 9457 problem document carrying the request
id, so a client can quote it and an operator can find the matching log line.
Nothing is swallowed: unexpected exceptions are logged with a traceback and
reported as a generic internal error without leaking internals to the caller.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from emrag.api.schemas.errors import Problem
from emrag.core.errors import EmragError
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"

_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _respond(problem: Problem) -> JSONResponse:
    headers = {}
    if problem.retry_after_s:
        headers["Retry-After"] = str(problem.retry_after_s)
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application."""

    @app.exception_handler(EmragError)
    async def _domain(request: Request, exc: EmragError) -> JSONResponse:
        problem = Problem.build(
            code=exc.code,
            status=exc.status,
            title=_TITLES.get(exc.status, "Error"),
            detail=exc.detail,
            request_id=_request_id(request),
            retry_after_s=getattr(exc, "retry_after_s", None),
        )
        if exc.status >= 500:
            logger.error("http.domain_error", code=exc.code, detail=exc.detail)
        else:
            logger.info("http.domain_error", code=exc.code, detail=exc.detail)
        return _respond(problem)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        detail = f"{location}: {first.get('msg', 'invalid value')}"
        return _respond(
            Problem.build(
                code="VALIDATION_ERROR",
                status=400,
                title=_TITLES[400],
                detail=detail,
                request_id=_request_id(request),
            )
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {
            401: "UNAUTHENTICATED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            429: "RATE_LIMITED",
        }.get(exc.status_code, "HTTP_ERROR")
        return _respond(
            Problem.build(
                code=code,
                status=exc.status_code,
                title=_TITLES.get(exc.status_code, "Error"),
                detail=str(exc.detail),
                request_id=_request_id(request),
            )
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "http.unhandled",
            error=f"{type(exc).__name__}: {exc}",
            path=request.url.path,
            exc_info=True,
        )
        return _respond(
            Problem.build(
                code="INTERNAL_ERROR",
                status=500,
                title=_TITLES[500],
                detail="An unexpected error occurred. Quote the request id when reporting it.",
                request_id=_request_id(request),
            )
        )
