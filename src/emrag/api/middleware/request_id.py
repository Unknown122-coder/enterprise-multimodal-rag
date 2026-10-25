"""Request id and access logging middleware."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from emrag.core.ids import new_request_id
from emrag.observability.logging import clear_request_context, get_logger, set_request_context
from emrag.observability.metrics import METRICS
from emrag.observability.tracing import current_trace_id

logger = get_logger(__name__)

HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it to the logging context and times the call."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request."""
        request_id = request.headers.get(HEADER) or new_request_id()
        request.state.request_id = request_id
        set_request_context(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
        response.headers[HEADER] = request_id
        response.headers["X-Response-Time-ms"] = str(latency_ms)
        trace_id = current_trace_id()
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        route = request.scope.get("route")
        template = getattr(route, "path", request.url.path)
        if template not in ("/healthz", "/readyz", "/v1/metrics"):
            METRICS.increment("http.requests", route=template, status=str(response.status_code))
            METRICS.observe("http.latency_ms", latency_ms, route=template)
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=latency_ms,
            )
        clear_request_context()
        return response
