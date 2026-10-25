"""Server-sent event encoding.

Events carry a name and a JSON payload. Any exception raised mid-stream is
delivered as an ``error`` event rather than truncating the response, so a client
always learns why a stream ended.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from emrag.api.schemas.errors import Problem
from emrag.core.errors import EmragError
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

HEARTBEAT = ": keepalive\n\n"


def encode(event: str, data: dict[str, Any]) -> str:
    """Encode one SSE frame."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def sse_stream(
    events: AsyncIterator[tuple[str, dict[str, Any]]],
) -> AsyncIterator[str]:
    """Adapt an event iterator into an SSE byte stream."""
    try:
        async for name, payload in events:
            yield encode(name, payload)
    except EmragError as exc:
        problem = Problem.build(
            code=exc.code,
            status=exc.status,
            title=type(exc).__name__,
            detail=exc.detail,
            retry_after_s=getattr(exc, "retry_after_s", None),
        )
        yield encode("error", problem.model_dump(exclude_none=True))
    except Exception as exc:
        logger.error("sse.unhandled", error=f"{type(exc).__name__}: {exc}", exc_info=True)
        yield encode(
            "error",
            Problem.build(
                code="INTERNAL_ERROR",
                status=500,
                title="Internal Server Error",
                detail="The stream ended unexpectedly.",
            ).model_dump(exclude_none=True),
        )
