"""RFC 9457 problem documents."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ERROR_BASE = "https://emrag.dev/errors"


class Problem(BaseModel):
    """A problem document, returned as ``application/problem+json``."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(default=f"{ERROR_BASE}/internal_error")
    title: str
    status: int
    detail: str
    code: str
    request_id: str | None = None
    retry_after_s: int | None = None

    @classmethod
    def build(
        cls,
        *,
        code: str,
        status: int,
        title: str,
        detail: str,
        request_id: str | None = None,
        retry_after_s: int | None = None,
    ) -> Problem:
        """Create a problem document with a canonical type URI."""
        return cls(
            type=f"{ERROR_BASE}/{code.lower()}",
            title=title,
            status=status,
            detail=detail,
            code=code,
            request_id=request_id,
            retry_after_s=retry_after_s,
        )
