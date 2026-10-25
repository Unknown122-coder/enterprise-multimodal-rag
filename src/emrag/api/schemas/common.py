"""Shared API primitives."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """A cursor-paginated page of results."""

    model_config = ConfigDict(extra="forbid")

    items: list[T] = Field(default_factory=list)
    next_cursor: str | None = None


class Health(BaseModel):
    """Liveness response."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    version: str


class Readiness(BaseModel):
    """Readiness response: the API serves traffic only when every check passes."""

    model_config = ConfigDict(extra="forbid")

    ready: bool
    checks: dict[str, bool]
    detail: dict[str, str] = Field(default_factory=dict)
