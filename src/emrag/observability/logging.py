"""Structured JSON logging.

One JSON object per line on stdout, which is what both Hugging Face Spaces and
Render collect. A redaction processor removes anything that looks like a secret so
logs can be shared freely.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="-")

# Keys whose values are secrets. The patterns are anchored on whole words so a
# token *count* is not mistaken for a token: redacting `prompt_tokens` would
# destroy exactly the quota accounting these logs exist to provide.
_SECRET_KEY_RE = re.compile(
    r"""
    (^|_)(
        api[_-]?key
      | access[_-]?token
      | refresh[_-]?token
      | auth[_-]?token
      | id[_-]?token
      | bearer[_-]?token
      | secret
      | password
      | passwd
      | credential
      | authorization
      | private[_-]?key
    )($|_)
    | ^(token|bearer|authorization|secret|password)$
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Keys that look sensitive but are counters or model metadata, kept readable.
_SAFE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens",
        "token_count",
        "est_tokens",
        "max_tokens",
        "tokens_used",
        "tpm_used",
        "tpm_limit",
        "tpd_used",
        "tpd_limit",
    }
)
_REDACTED = "***redacted***"
_configured = False


def is_secret_key(key: str) -> bool:
    """Whether a log field name should have its value redacted."""
    if key.lower() in _SAFE_KEYS:
        return False
    return bool(_SECRET_KEY_RE.search(key))


def _redact(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Replace secret-looking values with a placeholder."""
    for key, value in list(event_dict.items()):
        if is_secret_key(key):
            event_dict[key] = _REDACTED
        elif isinstance(value, dict):
            event_dict[key] = _redact(_logger, _name, dict(value))
    return event_dict


def _bind_context(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Attach the ambient request and tenant identifiers to every line."""
    event_dict.setdefault("request_id", request_id_var.get())
    event_dict.setdefault("tenant_id", tenant_id_var.get())
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    global _configured
    if _configured:
        return

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            _bind_context,
            _redact,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper()) if isinstance(level, str) else level
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.getLevelName(level.upper())
    )
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> Any:
    """Return a bound logger; configures logging lazily on first use.

    The module name is bound explicitly rather than taken from a stdlib logger,
    because events are rendered straight to stdout without going through the
    stdlib handler chain.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger().bind(logger=name)


def set_request_context(request_id: str, tenant_id: str = "-") -> None:
    """Bind the current request and tenant so every log line carries them."""
    request_id_var.set(request_id)
    tenant_id_var.set(tenant_id)


def clear_request_context() -> None:
    """Reset the ambient logging context."""
    request_id_var.set("-")
    tenant_id_var.set("-")
