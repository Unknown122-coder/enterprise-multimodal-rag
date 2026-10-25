"""Domain exceptions.

Every error carries a stable ``code`` and an HTTP ``status`` so the API layer can
render RFC 9457 problem documents without a translation table.
"""

from __future__ import annotations


class EmragError(Exception):
    """Base class for all application errors."""

    code = "INTERNAL_ERROR"
    status = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class ValidationFailed(EmragError):
    """Input failed validation."""

    code = "VALIDATION_ERROR"
    status = 400


class Unauthenticated(EmragError):
    """No usable credential was presented."""

    code = "UNAUTHENTICATED"
    status = 401


class Forbidden(EmragError):
    """The credential is valid but lacks access to the resource."""

    code = "FORBIDDEN"
    status = 403


class NotFound(EmragError):
    """The requested resource does not exist for this tenant."""

    code = "NOT_FOUND"
    status = 404


class PayloadTooLarge(EmragError):
    """Upload exceeded the configured size or page limit."""

    code = "PAYLOAD_TOO_LARGE"
    status = 413


class UnsupportedMedia(EmragError):
    """File type is not accepted."""

    code = "UNSUPPORTED_MEDIA"
    status = 415


class RateLimited(EmragError):
    """Caller exceeded the per-key request budget."""

    code = "RATE_LIMITED"
    status = 429

    def __init__(self, message: str, *, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaExhausted(EmragError):
    """All candidate models for a tier are out of upstream quota."""

    code = "QUOTA_EXHAUSTED"
    status = 503

    def __init__(self, message: str, *, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ModelUnavailable(EmragError):
    """No model is configured for the requested tier.

    Distinct from :class:`QuotaExhausted`: waiting does not help, because the
    cause is missing configuration rather than a spent budget.
    """

    code = "MODEL_UNAVAILABLE"
    status = 503


class ProviderError(EmragError):
    """An upstream LLM provider returned an unrecoverable error."""

    code = "PROVIDER_ERROR"
    status = 502


class StoreUnavailable(EmragError):
    """The vector store or artifact store could not be reached."""

    code = "STORE_UNAVAILABLE"
    status = 503


class IngestFailed(EmragError):
    """The ingestion pipeline failed for this document."""

    code = "INGEST_FAILED"
    status = 500


class DeadlineExceeded(EmragError):
    """The request could not finish inside its deadline."""

    code = "DEADLINE_EXCEEDED"
    status = 504


class ConfigurationMissing(EmragError):
    """A required configuration value is absent or invalid."""

    code = "CONFIGURATION_MISSING"
    status = 500
