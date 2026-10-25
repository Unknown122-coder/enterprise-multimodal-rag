"""Request tracing.

OpenTelemetry spans per pipeline stage, so a slow query can be attributed to
embedding, the vector search round trip, reranking or the model call rather than
guessed at. Vendor-neutral, because the free observability backends come and go.

Tracing is entirely optional and off unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
set. When it is off, :func:`span` is a context manager that does nothing, so call
sites need no conditionals and cost nothing measurable.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from typing import Any

from emrag.observability.logging import get_logger
from emrag.observability.metrics import METRICS

logger = get_logger(__name__)

SERVICE_NAME = "emrag"
_tracer: Any = None
_configured = False


def configure_tracing(endpoint: str | None, *, service_name: str = SERVICE_NAME) -> bool:
    """Set up OTLP tracing. Returns whether tracing is active.

    Missing packages or a bad endpoint are logged and ignored: telemetry must
    never be the reason a service fails to start.
    """
    global _tracer, _configured
    if _configured:
        return _tracer is not None
    _configured = True

    if not endpoint:
        logger.debug("tracing.disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("tracing.unavailable", reason="opentelemetry packages not installed")
        return False

    try:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
    except Exception as exc:  # noqa: BLE001 - never fail startup on telemetry
        logger.warning("tracing.setup_failed", error=str(exc)[:200])
        return False

    logger.info("tracing.enabled", endpoint=endpoint, service=service_name)
    return True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Time a pipeline stage, as an OTel span when enabled and a metric always.

    The metric is recorded either way, which is what makes this worth calling
    even on a deployment with no tracing backend.
    """
    started = time.perf_counter()
    if _tracer is None:
        try:
            yield
        finally:
            METRICS.observe(f"stage.{name}_ms", (time.perf_counter() - started) * 1000)
        return

    with _tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            if value is not None:
                active.set_attribute(key, value)
        try:
            yield
        except Exception as exc:
            active.record_exception(exc)
            raise
        finally:
            METRICS.observe(f"stage.{name}_ms", (time.perf_counter() - started) * 1000)


def current_trace_id() -> str | None:
    """Return the active trace id as hex, for the ``X-Trace-ID`` response header."""
    if _tracer is None:
        return None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if not context.is_valid:
            return None
        return format(context.trace_id, "032x")
    except Exception:  # noqa: BLE001
        return None


def is_enabled() -> bool:
    """Whether tracing is active."""
    return _tracer is not None


def reset_for_tests() -> None:
    """Forget the configured state, so a test can reconfigure."""
    global _tracer, _configured
    _tracer = None
    _configured = False
