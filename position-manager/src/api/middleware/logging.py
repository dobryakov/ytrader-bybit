"""Request/response logging middleware with trace ID support."""

from __future__ import annotations

from time import monotonic

from fastapi import Request

from ...config.logging import get_logger
from ...utils.tracing import get_or_create_trace_id, set_trace_id, clear_trace_id


logger = get_logger(__name__)


async def logging_middleware(request: Request, call_next):
    """Log incoming requests and outgoing responses with trace IDs."""
    start = monotonic()
    trace_id = request.headers.get("X-Trace-Id") or get_or_create_trace_id()
    set_trace_id(trace_id)

    # Best-effort request size introspection (without buffering large bodies).
    content_length = request.headers.get("content-length")

    logger.info(
        "request_received",
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        client=str(request.client.host if request.client else None),
        content_length=content_length,
        trace_id=trace_id,
    )

    try:
        response = await call_next(request)
    finally:
        duration_ms = int((monotonic() - start) * 1000)
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=getattr(response, "status_code", None),
            duration_ms=duration_ms,
            response_content_length=getattr(response, "headers", {}).get("content-length"),
            trace_id=trace_id,
        )
        clear_trace_id()

    # Propagate trace ID back to caller
    response.headers["X-Trace-Id"] = trace_id
    return response



