"""Request/response logging middleware with trace IDs.

This middleware logs incoming HTTP requests and outgoing responses, attaching a
trace ID to each request using structlog's contextvars integration.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.logging import get_logger
from ...utils.tracing import clear_trace_id, generate_trace_id, set_trace_id

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses with trace IDs."""

    async def dispatch(self, request: Request, call_next: Callable):
        # Generate and set trace ID for this request
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        start_time = time.time()
        logger.info(
            "http_request_received",
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params) if request.query_params else None,
            client_host=str(request.client.host if request.client else "unknown"),
            client_port=request.client.port if request.client else None,
            user_agent=request.headers.get("user-agent"),
            content_type=request.headers.get("content-type"),
            content_length=request.headers.get("content-length"),
            trace_id=trace_id,
        )

        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 0)
        except Exception as e:
            # Log errors with full context
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "http_request_error",
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
                trace_id=trace_id,
                exc_info=True,
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            status_code = getattr(response, "status_code", 0) if 'response' in locals() else 0
            logger.info(
                "http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                trace_id=trace_id,
            )
            # Clear context for next request
            clear_trace_id()

        # Propagate trace ID to clients via response header
        response.headers["X-Trace-Id"] = trace_id
        return response


