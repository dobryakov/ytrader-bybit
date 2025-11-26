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

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses with trace IDs."""

    async def dispatch(self, request: Request, call_next: Callable):
        trace_id = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

        start_time = time.time()
        logger.info(
            "http_request_received",
            method=request.method,
            path=request.url.path,
            client=str(request.client.host if request.client else "unknown"),
        )

        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=getattr(response, "status_code", 0),
                duration_ms=round(duration_ms, 2),
            )
            # Clear context for next request
            structlog.contextvars.reset_contextvars()

        # Propagate trace ID to clients if useful
        response.headers["X-Trace-Id"] = trace_id
        return response


