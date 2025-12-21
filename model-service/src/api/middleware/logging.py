"""
Request/response logging middleware.

Logs all API requests and responses with trace IDs for observability.
"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status

from ...config.logging import get_logger, bind_context

logger = get_logger(__name__)


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all API requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response with trace ID.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with logging
        """
        # Skip logging for health check endpoint (too noisy)
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Record start time
        start_time = time.time()

        # Extract request details
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        content_type = request.headers.get("content-type")

        # Get trace ID from headers (set by TraceIDMiddleware)
        trace_id = request.headers.get("X-Trace-ID")

        # Log request
        logger.info(
            "API request received",
            method=method,
            path=path,
            query_params=query_params,
            client_ip=client_ip,
            user_agent=user_agent,
            content_type=content_type,
            trace_id=trace_id,
        )

        # Bind context for request
        bind_context(
            method=method,
            path=path,
            client_ip=client_ip,
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "API request failed",
                method=method,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                trace_id=trace_id,
                exc_info=True,
            )
            raise

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Extract response details
        status_code = response.status_code
        response_content_type = response.headers.get("content-type")

        # Log response
        log_level = "warning" if status_code >= 400 else "info"
        log_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "response_content_type": response_content_type,
            "trace_id": trace_id,
        }

        if log_level == "warning":
            logger.warning("API request completed with error", **log_data)
        else:
            logger.info("API request completed", **log_data)

        return response

