"""Request/response logging middleware with trace ID support."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...config.logging import get_logger
from ...utils.tracing import get_or_create_trace_id, set_trace_id

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses with trace IDs."""

    def __init__(self, app: ASGIApp):
        """Initialize logging middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log with trace ID.

        Args:
            request: HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response
        """
        # Extract or create trace ID from headers
        trace_id = request.headers.get("X-Trace-Id") or get_or_create_trace_id()
        set_trace_id(trace_id)

        # Add trace ID to request state for use in handlers
        request.state.trace_id = trace_id

        # Log request
        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None

        logger.info(
            "http_request_received",
            method=method,
            path=path,
            query_params=query_params,
            client_host=request.client.host if request.client else None,
            trace_id=trace_id,
        )

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            elapsed_time = time.time() - start_time

            # Log response
            logger.info(
                "http_response_sent",
                method=method,
                path=path,
                status_code=status_code,
                elapsed_time=elapsed_time,
                trace_id=trace_id,
            )

            # Add trace ID to response headers
            response.headers["X-Trace-Id"] = trace_id

            return response

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                "http_request_failed",
                method=method,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
                elapsed_time=elapsed_time,
                trace_id=trace_id,
                exc_info=True,
            )
            raise

