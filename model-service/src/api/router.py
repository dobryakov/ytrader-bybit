"""
API routing structure with API key authentication middleware.

Sets up FastAPI router with authentication and route registration.
"""

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable

from ..config.settings import settings
from ..config.logging import get_logger, set_trace_id, bind_context
import uuid

logger = get_logger(__name__)

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Main API router
api_router = APIRouter(prefix="/api/v1")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with API key authentication.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response

        Raises:
            HTTPException: If API key is invalid or missing
        """
        # Skip authentication for health check endpoint
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            logger.warning("API request without API key", path=request.url.path, method=request.method)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
            )

        if api_key != settings.model_service_api_key:
            logger.warning("API request with invalid API key", path=request.url.path, method=request.method)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

        return await call_next(request)


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Middleware for trace ID propagation."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with trace ID.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response
        """
        # Get trace ID from header or generate new one
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())

        # Set trace ID in logging context
        set_trace_id(trace_id)

        # Bind additional context
        bind_context(
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else None,
        )

        # Call next middleware/handler
        response = await call_next(request)

        # Add trace ID to response header
        response.headers["X-Trace-ID"] = trace_id

        return response


def verify_api_key(api_key: str) -> bool:
    """
    Verify API key.

    Args:
        api_key: API key to verify

    Returns:
        True if valid, False otherwise
    """
    return api_key == settings.model_service_api_key

