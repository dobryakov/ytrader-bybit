"""API key authentication middleware.

This middleware enforces API key authentication for REST endpoints under `/api/`.
The API key is provided via the `X-API-Key` header or the `api_key` query parameter
and is validated against `settings.dashboard_api_key`.

On authentication failure, a JSON 401 response is returned.
"""

from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.settings import settings
from ...config.logging import get_logger
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing simple API key authentication for REST endpoints."""

    def __init__(self, app, api_prefix: str = "/api"):
        super().__init__(app)
        self.api_prefix = api_prefix

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request and validate API key for protected routes."""
        # Only protect API routes (e.g. /api/v1/...)
        path = request.url.path
        if not path.startswith(self.api_prefix):
            return await call_next(request)

        trace_id = get_or_create_trace_id()

        # Extract API key from header or query parameter
        header_key = request.headers.get("X-API-Key")
        query_key = request.query_params.get("api_key")
        api_key = header_key or query_key

        if not api_key:
            logger.warning(
                "api_authentication_failed_missing_key",
                path=path,
                method=request.method,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing API key",
                    "code": "AUTHENTICATION_FAILED",
                    "details": {"reason": "missing_api_key"},
                },
            )

        if api_key != settings.dashboard_api_key:
            logger.warning(
                "api_authentication_failed_invalid_key",
                path=path,
                method=request.method,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid API key",
                    "code": "AUTHENTICATION_FAILED",
                    "details": {"reason": "invalid_api_key"},
                },
            )

        return await call_next(request)

