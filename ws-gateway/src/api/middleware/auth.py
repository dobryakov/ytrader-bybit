"""API key authentication middleware.

This middleware enforces API key authentication for REST endpoints under `/api/`.
The API key is provided via the `X-API-Key` header or the `api_key` query parameter
and is validated against `settings.ws_gateway_api_key`.

On authentication failure, a JSON 401 response is returned matching the
`ErrorResponse` schema from the OpenAPI contracts.
"""

from __future__ import annotations

from typing import Callable
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...config.settings import settings
from ...config.logging import get_logger
from ...exceptions import AuthenticationError

logger = get_logger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing simple API key authentication for REST endpoints."""

    def __init__(self, app, api_prefix: str = "/api"):
        super().__init__(app)
        self.api_prefix = api_prefix

    async def dispatch(self, request: Request, call_next: Callable):
        # Only protect API routes (e.g. /api/v1/...)
        path = request.url.path
        if not path.startswith(self.api_prefix):
            return await call_next(request)

        # Extract API key from header or query parameter
        header_key = request.headers.get("X-API-Key")
        query_key = request.query_params.get("api_key")
        api_key = header_key or query_key

        try:
            if not api_key:
                raise AuthenticationError("Missing API key")
            if api_key != settings.ws_gateway_api_key:
                raise AuthenticationError("Invalid API key")
        except AuthenticationError as exc:
            logger.warning(
                "api_authentication_failed",
                path=path,
                method=request.method,
                error=str(exc),
            )
            # Match ErrorResponse schema
            return JSONResponse(
                status_code=401,
                content={
                    "error": exc.message,
                    "code": "AUTHENTICATION_FAILED",
                    "details": {"reason": "invalid_or_missing_api_key"},
                },
            )

        return await call_next(request)


