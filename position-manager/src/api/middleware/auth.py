"""API key authentication middleware for Position Manager."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader

from ...config.settings import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def api_key_auth(api_key: str = Header(default=None, alias="X-API-Key")) -> str:
    """Simple dependency for enforcing X-API-Key header."""
    if not api_key or api_key != settings.position_manager_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


async def api_key_middleware(request: Request, call_next):
    """Middleware variant to ensure API key is present for all protected routes.

    We still prefer explicit dependencies on routers, but this can be used
    for global protection if needed.
    """
    # Allow unauthenticated access to health endpoint
    if request.url.path == "/health":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != settings.position_manager_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    response = await call_next(request)
    return response



