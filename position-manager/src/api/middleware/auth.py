"""API key authentication middleware for Position Manager."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Header, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader

from ...config.logging import get_logger
from ...config.settings import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


_rate_limit_state: Dict[Tuple[str, str], Tuple[int, float]] = defaultdict(lambda: (0, 0.0))

# Simple in-memory counters for rate limiting observability (T108).
_rate_limit_metrics = {
    "total_requests": 0,
    "rate_limited_requests": 0,
}

logger = get_logger(__name__)


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

    # --- Simple per-API-key rate limiting (sliding window approximation) ---
    if settings.position_manager_rate_limit_enabled:
        _rate_limit_metrics["total_requests"] += 1
        # Window of 60 seconds using configured default limit
        limit = settings.position_manager_rate_limit_default
        now = time.time()
        window_start = int(now // 60) * 60
        key = (api_key, str(window_start))

        count, ts = _rate_limit_state[key]
        if ts != window_start:
            count, ts = 0, window_start

        if count >= limit:
            retry_after = max(1, window_start + 60 - int(now))
            _rate_limit_metrics["rate_limited_requests"] += 1
            logger.warning(
                "api_rate_limit_exceeded",
                api_key_hash=hash(api_key),
                limit=limit,
                window_start=window_start,
                retry_after=retry_after,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        _rate_limit_state[key] = (count + 1, ts)

    response = await call_next(request)
    return response



