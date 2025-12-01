"""Metrics and observability endpoints for Position Manager."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...config.rabbitmq import RabbitMQConnection
from ...services.portfolio_manager import default_portfolio_manager
from ...services.position_manager import PositionManager
from ...utils.tracing import get_or_create_trace_id


logger = get_logger(__name__)
router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> JSONResponse:
    """Return basic service metrics suitable for monitoring systems.

    This endpoint is intentionally lightweight and unauthenticated so that
    health/monitoring agents can scrape it without API keys.
    """
    trace_id = get_or_create_trace_id()

    # Connectivity flags
    db_connected = DatabaseConnection.is_connected()
    queue_connected = RabbitMQConnection.is_connected()

    # Portfolio cache metrics
    cache_hit_rate = default_portfolio_manager._cache_hit_rate  # type: ignore[attr-defined]

    # Validation statistics
    position_manager = PositionManager()
    validation_stats = position_manager.get_validation_statistics()

    # Response payload
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database_connected": db_connected,
        "queue_connected": queue_connected,
        "portfolio_cache_hit_rate": cache_hit_rate,
        "validation_statistics": validation_stats,
    }

    logger.info("metrics_endpoint_completed", trace_id=trace_id, **payload)
    return JSONResponse(status_code=200, content=payload)


