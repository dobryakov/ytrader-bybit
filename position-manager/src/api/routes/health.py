"""Health check endpoint for Position Manager."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...config.rabbitmq import RabbitMQConnection
from ...utils.tracing import get_or_create_trace_id


logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Return basic service health information."""
    trace_id = get_or_create_trace_id()
    logger.info("health_check_request", trace_id=trace_id)

    db_connected = False
    queue_connected = False
    positions_count = 0

    try:
        # Database connectivity and positions count
        pool = await DatabaseConnection.get_pool()
        row = await pool.fetchrow("SELECT COUNT(*) AS cnt FROM positions")
        positions_count = int(row["cnt"]) if row and "cnt" in row else 0
        db_connected = True
    except Exception as e:  # pragma: no cover
        logger.error("health_check_db_failed", error=str(e), trace_id=trace_id)

    try:
        # RabbitMQ connectivity
        await RabbitMQConnection.get_connection()
        queue_connected = RabbitMQConnection.is_connected()
    except Exception as e:  # pragma: no cover
        logger.error("health_check_queue_failed", error=str(e), trace_id=trace_id)

    status = "healthy" if db_connected and queue_connected else "degraded"

    payload = {
        "status": status,
        "service": "position-manager",
        "database_connected": db_connected,
        "queue_connected": queue_connected,
        "positions_count": positions_count,
    }

    logger.info("health_check_completed", **payload, trace_id=trace_id)
    return JSONResponse(status_code=200, content=payload)



