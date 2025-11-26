"""
Health check endpoint.

Provides health check endpoint with database, message queue, and model storage checks.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import Dict, Any

from ..database.connection import db_pool
from ..config.rabbitmq import rabbitmq_manager
from ..services.storage import model_storage
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    checks: Dict[str, Any]


@router.get("", status_code=status.HTTP_200_OK, response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Checks:
    - Database connectivity
    - Message queue connectivity
    - Model storage accessibility

    Returns:
        Health check response with status and check results
    """
    checks = {}
    overall_healthy = True

    # Check database
    try:
        if db_pool.is_connected:
            # Try a simple query
            await db_pool.fetchval("SELECT 1")
            checks["database"] = {"status": "healthy", "connected": True}
        else:
            checks["database"] = {"status": "unhealthy", "connected": False, "error": "Not connected"}
            overall_healthy = False
    except Exception as e:
        logger.error("Database health check failed", error=str(e), exc_info=True)
        checks["database"] = {"status": "unhealthy", "connected": False, "error": str(e)}
        overall_healthy = False

    # Check message queue
    try:
        if rabbitmq_manager.is_connected:
            checks["message_queue"] = {"status": "healthy", "connected": True}
        else:
            checks["message_queue"] = {"status": "unhealthy", "connected": False, "error": "Not connected"}
            overall_healthy = False
    except Exception as e:
        logger.error("Message queue health check failed", error=str(e), exc_info=True)
        checks["message_queue"] = {"status": "unhealthy", "connected": False, "error": str(e)}
        overall_healthy = False

    # Check model storage
    try:
        storage_health = model_storage.health_check()
        checks["model_storage"] = storage_health
        if not storage_health.get("healthy", False):
            overall_healthy = False
    except Exception as e:
        logger.error("Model storage health check failed", error=str(e), exc_info=True)
        checks["model_storage"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False

    response_status = "healthy" if overall_healthy else "unhealthy"
    status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    logger.info("Health check completed", status=response_status, checks=checks)

    return HealthResponse(
        status=response_status,
        service="model-service",
        checks=checks,
    )

