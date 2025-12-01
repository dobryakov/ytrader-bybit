"""Health check endpoints for service monitoring."""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.rabbitmq import RabbitMQConnection
from ...config.settings import settings
from ...config.logging import get_logger
from ...utils.bybit_client import get_bybit_client
from ...utils.metrics import get_metrics_summary

logger = get_logger(__name__)

router = APIRouter()


async def _check_database() -> str:
    """Check database connection availability.

    Returns:
        'available' if database is accessible, 'unavailable' otherwise
    """
    try:
        pool = await DatabaseConnection.get_pool()
        # Simple query to check connection
        await pool.fetchval("SELECT 1")
        return "available"
    except Exception as e:
        logger.warning("database_health_check_failed", error=str(e))
        return "unavailable"


async def _check_rabbitmq() -> str:
    """Check RabbitMQ connection availability.

    Returns:
        'available' if RabbitMQ is accessible, 'unavailable' otherwise
    """
    try:
        connection = await RabbitMQConnection.get_connection()
        if connection.is_closed:
            return "unavailable"
        return "available"
    except Exception as e:
        logger.warning("rabbitmq_health_check_failed", error=str(e))
        return "unavailable"


async def _check_bybit_api() -> str:
    """Check Bybit API availability.

    Returns:
        'available' if Bybit API is accessible, 'unavailable' otherwise
    """
    try:
        bybit_client = get_bybit_client()
        # Simple API call to check connectivity (server time endpoint doesn't require authentication)
        response = await bybit_client.get("/v5/market/time", authenticated=False)
        if response.get("retCode") == 0:
            return "available"
        return "unavailable"
    except Exception as e:
        logger.warning("bybit_api_health_check_failed", error=str(e))
        return "unavailable"


async def _check_ws_gateway() -> str:
    """Check WebSocket Gateway availability.

    Returns:
        'available' if WebSocket Gateway is accessible, 'unavailable' otherwise
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"http://{settings.ws_gateway_host}:{settings.ws_gateway_port}/health"
            response = await client.get(url)
            if response.status_code == 200:
                return "available"
            return "unavailable"
    except Exception as e:
        logger.warning("ws_gateway_health_check_failed", error=str(e))
        return "unavailable"


@router.get("/health")
async def get_health():
    """Health check endpoint.

    Returns:
        Overall health status of the service with core dependency flags.

    This endpoint is designed for Grafana System Health dashboard
    (spec task T074) and returns a compact schema:

    {
        "status": "healthy" | "unhealthy",
        "service": "order-manager",
        "database_connected": bool,
        "queue_connected": bool,
        "timestamp": "<ISO8601 UTC>"
    }
    """
    # Reuse existing dependency check helpers to avoid duplication
    database_status = await _check_database()
    rabbitmq_status = await _check_rabbitmq()

    database_connected = database_status == "available"
    queue_connected = rabbitmq_status == "available"

    is_healthy = database_connected and queue_connected
    status = "healthy" if is_healthy else "unhealthy"

    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "service": "order-manager",
            "database_connected": database_connected,
            "queue_connected": queue_connected,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )


@router.get("/live")
async def get_liveness():
    """Liveness probe endpoint.

    Returns:
        Liveness status (service is running and responsive)
    """
    try:
        # Simple check to ensure service is responsive
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )


@router.get("/ready")
async def get_readiness():
    """Readiness probe endpoint.

    Checks if service can accept requests and all dependencies are available.

    Returns:
        Readiness status with dependency information
    """
    try:
        # Check all dependencies
        database_status = await _check_database()
        rabbitmq_status = await _check_rabbitmq()
        bybit_api_status = await _check_bybit_api()
        ws_gateway_status = await _check_ws_gateway()

        dependencies: Dict[str, str] = {
            "database": database_status,
            "rabbitmq": rabbitmq_status,
            "bybit_api": bybit_api_status,
            "ws_gateway": ws_gateway_status,
        }

        # Service is ready if all critical dependencies are available
        # Database and RabbitMQ are critical; Bybit API and WS Gateway may be optional for basic operation
        is_ready = database_status == "available" and rabbitmq_status == "available"

        status = "ready" if is_ready else "not_ready"
        status_code = 200 if is_ready else 503

        return JSONResponse(
            status_code=status_code,
            content={
                "status": status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dependencies": dependencies,
            },
        )
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "dependencies": {
                    "database": "unknown",
                    "rabbitmq": "unknown",
                    "bybit_api": "unknown",
                    "ws_gateway": "unknown",
                },
            },
        )


@router.get("/metrics")
async def get_metrics():
    """Get performance metrics summary.

    Returns:
        Performance metrics including latency statistics and counters
    """
    try:
        from ...utils.metrics import get_metrics_summary

        metrics_summary = get_metrics_summary()
        return JSONResponse(
            status_code=200,
            content={
                "metrics": metrics_summary,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
    except Exception as e:
        logger.error("metrics_retrieval_failed", error=str(e), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to retrieve metrics",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

