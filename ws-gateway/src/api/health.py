"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

from ..services.database.connection import DatabaseConnection
from ..services.queue.connection import QueueConnection
from ..services.websocket.connection import get_connection

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "unhealthy"]
    service: str
    websocket_connected: bool = False
    websocket_status: str = "unknown"
    database_connected: bool = False
    queue_connected: bool = False


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service health status including connection status for WebSocket, database, and queue.
    """
    # Check WebSocket connection status
    websocket_connection = get_connection()
    websocket_connected = websocket_connection.is_connected
    websocket_status = websocket_connection.state.status.value

    # Check database connection status
    database_connected = DatabaseConnection.is_connected()

    # Check queue connection status
    queue_connected = QueueConnection.is_connected()

    # Determine overall health status
    # Service is healthy if database and queue are connected
    # WebSocket connection status is informational but doesn't affect health
    # (service can be healthy even if WebSocket is temporarily disconnected)
    overall_status = "healthy" if (database_connected and queue_connected) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        service="ws-gateway",
        websocket_connected=websocket_connected,
        websocket_status=websocket_status,
        database_connected=database_connected,
        queue_connected=queue_connected,
    )

