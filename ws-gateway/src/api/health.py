"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

from ..config.settings import settings
from ..services.database.connection import DatabaseConnection
from ..services.database.subscription_repository import SubscriptionRepository
from ..services.queue.connection import QueueConnection
from ..services.websocket.connection import get_connection

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "unhealthy"]
    service: str
    websocket_connected: bool = False
    websocket_status: str = "unknown"
    websocket_environment: str = "unknown"
    websocket_endpoint_type: str = "unknown"
    websocket_public_category: str | None = None
    websocket_last_message_at: str | None = None
    database_connected: bool = False
    queue_connected: bool = False
    active_subscriptions: int = 0


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
    websocket_environment = websocket_connection.state.environment
    websocket_endpoint_type = websocket_connection.state.endpoint_type
    websocket_public_category = (
        settings.bybit_ws_public_category
        if websocket_connection.state.endpoint_type == "public"
        else None
    )
    websocket_last_message_at = (
        websocket_connection.state.last_message_at.isoformat()
        if websocket_connection.state.last_message_at
        else None
    )

    # Check database connection status
    database_connected = DatabaseConnection.is_connected()

    # Check queue connection status
    queue_connected = QueueConnection.is_connected()

    # Determine overall health status
    # Service is healthy if database and queue are connected
    # WebSocket connection status is informational but doesn't affect health
    # (service can be healthy even if WebSocket is temporarily disconnected)
    overall_status = (
        "healthy" if (database_connected and queue_connected) else "unhealthy"
    )

    # Count active subscriptions (best-effort; errors do not flip overall status)
    active_subscriptions = 0
    if database_connected:
        try:
            active_subscriptions = await SubscriptionRepository.count_active_subscriptions()
        except Exception:
            # Already logged at repository/database layer; keep health check resilient
            active_subscriptions = 0

    return HealthResponse(
        status=overall_status,
        service="ws-gateway",
        websocket_connected=websocket_connected,
        websocket_status=websocket_status,
        websocket_environment=websocket_environment,
        websocket_endpoint_type=websocket_endpoint_type,
        websocket_public_category=websocket_public_category,
        websocket_last_message_at=websocket_last_message_at,
        database_connected=database_connected,
        queue_connected=queue_connected,
        active_subscriptions=active_subscriptions,
    )

