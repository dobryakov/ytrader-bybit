"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "unhealthy"]
    service: str
    websocket_connected: bool = False
    database_connected: bool = False
    queue_connected: bool = False


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service health status including connection status for WebSocket, database, and queue.
    """
    # TODO: Check actual connection status when WebSocket, database, and queue are implemented
    # For now, return basic health status
    return HealthResponse(
        status="healthy",
        service="ws-gateway",
        websocket_connected=False,
        database_connected=False,
        queue_connected=False,
    )

