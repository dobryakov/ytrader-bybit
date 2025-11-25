"""WebSocket connection state model."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConnectionStatus(str, Enum):
    """WebSocket connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class WebSocketState(BaseModel):
    """Represents the state of the WebSocket connection to Bybit."""

    connection_id: UUID = Field(default_factory=uuid4, description="Unique connection identifier")
    environment: str = Field(..., description="Bybit environment: 'mainnet' or 'testnet'")
    status: ConnectionStatus = Field(
        default=ConnectionStatus.DISCONNECTED, description="Current connection status"
    )
    connected_at: Optional[datetime] = Field(
        default=None, description="When connection was established"
    )
    last_heartbeat_at: Optional[datetime] = Field(
        default=None, description="Last successful heartbeat timestamp"
    )
    reconnect_count: int = Field(default=0, description="Number of reconnection attempts")
    last_error: Optional[str] = Field(default=None, description="Last error message (if any)")
    subscriptions_active: int = Field(
        default=0, description="Count of active subscriptions"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

