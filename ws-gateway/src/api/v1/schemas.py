"""Pydantic schemas for v1 subscription REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


ChannelType = Literal[
    "trades",
    "ticker",
    "orderbook",
    "order",
    "balance",
    "position",
    "kline",
    "liquidation",
]


class CreateSubscriptionRequest(BaseModel):
    """Request body for creating a new subscription."""

    channel_type: ChannelType = Field(
        ...,
        description="Type of data channel to subscribe to",
    )
    symbol: Optional[str] = Field(
        default=None,
        description=(
            "Trading pair symbol (e.g., 'BTCUSDT'). Required for symbol-specific "
            "channels (trades, ticker, orderbook, order, kline)."
        ),
    )
    requesting_service: str = Field(
        ...,
        description="Identifier of the microservice requesting this subscription",
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Optional: full Bybit topic string. If omitted, it will be derived "
            "from channel_type and symbol."
        ),
    )


class SubscriptionResponse(BaseModel):
    """Subscription representation returned by the API."""

    id: UUID
    channel_type: ChannelType
    symbol: Optional[str] = None
    topic: str
    requesting_service: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_event_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str
    code: Optional[str] = None
    details: Optional[dict] = None


class SubscriptionListResponse(BaseModel):
    """Response for listing subscriptions."""

    subscriptions: list[SubscriptionResponse]
    total: int


