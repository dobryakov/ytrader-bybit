"""Pydantic schemas for v1 subscription REST API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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
    "funding",
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
            "channels (trades, ticker, orderbook, order, kline, funding)."
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


class BalanceRecord(BaseModel):
    """Represents a single balance record for a specific coin."""

    id: UUID
    coin: str
    wallet_balance: Decimal
    available_balance: Decimal
    frozen: Decimal
    event_timestamp: datetime
    received_at: datetime
    trace_id: Optional[str] = None
    equity: Optional[Decimal] = None
    usd_value: Optional[Decimal] = None
    margin_collateral: bool = False
    total_order_im: Decimal = Decimal("0")
    total_position_im: Decimal = Decimal("0")


class LatestBalanceView(BaseModel):
    """View model for latest balance per coin."""

    coin: str
    wallet_balance: Decimal
    available_balance: Decimal
    frozen: Decimal
    equity: Optional[Decimal] = None
    usd_value: Optional[Decimal] = None
    margin_collateral: bool = False
    total_order_im: Decimal = Decimal("0")
    total_position_im: Decimal = Decimal("0")
    event_timestamp: datetime
    received_at: datetime


class MarginBalanceView(BaseModel):
    """View model for latest account-level margin balance."""

    account_type: str
    total_equity: Decimal
    total_wallet_balance: Decimal
    total_margin_balance: Decimal
    total_available_balance: Decimal
    total_initial_margin: Decimal
    total_maintenance_margin: Decimal
    total_order_im: Decimal
    base_currency: str
    event_timestamp: datetime
    received_at: datetime


class LatestBalancesResponse(BaseModel):
    """Response for GET /api/v1/balances (latest balances per coin + margin summary)."""

    balances: list[LatestBalanceView]
    margin_balance: Optional[MarginBalanceView] = None
    total: int


class BalanceHistoryResponse(BaseModel):
    """Response for GET /api/v1/balances/history (historical balance records)."""

    balances: list[BalanceRecord]
    total: int


