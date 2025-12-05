"""Subscription model for Bybit WebSocket channels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4


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


@dataclass
class Subscription:
    """Represents an active subscription to a Bybit WebSocket data channel.

    This mirrors the `subscriptions` table defined in data-model.md but is kept
    as an in-memory representation used by services.
    """

    id: UUID
    channel_type: ChannelType
    topic: str
    requesting_service: str
    is_active: bool = True
    symbol: Optional[str] = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_event_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        channel_type: ChannelType,
        topic: str,
        requesting_service: str,
        symbol: Optional[str] = None,
    ) -> "Subscription":
        """Factory for a new subscription instance."""
        now = datetime.utcnow()
        return cls(
            id=uuid4(),
            channel_type=channel_type,
            topic=topic,
            requesting_service=requesting_service,
            is_active=True,
            symbol=symbol,
            created_at=now,
            updated_at=now,
            last_event_at=None,
        )


