"""In-memory Event model for messages received from Bybit WebSocket."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal
from uuid import UUID, uuid4


EventType = Literal["trade", "ticker", "orderbook", "order", "balance", "position", "kline", "liquidation", "funding"]


@dataclass
class Event:
    """Represents a data message received from Bybit WebSocket.

    This structure matches the Event definition in data-model.md and is used
    throughout the event processing pipeline before data is routed to queues
    or persisted.
    """

    event_id: UUID
    event_type: EventType
    topic: str
    timestamp: datetime
    received_at: datetime
    payload: Dict[str, Any]
    trace_id: str

    @classmethod
    def create(
        cls,
        event_type: EventType,
        topic: str,
        timestamp: datetime,
        payload: Dict[str, Any],
        trace_id: str,
    ) -> "Event":
        """Factory for creating a new Event instance."""
        return cls(
            event_id=uuid4(),
            event_type=event_type,
            topic=topic,
            timestamp=timestamp,
            received_at=datetime.utcnow(),
            payload=payload,
            trace_id=trace_id,
        )


