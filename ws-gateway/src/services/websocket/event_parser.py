"""Event parsing and validation for Bybit WebSocket messages."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from ...config.logging import get_logger
from ...models.event import Event
from ...models.subscription import Subscription

logger = get_logger(__name__)


def _parse_timestamp(ts: Any) -> Optional[dt.datetime]:
    """Parse Bybit timestamp (usually milliseconds since epoch)."""
    if ts is None:
        return None
    try:
        # Bybit often uses millisecond timestamps
        ts_int = int(ts)
        if ts_int > 10_000_000_000:  # treat as ms
            return dt.datetime.utcfromtimestamp(ts_int / 1000.0)
        return dt.datetime.utcfromtimestamp(ts_int)
    except Exception:
        return None


def parse_events_from_message(
    message: Dict[str, Any],
    subscription_lookup: Dict[str, Subscription],
    trace_id: str,
) -> List[Event]:
    """Parse one or more Event objects from a raw Bybit message.

    Args:
        message: Raw JSON-decoded message from WebSocket.
        subscription_lookup: Mapping from topic string to Subscription.
        trace_id: Trace identifier for logging/observability.
    """
    events: List[Event] = []

    topic = message.get("topic")
    if not topic:
        # Non-data messages (pings, acks, etc.) are ignored here
        logger.debug("event_parser_no_topic", message=message)
        return events

    subscription = subscription_lookup.get(topic)
    if not subscription:
        logger.debug("event_parser_unknown_topic", topic=topic)
        return events

    data = message.get("data")
    if not data:
        logger.debug("event_parser_no_data", topic=topic)
        return events

    received_at = dt.datetime.utcnow()
    ts = _parse_timestamp(message.get("ts")) or received_at

    # Normalise to list
    data_items = data if isinstance(data, list) else [data]

    for item in data_items:
        payload = dict(item)
        # Common payload enrichment
        payload.setdefault("topic", topic)

        event = Event.create(
            event_type=subscription.channel_type,
            topic=topic,
            timestamp=ts,
            payload=payload,
            trace_id=trace_id,
        )
        events.append(event)

    logger.debug(
        "events_parsed",
        topic=topic,
        count=len(events),
        trace_id=trace_id,
    )
    return events


