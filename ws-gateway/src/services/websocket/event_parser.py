"""Event parsing and validation for Bybit WebSocket messages."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from ...config.logging import get_logger
from ...models.event import Event
from ...models.subscription import Subscription
from ...utils.tracing import get_or_create_trace_id

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


def _validate_message_structure(message: Dict[str, Any]) -> bool:
    """
    Validate that message has required structure.

    Args:
        message: Message dictionary to validate

    Returns:
        True if message is valid, False otherwise
    """
    if not isinstance(message, dict):
        return False
    # Basic validation - message should have either 'topic' or 'op' field
    return "topic" in message or "op" in message


def parse_events_from_message(
    message: Dict[str, Any],
    subscription_lookup: Dict[str, Subscription],
    trace_id: str,
) -> List[Event]:
    """Parse one or more Event objects from a raw Bybit message.

    Handles malformed messages gracefully by validating structure before parsing.

    Args:
        message: Raw JSON-decoded message from WebSocket.
        subscription_lookup: Mapping from topic string to Subscription.
        trace_id: Trace identifier for logging/observability.

    Returns:
        List of parsed Event objects, empty list if message is malformed or invalid.
    """
    events: List[Event] = []

    # Validate message structure (EC2: Handle malformed messages)
    if not _validate_message_structure(message):
        trace_id = get_or_create_trace_id()
        logger.warning(
            "event_parser_malformed_message",
            message_type=type(message).__name__,
            message_preview=str(message)[:200] if message else None,
            trace_id=trace_id,
        )
        return events

    topic = message.get("topic")
    if not topic:
        # Non-data messages (pings, acks, etc.) are ignored here
        trace_id = get_or_create_trace_id()
        logger.debug(
            "event_parser_no_topic",
            message=message,  # Full message for debugging
            trace_id=trace_id,
        )
        return events

    subscription = subscription_lookup.get(topic)
    if not subscription:
        trace_id = get_or_create_trace_id()
        logger.debug(
            "event_parser_unknown_topic",
            topic=topic,
            message=message,  # Full message for debugging
            trace_id=trace_id,
        )
        return events

    data = message.get("data")
    if not data:
        trace_id = get_or_create_trace_id()
        logger.debug(
            "event_parser_no_data",
            topic=topic,
            message=message,  # Full message for debugging
            trace_id=trace_id,
        )
        return events

    received_at = dt.datetime.utcnow()
    # Wallet messages use "creationTime" instead of "ts"
    ts = _parse_timestamp(message.get("ts") or message.get("creationTime")) or received_at

    # Normalise to list
    data_items = data if isinstance(data, list) else [data]

    # For wallet/balance messages, preserve the full data structure in payload
    # so balance_service can parse the coin array correctly
    if subscription.channel_type == "balance":
        # For balance events, wrap the entire data array in the payload
        # This allows balance_service to access data[0].coin[] structure
        payload = {
            "data": data_items,
            "topic": topic,
        }
        # Also include top-level message fields that might be useful
        if "id" in message:
            payload["id"] = message["id"]
        if "creationTime" in message:
            payload["creationTime"] = message["creationTime"]
        
        event = Event.create(
            event_type=subscription.channel_type,
            topic=topic,
            timestamp=ts,
            payload=payload,
            trace_id=trace_id,
        )
        events.append(event)
    else:
        # For other event types, create one event per data item
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

    logger.info(
        "events_parsed",
        topic=topic,
        count=len(events),
        subscription_id=str(subscription.id) if subscription else None,
        channel_type=subscription.channel_type if subscription else None,
        message_ts=message.get("ts"),
        message_op=message.get("op"),
        full_message=message,  # Full message details for debugging
        trace_id=trace_id,
    )
    return events


