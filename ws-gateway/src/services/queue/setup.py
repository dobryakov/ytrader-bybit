"""Queue initialization and configuration (durability, retention)."""

from typing import Set

import aio_pika
from aio_pika import ExchangeType

from ...config.logging import get_logger
from ...config.settings import settings
from ...exceptions import QueueError
from ...models.event import EventType
from .connection import QueueConnection

logger = get_logger(__name__)

# Queue retention limits per FR-019
QUEUE_RETENTION_HOURS = 24
QUEUE_RETENTION_MESSAGES = 100_000

# All supported event types
SUPPORTED_EVENT_TYPES: Set[EventType] = {
    "trade",
    "ticker",
    "orderbook",
    "order",
    "balance",
    "position",
    "kline",
    "liquidation",
    "funding",
}


def get_queue_name(event_type: EventType) -> str:
    """
    Get queue name for an event type.

    Queue naming convention: ws-gateway.{event_class}

    Args:
        event_type: Event type

    Returns:
        Queue name following convention ws-gateway.{event_class}
    """
    return f"ws-gateway.{event_type}"


async def setup_queues() -> None:
    """
    Initialize and configure all RabbitMQ queues for event delivery.

    This function:
    1. Creates durable queues for each event type
    2. Configures queue retention limits (24 hours or 100K messages)
    3. Sets up queue durability for reliability

    Per requirement FR-010, queues are organized by event class.
    Per requirement FR-019, queues have retention limits of 24 hours or 100K messages.
    """
    try:
        channel = await QueueConnection.get_channel()

        # Create default exchange if it doesn't exist (RabbitMQ has default exchange by default)
        # We'll use the default exchange for routing

        # Create queues for each event type
        for event_type in SUPPORTED_EVENT_TYPES:
            queue_name = get_queue_name(event_type)

            try:
                # Declare queue with durability
                # Note: RabbitMQ queue arguments for retention:
                # - x-message-ttl: Message TTL in milliseconds (24 hours = 86400000 ms)
                # - x-max-length: Maximum number of messages (100K)
                # We'll use the smaller of the two limits (whichever comes first)
                queue = await channel.declare_queue(
                    queue_name,
                    durable=True,  # Queue survives broker restart
                    arguments={
                        # Message TTL: 24 hours in milliseconds
                        "x-message-ttl": QUEUE_RETENTION_HOURS * 60 * 60 * 1000,
                        # Maximum queue length: 100K messages
                        "x-max-length": QUEUE_RETENTION_MESSAGES,
                        # Overflow behavior: drop oldest messages when limit reached
                        "x-overflow": "drop-head",
                    },
                )

                logger.info(
                    "queue_configured",
                    queue_name=queue_name,
                    event_type=event_type,
                    retention_hours=QUEUE_RETENTION_HOURS,
                    retention_messages=QUEUE_RETENTION_MESSAGES,
                )
            except Exception as e:
                logger.error(
                    "queue_setup_failed",
                    queue_name=queue_name,
                    event_type=event_type,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Continue with other queues even if one fails
                continue

        logger.info(
            "queues_setup_complete",
            queue_count=len(SUPPORTED_EVENT_TYPES),
        )

    except Exception as e:
        logger.error(
            "queue_setup_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise QueueError(f"Failed to setup queues: {e}") from e


async def ensure_queue_exists(event_type: EventType) -> None:
    """
    Ensure a specific queue exists, creating it if necessary.

    Args:
        event_type: Event type to ensure queue for
    """
    if event_type not in SUPPORTED_EVENT_TYPES:
        logger.warning(
            "unsupported_event_type",
            event_type=event_type,
        )
        return

    try:
        channel = await QueueConnection.get_channel()
        queue_name = get_queue_name(event_type)

        await channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-message-ttl": QUEUE_RETENTION_HOURS * 60 * 60 * 1000,
                "x-max-length": QUEUE_RETENTION_MESSAGES,
                "x-overflow": "drop-head",
            },
        )

        logger.debug("queue_ensured", queue_name=queue_name, event_type=event_type)
    except Exception as e:
        logger.error(
            "queue_ensure_failed",
            event_type=event_type,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise QueueError(f"Failed to ensure queue for {event_type}: {e}") from e

