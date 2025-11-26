"""Queue publisher service using aio-pika."""

import json
from datetime import datetime
from typing import Optional

import aio_pika
from aio_pika import Message, DeliveryMode

from ...config.logging import get_logger
from ...config.settings import settings
from ...exceptions import QueueError
from ...models.event import Event
from .connection import QueueConnection

logger = get_logger(__name__)


class QueuePublisher:
    """Publishes events to RabbitMQ queues organized by event class."""

    def __init__(self):
        """Initialize queue publisher."""
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._queues: dict[str, aio_pika.abc.AbstractQueue] = {}

    async def initialize(self) -> None:
        """Initialize publisher with channel and ensure queues exist."""
        try:
            self._channel = await QueueConnection.get_channel()
            logger.info("queue_publisher_initialized")
        except Exception as e:
            logger.error(
                "queue_publisher_initialization_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise QueueError(f"Failed to initialize queue publisher: {e}") from e

    async def _ensure_queue(self, queue_name: str) -> aio_pika.abc.AbstractQueue:
        """Ensure queue exists, creating it if necessary."""
        if queue_name in self._queues:
            return self._queues[queue_name]

        if not self._channel:
            await self.initialize()

        try:
            # Queue will be created by setup module, but we ensure it exists here
            # This is a fallback in case setup wasn't called
            queue = await self._channel.declare_queue(
                queue_name,
                durable=True,
            )
            self._queues[queue_name] = queue
            logger.debug("queue_ensured", queue_name=queue_name)
            return queue
        except Exception as e:
            logger.error(
                "queue_ensure_failed",
                queue_name=queue_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise QueueError(f"Failed to ensure queue {queue_name}: {e}") from e

    async def publish_event(self, event: Event, queue_name: str) -> bool:
        """
        Publish an event to the specified queue.

        Args:
            event: Event to publish
            queue_name: Name of the target queue

        Returns:
            True if published successfully, False otherwise

        Note:
            Per requirement FR-017, queue publishing failures should not block
            event processing. This method logs errors and returns False on failure.
        """
        if not self._channel or self._channel.is_closed:
            try:
                await self.initialize()
            except Exception as e:
                logger.error(
                    "queue_publish_connection_failed",
                    queue_name=queue_name,
                    event_id=str(event.event_id),
                    event_type=event.event_type,
                    trace_id=event.trace_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

        try:
            queue = await self._ensure_queue(queue_name)

            # Serialize event to JSON
            event_data = {
                "event_id": str(event.event_id),
                "event_type": event.event_type,
                "topic": event.topic,
                "timestamp": event.timestamp.isoformat(),
                "received_at": event.received_at.isoformat(),
                "payload": event.payload,
                "trace_id": event.trace_id,
            }

            message_body = json.dumps(event_data).encode("utf-8")

            # Create message with durability
            message = Message(
                message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                timestamp=datetime.utcnow(),
                headers={
                    "event_type": event.event_type,
                    "topic": event.topic,
                    "trace_id": event.trace_id,
                },
            )

            # Publish to queue
            await self._channel.default_exchange.publish(
                message,
                routing_key=queue_name,
            )

            logger.info(
                "queue_event_published",
                queue_name=queue_name,
                event_id=str(event.event_id),
                event_type=event.event_type,
                topic=event.topic,
                trace_id=event.trace_id,
            )
            return True

        except Exception as e:
            # Per T053: Handle queue connection failures gracefully (log and continue)
            logger.error(
                "queue_publish_failed",
                queue_name=queue_name,
                event_id=str(event.event_id),
                event_type=event.event_type,
                topic=event.topic,
                trace_id=event.trace_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def close(self) -> None:
        """Close publisher resources."""
        self._queues.clear()
        self._channel = None
        logger.info("queue_publisher_closed")


# Global publisher instance
_publisher: Optional[QueuePublisher] = None


async def get_publisher() -> QueuePublisher:
    """Get or create global queue publisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = QueuePublisher()
        await _publisher.initialize()
    return _publisher


async def close_publisher() -> None:
    """Close global queue publisher."""
    global _publisher
    if _publisher:
        await _publisher.close()
        _publisher = None

