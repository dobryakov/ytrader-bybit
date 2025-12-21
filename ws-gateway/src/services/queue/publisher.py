"""Queue publisher service using aio-pika."""

import asyncio
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
from .monitoring import record_publish
from .setup import ensure_queue_exists

logger = get_logger(__name__)

# Retry configuration
PUBLISH_MAX_RETRIES = 3
PUBLISH_INITIAL_DELAY = 0.1  # 100ms
PUBLISH_MAX_DELAY = 1.0  # 1 second
PUBLISH_BACKOFF_MULTIPLIER = 2.0

ENSURE_QUEUE_MAX_RETRIES = 5
ENSURE_QUEUE_INITIAL_DELAY = 0.2  # 200ms
ENSURE_QUEUE_MAX_DELAY = 2.0  # 2 seconds
ENSURE_QUEUE_BACKOFF_MULTIPLIER = 2.0


class QueuePublisher:
    """Publishes events to RabbitMQ queues organized by event class."""

    def __init__(self):
        """Initialize queue publisher."""
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._queues: dict[str, aio_pika.abc.AbstractQueue] = {}

    def _is_channel_error(self, error: Exception) -> bool:
        """Check if error indicates channel needs to be recovered."""
        return isinstance(
            error,
            (
                aio_pika.exceptions.ChannelClosed,
                aio_pika.exceptions.ChannelNotFoundEntity,
                aio_pika.exceptions.ChannelInvalidStateError,
            ),
        ) or "Channel closed" in str(error) or "RPC timeout" in str(error)

    async def _recover_channel(self) -> None:
        """Recover channel after error: invalidate and clear cached queues."""
        logger.warning("queue_channel_recovery_started")
        self._channel = None
        self._queues.clear()
        QueueConnection.invalidate_channel()
        try:
            self._channel = await QueueConnection.get_channel()
            logger.info("queue_channel_recovered")
        except Exception as e:
            logger.error(
                "queue_channel_recovery_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

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
        """Ensure queue exists, creating it if necessary with retry logic."""
        if queue_name in self._queues:
            return self._queues[queue_name]

        delay = ENSURE_QUEUE_INITIAL_DELAY
        last_error = None

        for attempt in range(1, ENSURE_QUEUE_MAX_RETRIES + 1):
            should_retry = False
            try:
                if not self._channel or self._channel.is_closed:
                    await self.initialize()

                # First, try to get existing queue with passive=True
                # This avoids conflicts if queue already exists with TTL
                queue = await self._channel.declare_queue(
                    queue_name,
                    durable=True,
                    passive=True,  # Only check if queue exists, don't create
                )
                self._queues[queue_name] = queue
                logger.debug("queue_ensured_passive", queue_name=queue_name)
                return queue
            except (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.ChannelClosed) as e:
                # Check if this is a "queue not found" error vs channel error
                # When using passive=True, "NOT_FOUND - no queue" means queue doesn't exist
                error_str = str(e).upper()
                is_queue_not_found = "NO QUEUE" in error_str or "NOT_FOUND" in error_str
                
                if is_queue_not_found:
                    # Queue doesn't exist - treat as "Other errors" to create it
                    last_error = e
                    logger.debug(
                        "queue_ensure_other_error",
                        queue_name=queue_name,
                        attempt=attempt,
                        error=str(e),
                    )
                    # Extract event_type from queue_name (format: ws-gateway.{event_type})
                    event_type = queue_name.replace("ws-gateway.", "")
                    try:
                        await ensure_queue_exists(event_type)
                        # Queue should exist now, retry getting it
                        should_retry = True
                    except Exception as create_error:
                        logger.warning(
                            "queue_ensure_create_failed",
                            queue_name=queue_name,
                            attempt=attempt,
                            error=str(create_error),
                        )
                        should_retry = True
                else:
                    # Channel error - recover and retry
                    last_error = e
                    logger.warning(
                        "queue_ensure_channel_error",
                        queue_name=queue_name,
                        attempt=attempt,
                        max_retries=ENSURE_QUEUE_MAX_RETRIES,
                        error=str(e),
                    )
                    try:
                        await self._recover_channel()
                        should_retry = True
                    except Exception as recover_error:
                        logger.error(
                            "queue_ensure_channel_recovery_failed",
                            queue_name=queue_name,
                            error=str(recover_error),
                        )
                        if attempt == ENSURE_QUEUE_MAX_RETRIES:
                            raise QueueError(f"Failed to recover channel after {attempt} attempts: {recover_error}") from recover_error
                        should_retry = True
            except Exception as e:
                # Other errors (e.g., queue doesn't exist) - try to create it
                last_error = e
                logger.debug(
                    "queue_ensure_other_error",
                    queue_name=queue_name,
                    attempt=attempt,
                    error=str(e),
                )
                # Extract event_type from queue_name (format: ws-gateway.{event_type})
                event_type = queue_name.replace("ws-gateway.", "")
                try:
                    await ensure_queue_exists(event_type)
                    # Queue should exist now, retry getting it
                    should_retry = True
                except Exception as create_error:
                    logger.warning(
                        "queue_ensure_create_failed",
                        queue_name=queue_name,
                        attempt=attempt,
                        error=str(create_error),
                    )
                    should_retry = True

            # Wait before next retry if needed
            if should_retry and attempt < ENSURE_QUEUE_MAX_RETRIES:
                logger.debug(
                    "queue_ensure_retry",
                    queue_name=queue_name,
                    attempt=attempt,
                    next_attempt_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * ENSURE_QUEUE_BACKOFF_MULTIPLIER, ENSURE_QUEUE_MAX_DELAY)

        # All retries exhausted
        error_msg = f"Failed to ensure queue {queue_name} after {ENSURE_QUEUE_MAX_RETRIES} attempts"
        logger.error(
            "queue_ensure_final_failed",
            queue_name=queue_name,
            error=str(last_error) if last_error else "Unknown error",
            error_type=type(last_error).__name__ if last_error else "Unknown",
        )
        raise QueueError(error_msg) from last_error

    async def publish_event(self, event: Event, queue_name: str) -> bool:
        """
        Publish an event to the specified queue with retry logic.

        Args:
            event: Event to publish
            queue_name: Name of the target queue

        Returns:
            True if published successfully, False otherwise

        Note:
            Per requirement FR-017, queue publishing failures should not block
            event processing. This method logs errors and returns False on failure.
        """
        delay = PUBLISH_INITIAL_DELAY
        last_error = None

        for attempt in range(1, PUBLISH_MAX_RETRIES + 1):
            try:
                # Ensure channel is available
                if not self._channel or self._channel.is_closed:
                    await self.initialize()

                # Ensure queue exists
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

                # Record publish for backlog monitoring (EC7: Track publish rate)
                record_publish(event.event_type, message_count=1)

                return True

            except Exception as e:
                last_error = e

                # Check if this is a channel error that can be recovered
                if self._is_channel_error(e):
                    logger.warning(
                        "queue_publish_channel_error",
                        queue_name=queue_name,
                        event_id=str(event.event_id),
                        event_type=event.event_type,
                        trace_id=event.trace_id,
                        attempt=attempt,
                        max_retries=PUBLISH_MAX_RETRIES,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                    # Try to recover channel
                    try:
                        await self._recover_channel()
                    except Exception as recover_error:
                        logger.error(
                            "queue_publish_channel_recovery_failed",
                            queue_name=queue_name,
                            event_id=str(event.event_id),
                            error=str(recover_error),
                        )
                        # If recovery fails and this is the last attempt, give up
                        if attempt == PUBLISH_MAX_RETRIES:
                            break

                    # Wait before retry
                    if attempt < PUBLISH_MAX_RETRIES:
                        await asyncio.sleep(delay)
                        delay = min(delay * PUBLISH_BACKOFF_MULTIPLIER, PUBLISH_MAX_DELAY)
                        continue
                else:
                    # Non-channel error - log and give up (no retry for other errors)
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
                    # Per T053: Handle queue connection failures gracefully (log and continue)
                    return False

        # All retries exhausted for channel errors
        logger.error(
            "queue_publish_failed_after_retries",
            queue_name=queue_name,
            event_id=str(event.event_id),
            event_type=event.event_type,
            topic=event.topic,
            trace_id=event.trace_id,
            attempts=PUBLISH_MAX_RETRIES,
            error=str(last_error) if last_error else "Unknown error",
            error_type=type(last_error).__name__ if last_error else "Unknown",
        )
        # Per T053: Handle queue connection failures gracefully (log and continue)
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

