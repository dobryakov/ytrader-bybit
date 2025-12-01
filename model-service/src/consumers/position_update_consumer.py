"""
Position update event consumer.

Consumes position update events from RabbitMQ queue position-manager.position_updated,
parses events, and invalidates local cache for affected assets.
"""

import json
import asyncio
from typing import Optional, Any, Dict

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
import aio_pika.exceptions

from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger
from ..services.position_cache import position_cache

logger = get_logger(__name__)


class PositionUpdateConsumer:
    """Consumes position update events from RabbitMQ queue and invalidates cache."""

    def __init__(self):
        """Initialize position update consumer."""
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self._queue_name = "position-manager.position_updated"

    async def start(self) -> None:
        """Start consuming position update events from RabbitMQ queue."""
        if self._running:
            logger.warning("Position update consumer already running")
            return

        self._running = True
        logger.info("Starting position update consumer", queue=self._queue_name)

        try:
            self._consumer_task = asyncio.create_task(self._consume_queue())
            logger.info("Position update consumer started", queue=self._queue_name)
        except Exception as e:
            logger.error(
                "Failed to start position update consumer",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop consuming position update events."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping position update consumer", queue=self._queue_name)

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            logger.info("Position update consumer stopped", queue=self._queue_name)

    async def _consume_queue(self) -> None:
        """Consume messages from the position update queue with reconnection logic."""
        max_reconnect_attempts = 5
        reconnect_delay = 2.0

        while self._running:
            try:
                channel = await rabbitmq_manager.get_channel()
                # Declare queue - create if it doesn't exist
                try:
                    queue = await channel.declare_queue(self._queue_name, durable=True, passive=False)
                except Exception as e:
                    # Check if error is due to connection issue (not missing queue)
                    error_str = str(e)
                    error_type_name = type(e).__name__
                    is_queue_not_found = (
                        "no queue" in error_str.lower()
                        or "NOT_FOUND" in error_str.upper()
                        or "ChannelNotFoundEntity" in error_type_name
                        or "ChannelNotFoundEntity" in error_str
                    )

                    if is_queue_not_found:
                        # Queue doesn't exist yet - this is expected when position-manager isn't running
                        logger.warning(
                            "Position update queue not found (position-manager may not be started yet), will retry",
                            queue=self._queue_name,
                        )
                        await asyncio.sleep(reconnect_delay * 5)  # Wait 10 seconds before retrying
                        continue  # Continue loop to retry connection
                    else:
                        # Other connection errors - retry with exponential backoff
                        raise

                logger.info("Connected to position update queue", queue=self._queue_name)

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            break
                        try:
                            async with message.process():
                                await self._process_message(message)
                        except Exception as e:
                            logger.error(
                                "Error processing position update message",
                                queue=self._queue_name,
                                error=str(e),
                                exc_info=True,
                            )
                            # Continue processing other messages - don't let one bad message stop the consumer

            except asyncio.CancelledError:
                logger.info("Position update consumer cancelled", queue=self._queue_name)
                raise
            except Exception as e:
                if not self._running:
                    break

                # Check if error is due to missing queue or connection issue
                error_str = str(e)
                error_type_name = type(e).__name__
                is_queue_not_found = (
                    "no queue" in error_str.lower()
                    or "NOT_FOUND" in error_str.upper()
                    or "ChannelNotFoundEntity" in error_type_name
                    or "ChannelNotFoundEntity" in error_str
                )
                is_connection_error = (
                    "ConnectionClosed" in error_type_name
                    or "ConnectionClosed" in error_str
                    or isinstance(e, (aio_pika.exceptions.ChannelNotFoundEntity, aio_pika.exceptions.ConnectionClosed))
                )

                if is_queue_not_found or is_connection_error:
                    # Queue doesn't exist yet or connection lost - this is expected when position-manager isn't running
                    if is_queue_not_found:
                        logger.warning(
                            "Position update queue not found (position-manager may not be started yet), will retry",
                            queue=self._queue_name,
                        )
                    else:
                        logger.warning(
                            "Connection lost, will retry",
                            queue=self._queue_name,
                            error_type=error_type_name,
                        )
                    await asyncio.sleep(reconnect_delay * 5)  # Wait 10 seconds before retrying
                    continue  # Continue loop to retry connection

                # For other errors, log as error and retry with exponential backoff
                logger.error(
                    "Position update consumer error, attempting reconnection",
                    queue=self._queue_name,
                    error=str(e),
                    error_type=error_type_name,
                    exc_info=True,
                )

                # Attempt reconnection with exponential backoff
                for attempt in range(max_reconnect_attempts):
                    if not self._running:
                        break

                    try:
                        await asyncio.sleep(reconnect_delay * (2 ** attempt))
                        logger.info(
                            "Reconnecting to position update queue",
                            queue=self._queue_name,
                            attempt=attempt + 1,
                        )
                        break  # Exit retry loop to attempt reconnection
                    except asyncio.CancelledError:
                        raise
                else:
                    # All reconnection attempts failed
                    logger.error(
                        "Failed to reconnect to position update queue after all attempts",
                        queue=self._queue_name,
                        max_attempts=max_reconnect_attempts,
                    )
                    await asyncio.sleep(reconnect_delay * 5)  # Wait before next retry cycle
                    continue  # Continue loop instead of raising

    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """
        Process a single position update message.

        Args:
            message: Incoming message from RabbitMQ
        """
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            # Extract asset from event
            asset = data.get("asset") or data.get("symbol") or data.get("trading_pair")
            trace_id = data.get("trace_id") or data.get("traceId")

            if not asset:
                logger.warning(
                    "Position update event missing asset field",
                    queue=self._queue_name,
                    event_keys=list(data.keys()),
                )
                return

            # Invalidate cache for the affected asset
            await position_cache.invalidate(asset)
            logger.info(
                "Cache invalidated for position update",
                asset=asset,
                trace_id=trace_id,
            )

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse position update JSON",
                queue=self._queue_name,
                error=str(e),
                body_preview=body[:200] if len(body) > 200 else body,
            )
        except Exception as e:
            logger.error(
                "Error processing position update message",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )


# Global position update consumer instance
position_update_consumer = PositionUpdateConsumer()

