"""
Dataset ready notification consumer.

Consumes dataset completion notifications from RabbitMQ queue features.dataset.ready,
parses dataset completion notifications, and triggers model training when dataset is ready.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from uuid import UUID

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
import aio_pika.exceptions

from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger

logger = get_logger(__name__)


class DatasetReadyConsumer:
    """Consumes dataset ready notifications from RabbitMQ queue."""

    def __init__(self, dataset_ready_callback: Optional[callable] = None):
        """
        Initialize dataset ready consumer.

        Args:
            dataset_ready_callback: Optional callback function to handle dataset ready notifications
                                   Should accept (dataset_id: UUID, symbol: str, trace_id: Optional[str])
        """
        self.dataset_ready_callback = dataset_ready_callback
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self._queue_name = "features.dataset.ready"

    async def start(self) -> None:
        """Start consuming dataset ready notifications from RabbitMQ queue."""
        if self._running:
            logger.warning("Dataset ready consumer already running")
            return

        self._running = True
        logger.info("Starting dataset ready consumer", queue=self._queue_name)

        try:
            self._consumer_task = asyncio.create_task(self._consume_queue())
            logger.info("Dataset ready consumer started", queue=self._queue_name)
        except Exception as e:
            logger.error(
                "Failed to start dataset ready consumer",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop consuming dataset ready notifications."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping dataset ready consumer", queue=self._queue_name)

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            logger.info("Dataset ready consumer stopped", queue=self._queue_name)

    async def _consume_queue(self) -> None:
        """Consume messages from the dataset ready queue with reconnection logic."""
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
                        # Queue doesn't exist yet - this is expected when feature-service isn't running
                        logger.warning(
                            "Dataset ready queue not found (feature-service may not be started yet), will retry",
                            queue=self._queue_name,
                        )
                        await asyncio.sleep(reconnect_delay * 5)  # Wait 10 seconds before retrying
                        continue  # Continue loop to retry connection
                    else:
                        # Other connection errors - retry with exponential backoff
                        raise

                logger.info("Connected to dataset ready queue", queue=self._queue_name)

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            break
                        try:
                            async with message.process():
                                await self._process_message(message)
                        except Exception as e:
                            logger.error(
                                "Error processing dataset ready message",
                                queue=self._queue_name,
                                error=str(e),
                                exc_info=True,
                            )
                            # Continue processing other messages - don't let one bad message stop the consumer

            except asyncio.CancelledError:
                logger.info("Dataset ready consumer cancelled", queue=self._queue_name)
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
                    # Queue doesn't exist yet or connection lost
                    if is_queue_not_found:
                        logger.warning(
                            "Dataset ready queue not found (feature-service may not be started yet), will retry",
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
                    "Dataset ready consumer error, attempting reconnection",
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
                            "Reconnecting to dataset ready queue",
                            queue=self._queue_name,
                            attempt=attempt + 1,
                        )
                        break  # Exit retry loop to attempt reconnection
                    except asyncio.CancelledError:
                        raise
                else:
                    # All reconnection attempts failed
                    logger.error(
                        "Failed to reconnect to dataset ready queue after all attempts",
                        queue=self._queue_name,
                        max_attempts=max_reconnect_attempts,
                    )
                    await asyncio.sleep(reconnect_delay * 5)  # Wait before next retry cycle
                    continue  # Continue loop instead of raising

    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """
        Process a single dataset ready notification message.

        Args:
            message: Incoming message from RabbitMQ
        """
        body = None
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            # Validate and extract dataset information
            validation_result = self._validate_dataset_ready_event(data)
            if not validation_result["valid"]:
                logger.warning(
                    "Dataset ready event validation failed",
                    queue=self._queue_name,
                    errors=validation_result["errors"],
                    event_keys=list(data.keys()),
                    trace_id=data.get("trace_id"),
                )
                return

            # Extract validated fields
            dataset_id = validation_result["dataset_id"]
            symbol = validation_result.get("symbol")
            trace_id = data.get("trace_id") or data.get("traceId")

            logger.info(
                "Dataset ready notification received",
                dataset_id=str(dataset_id),
                symbol=symbol,
                trace_id=trace_id,
            )

            # Call callback if provided
            if self.dataset_ready_callback:
                try:
                    await self.dataset_ready_callback(dataset_id, symbol, trace_id)
                except Exception as e:
                    logger.error(
                        "Error in dataset ready callback",
                        dataset_id=str(dataset_id),
                        symbol=symbol,
                        error=str(e),
                        trace_id=trace_id,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Dataset ready notification received but no callback configured",
                    dataset_id=str(dataset_id),
                    symbol=symbol,
                    trace_id=trace_id,
                )

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse dataset ready JSON",
                queue=self._queue_name,
                error=str(e),
                body_preview=body[:200] if body and len(body) > 200 else body,
            )
        except Exception as e:
            logger.error(
                "Error processing dataset ready message",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )

    def _validate_dataset_ready_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dataset ready notification event.

        Args:
            data: Parsed event data

        Returns:
            Dictionary with:
                - valid: bool - whether event is valid
                - errors: List[str] - list of validation errors
                - dataset_id: Optional[UUID] - validated dataset_id if valid
                - symbol: Optional[str] - symbol if present
        """
        errors = []
        dataset_id = None
        symbol = data.get("symbol")

        # Required field: dataset_id
        dataset_id_str = data.get("dataset_id") or data.get("datasetId") or data.get("id")
        if not dataset_id_str:
            errors.append("Missing required field: dataset_id (or datasetId or id)")
        else:
            try:
                dataset_id = UUID(str(dataset_id_str))
            except (ValueError, TypeError) as e:
                errors.append(f"Invalid dataset_id format: {dataset_id_str} - {str(e)}")

        # Optional field: status (should be 'ready' for dataset ready notification)
        status = data.get("status")
        if status and status != "ready":
            logger.warning(
                "Dataset ready notification received with non-ready status",
                dataset_id=str(dataset_id) if dataset_id else None,
                status=status,
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "dataset_id": dataset_id if len(errors) == 0 else None,
            "symbol": symbol,
        }

