"""
Order execution event consumer.

Consumes order execution events from RabbitMQ queue order-manager.order_events,
parses and validates events, and handles corrupted/invalid events with logging
and graceful continuation.
"""

import json
import asyncio
from typing import Callable, Optional, Any, Dict
from datetime import datetime

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger
from ..config.exceptions import MessageQueueError
from ..models.execution_event import OrderExecutionEvent
from ..database.repositories.execution_event_repo import ExecutionEventRepository

logger = get_logger(__name__)


class ExecutionEventConsumer:
    """Consumes order execution events from RabbitMQ queue."""

    def __init__(self, event_callback: Optional[Callable[[OrderExecutionEvent], asyncio.Task]] = None):
        """
        Initialize execution event consumer.

        Args:
            event_callback: Optional callback function to process events
                            (should return an asyncio.Task for async processing)
        """
        self.event_callback = event_callback
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self._queue_name = "order-manager.order_events"
        self.execution_event_repo = ExecutionEventRepository()

    async def start(self) -> None:
        """Start consuming execution events from RabbitMQ queue."""
        if self._running:
            logger.warning("Execution event consumer already running")
            return

        self._running = True
        logger.info("Starting execution event consumer", queue=self._queue_name)

        try:
            self._consumer_task = asyncio.create_task(self._consume_queue())
            logger.info("Execution event consumer started", queue=self._queue_name)
        except Exception as e:
            logger.error("Failed to start execution event consumer", queue=self._queue_name, error=str(e), exc_info=True)
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop consuming execution events."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping execution event consumer", queue=self._queue_name)

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            logger.info("Execution event consumer stopped", queue=self._queue_name)

    async def _consume_queue(self) -> None:
        """Consume messages from the execution events queue with reconnection logic."""
        from ..config.retry import retry_async

        max_reconnect_attempts = 5
        reconnect_delay = 2.0

        while self._running:
            try:
                async def _connect_and_consume():
                    channel = await rabbitmq_manager.get_channel()
                    # Use existing queue without redeclaring (queues are created by order-manager)
                    # Just bind to the existing queue
                    queue = await channel.declare_queue(self._queue_name, durable=True, passive=True)
                    return queue

                queue = await retry_async(
                    _connect_and_consume,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=5.0,
                    operation_name="connect_to_queue",
                )

                logger.info("Connected to execution events queue", queue=self._queue_name)

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        if not self._running:
                            break
                        try:
                            async with message.process():
                                await self._process_message(message)
                        except Exception as e:
                            logger.error(
                                "Error processing execution event message",
                                queue=self._queue_name,
                                error=str(e),
                                exc_info=True,
                            )
                            # Continue processing other messages - don't let one bad message stop the consumer

            except asyncio.CancelledError:
                logger.info("Execution event consumer cancelled", queue=self._queue_name)
                raise
            except Exception as e:
                if not self._running:
                    break

                logger.error(
                    "Execution event consumer error, attempting reconnection",
                    queue=self._queue_name,
                    error=str(e),
                    exc_info=True,
                )

                # Attempt reconnection with exponential backoff
                for attempt in range(max_reconnect_attempts):
                    if not self._running:
                        break

                    try:
                        await asyncio.sleep(reconnect_delay * (2 ** attempt))
                        logger.info(
                            "Reconnecting to execution events queue",
                            queue=self._queue_name,
                            attempt=attempt + 1,
                        )
                        break  # Exit retry loop to attempt reconnection
                    except asyncio.CancelledError:
                        raise
                else:
                    # All reconnection attempts failed
                    logger.error(
                        "Failed to reconnect to execution events queue after all attempts",
                        queue=self._queue_name,
                        max_attempts=max_reconnect_attempts,
                    )
                    raise

    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """
        Process a single execution event message.

        Args:
            message: Incoming message from RabbitMQ
        """
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            # Validate and parse the execution event
            execution_event = self._validate_and_parse_event(data)

            if execution_event:
                # Persist execution event to database
                try:
                    await self._persist_execution_event(execution_event)
                except Exception as e:
                    logger.error(
                        "Failed to persist execution event to database",
                        event_id=execution_event.event_id,
                        error=str(e),
                        exc_info=True,
                    )
                    # Continue processing even if persistence fails

                # Process the event via callback if provided
                if self.event_callback:
                    try:
                        # Callback should handle async processing
                        await self.event_callback(execution_event)
                    except Exception as e:
                        logger.error(
                            "Error in execution event callback",
                            event_id=execution_event.event_id,
                            error=str(e),
                            exc_info=True,
                        )
                else:
                    logger.debug("Execution event received (no callback)", event_id=execution_event.event_id)

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse execution event JSON",
                queue=self._queue_name,
                error=str(e),
                body_preview=body[:200] if len(body) > 200 else body,
            )
        except Exception as e:
            logger.error(
                "Error processing execution event message",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )

    def _validate_and_parse_event(self, data: Dict[str, Any]) -> Optional[OrderExecutionEvent]:
        """
        Validate and parse execution event from dictionary.

        Args:
            data: Raw event data from message queue

        Returns:
            Parsed OrderExecutionEvent or None if validation fails
        """
        try:
            # Check for required top-level fields
            required_fields = [
                "order_id",
                "signal_id",
                "strategy_id",
                "asset",
                "side",
                "execution_price",
                "execution_quantity",
                "execution_fees",
                "executed_at",
                "signal_price",
                "signal_timestamp",
            ]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.warning(
                    "Execution event missing required fields",
                    missing_fields=missing_fields,
                    event_data_keys=list(data.keys()),
                )
                return None

            # Validate field types and ranges
            if not isinstance(data["execution_price"], (int, float)) or data["execution_price"] <= 0:
                logger.warning("Invalid execution_price", execution_price=data.get("execution_price"))
                return None

            if not isinstance(data["execution_quantity"], (int, float)) or data["execution_quantity"] <= 0:
                logger.warning("Invalid execution_quantity", execution_quantity=data.get("execution_quantity"))
                return None

            if not isinstance(data["execution_fees"], (int, float)) or data["execution_fees"] < 0:
                logger.warning("Invalid execution_fees", execution_fees=data.get("execution_fees"))
                return None

            if data["side"].lower() not in ("buy", "sell"):
                logger.warning("Invalid side", side=data.get("side"))
                return None

            # Validate market_conditions
            if "market_conditions" not in data:
                logger.warning("Execution event missing market_conditions")
                return None

            market_conditions = data["market_conditions"]
            required_market_fields = ["spread", "volume_24h", "volatility"]
            missing_market_fields = [field for field in required_market_fields if field not in market_conditions]
            if missing_market_fields:
                logger.warning(
                    "Execution event market_conditions missing required fields",
                    missing_fields=missing_market_fields,
                )
                return None

            # Validate performance
            if "performance" not in data:
                logger.warning("Execution event missing performance")
                return None

            performance = data["performance"]
            required_performance_fields = ["slippage", "slippage_percent"]
            missing_performance_fields = [field for field in required_performance_fields if field not in performance]
            if missing_performance_fields:
                logger.warning(
                    "Execution event performance missing required fields",
                    missing_fields=missing_performance_fields,
                )
                return None

            # Parse datetime strings if needed
            if isinstance(data.get("executed_at"), str):
                try:
                    data["executed_at"] = datetime.fromisoformat(data["executed_at"].replace("Z", "+00:00"))
                except ValueError as e:
                    logger.warning("Invalid executed_at format", executed_at=data.get("executed_at"), error=str(e))
                    return None

            if isinstance(data.get("signal_timestamp"), str):
                try:
                    data["signal_timestamp"] = datetime.fromisoformat(data["signal_timestamp"].replace("Z", "+00:00"))
                except ValueError as e:
                    logger.warning(
                        "Invalid signal_timestamp format", signal_timestamp=data.get("signal_timestamp"), error=str(e)
                    )
                    return None

            # Create OrderExecutionEvent using from_dict
            execution_event = OrderExecutionEvent.from_dict(data)

            logger.debug("Execution event validated and parsed", event_id=execution_event.event_id)
            return execution_event

        except ValueError as e:
            logger.warning("Execution event validation failed", error=str(e), event_data_keys=list(data.keys()))
            return None
        except Exception as e:
            logger.error("Error validating execution event", error=str(e), exc_info=True)
            return None

    async def _persist_execution_event(self, execution_event: OrderExecutionEvent) -> None:
        """
        Persist execution event to PostgreSQL database.

        Args:
            execution_event: Validated execution event to persist

        Note:
            This method handles database errors gracefully and continues processing
            even if persistence fails, as per T084 requirements.
        """
        try:
            await self.execution_event_repo.create(
                signal_id=execution_event.signal_id,
                strategy_id=execution_event.strategy_id,
                asset=execution_event.asset,
                side=execution_event.side,
                execution_price=execution_event.execution_price,
                execution_quantity=execution_event.execution_quantity,
                execution_fees=execution_event.execution_fees,
                executed_at=execution_event.executed_at,
                signal_price=execution_event.signal_price,
                signal_timestamp=execution_event.signal_timestamp,
                performance={
                    "slippage": execution_event.performance.slippage,
                    "slippage_percent": execution_event.performance.slippage_percent,
                    "realized_pnl": execution_event.performance.realized_pnl,
                    "return_percent": execution_event.performance.return_percent,
                },
            )
            logger.debug("Execution event persisted to database", event_id=execution_event.event_id, signal_id=execution_event.signal_id)
        except Exception as e:
            # Log error but don't raise - continue processing on persistence failures
            logger.warning(
                "Failed to persist execution event (continuing)",
                event_id=execution_event.event_id,
                error=str(e),
            )

