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
from ..database.connection import db_pool
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
                    # Declare queue - create if it doesn't exist (will be created by order-manager publisher later)
                    # We declare it here so consumer can start even if no events have been published yet
                    queue = await channel.declare_queue(self._queue_name, durable=True, passive=False)
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

                # Check if error is due to missing queue (expected when order-manager not started)
                error_str = str(e)
                error_type_name = type(e).__name__
                is_queue_not_found = (
                    "no queue" in error_str.lower()
                    or "NOT_FOUND" in error_str.upper()
                    or "ChannelNotFoundEntity" in error_type_name
                    or "ChannelNotFoundEntity" in error_str
                )

                if is_queue_not_found:
                    # Queue doesn't exist yet - this is expected when order-manager isn't running
                    # Log as warning instead of error, with less frequent logging
                    logger.warning(
                        "Execution events queue not found (order-manager may not be started yet), will retry",
                        queue=self._queue_name,
                    )
                    # Wait longer before retrying for missing queue
                    await asyncio.sleep(reconnect_delay * 5)  # Wait 10 seconds before retrying
                    continue  # Continue loop to retry connection

                # For other errors, log as error and retry with exponential backoff
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

            # Transform order-manager format to expected format
            transformed_data = await self._transform_order_manager_event(data)
            if not transformed_data:
                logger.warning("Failed to transform order-manager event", event_data_keys=list(data.keys()))
                return

            # Validate and parse the execution event
            execution_event = self._validate_and_parse_event(transformed_data)

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

    async def _transform_order_manager_event(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform order-manager event format to expected execution event format.

        Args:
            data: Raw event data from order-manager

        Returns:
            Transformed event data or None if transformation fails
        """
        try:
            # Check if this is order-manager format (has 'order' key)
            if "order" not in data:
                # Already in expected format or unknown format
                return data

            order_data = data.get("order", {})
            signal_id = order_data.get("signal_id")
            if not signal_id:
                logger.warning("Order event missing signal_id", event_data_keys=list(data.keys()))
                return None

            # Get signal information from database
            signal_info = await self._get_signal_info(signal_id)
            if not signal_info:
                logger.warning("Signal not found in database", signal_id=signal_id)
                return None

            # Extract order execution details
            order_id = order_data.get("order_id")
            asset = order_data.get("asset")
            side = order_data.get("side", "").lower()
            filled_quantity = float(order_data.get("filled_quantity", "0"))
            average_price = float(order_data.get("average_price", "0")) if order_data.get("average_price") else None
            fees = float(order_data.get("fees", "0")) if order_data.get("fees") else 0.0
            executed_at_str = order_data.get("executed_at")

            if not average_price or average_price <= 0:
                logger.warning("Order event missing or invalid average_price", order_id=order_id)
                return None

            if filled_quantity <= 0:
                logger.warning("Order event has zero or negative filled_quantity", order_id=order_id)
                return None

            # Parse executed_at
            executed_at = None
            if executed_at_str:
                try:
                    executed_at = datetime.fromisoformat(executed_at_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    executed_at = datetime.utcnow()
            else:
                executed_at = datetime.utcnow()

            # Get signal price and timestamp
            signal_price = float(signal_info.get("price", "0"))
            signal_timestamp_str = signal_info.get("timestamp")
            signal_timestamp = None
            if signal_timestamp_str:
                try:
                    signal_timestamp = datetime.fromisoformat(signal_timestamp_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    signal_timestamp = executed_at  # Fallback to executed_at
            else:
                signal_timestamp = executed_at

            if signal_price <= 0:
                logger.warning("Signal has invalid price", signal_id=signal_id, price=signal_price)
                return None

            # Calculate slippage
            slippage = average_price - signal_price
            slippage_percent = (slippage / signal_price * 100) if signal_price > 0 else 0.0

            # Get market conditions from event or use defaults
            market_conditions = data.get("market_conditions", {})
            if not market_conditions or not all(k in market_conditions for k in ["spread", "volume_24h", "volatility"]):
                # Use defaults if not provided
                market_conditions = {
                    "spread": 0.0015,  # Default 0.15%
                    "volume_24h": 1000000.0,  # Default volume
                    "volatility": 0.02,  # Default 2%
                }
                logger.debug("Using default market conditions", order_id=order_id)

            # Build transformed event
            transformed = {
                "order_id": order_id,
                "signal_id": signal_id,
                "strategy_id": signal_info.get("strategy_id", "unknown"),
                "asset": asset,
                "side": side,
                "execution_price": average_price,
                "execution_quantity": filled_quantity,
                "execution_fees": fees,
                "executed_at": executed_at.isoformat() + "Z",
                "signal_price": signal_price,
                "signal_timestamp": signal_timestamp.isoformat() + "Z",
                "market_conditions": {
                    "spread": float(market_conditions.get("spread", 0.0015)),
                    "volume_24h": float(market_conditions.get("volume_24h", 1000000.0)),
                    "volatility": float(market_conditions.get("volatility", 0.02)),
                },
                "performance": {
                    "slippage": slippage,
                    "slippage_percent": slippage_percent,
                    "realized_pnl": None,  # Will be calculated later when position is closed
                    "return_percent": None,  # Will be calculated later when position is closed
                },
                "trace_id": data.get("trace_id"),
            }

            logger.debug("Transformed order-manager event", order_id=order_id, signal_id=signal_id)
            return transformed

        except Exception as e:
            logger.error("Error transforming order-manager event", error=str(e), exc_info=True)
            return None

    async def _get_signal_info(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get signal information from database.

        Args:
            signal_id: Signal identifier

        Returns:
            Signal information dictionary or None if not found
        """
        try:
            pool = await db_pool.get_pool()
            query = """
                SELECT signal_id, strategy_id, price, timestamp
                FROM trading_signals
                WHERE signal_id = $1
                LIMIT 1
            """
            from uuid import UUID
            signal_uuid = UUID(signal_id) if isinstance(signal_id, str) else signal_id
            row = await pool.fetchrow(query, signal_uuid)

            if row:
                return {
                    "signal_id": str(row["signal_id"]),
                    "strategy_id": row["strategy_id"],
                    "price": str(row["price"]),
                    "timestamp": row["timestamp"].isoformat() + "Z" if row["timestamp"] else None,
                }
            return None

        except Exception as e:
            logger.error("Error querying signal info", signal_id=signal_id, error=str(e), exc_info=True)
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

