"""
Order execution event consumer.

Consumes order execution events from RabbitMQ queue order-manager.order_events,
parses and validates events, and handles corrupted/invalid events with logging
and graceful continuation.
"""

import json
import asyncio
from typing import Callable, Optional, Any, Dict
from datetime import datetime, timezone

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
import aio_pika.exceptions

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
                channel = await rabbitmq_manager.get_channel()
                # Declare queue - create if it doesn't exist (will be created by order-manager publisher later)
                # We declare it here so consumer can start even if no events have been published yet
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
                        # Queue doesn't exist yet - this is expected when order-manager isn't running
                        # Log as warning and retry after delay
                        logger.warning(
                            "Execution events queue not found (order-manager may not be started yet), will retry",
                            queue=self._queue_name,
                        )
                        await asyncio.sleep(reconnect_delay * 5)  # Wait 10 seconds before retrying
                        continue  # Continue loop to retry connection
                    else:
                        # Other connection errors - retry with exponential backoff
                        raise

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
                    # Queue doesn't exist yet or connection lost - this is expected when order-manager isn't running
                    # Log as warning and retry after delay
                    if is_queue_not_found:
                        logger.warning(
                            "Execution events queue not found (order-manager may not be started yet), will retry",
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
                    "Execution event consumer error, attempting reconnection",
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
                    await asyncio.sleep(reconnect_delay * 5)  # Wait before next retry cycle
                    continue  # Continue loop instead of raising

    async def _process_message(self, message: AbstractIncomingMessage) -> None:
        """
        Process a single execution event message.

        Args:
            message: Incoming message from RabbitMQ
        """
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            # Only process filled events - other events don't have execution details
            event_type = data.get("event_type", "")
            if event_type != "filled":
                logger.debug("Skipping non-filled event", event_type=event_type, event_keys=list(data.keys()))
                return
            
            # Transform order-manager format to expected format
            logger.debug("Processing filled order event", event_keys=list(data.keys()), has_order="order" in data)
            transformed_data = await self._transform_order_manager_event(data)
            if not transformed_data:
                logger.warning("Failed to transform order-manager event", event_data_keys=list(data.keys()), has_order="order" in data)
                return
            logger.debug("Event transformed successfully", transformed_keys=list(transformed_data.keys()))

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
            data: Transformed event data (should be in expected format after transformation)

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
                    "Execution event missing required fields after transformation",
                    missing_fields=missing_fields,
                    event_data_keys=list(data.keys()),
                    has_order_key="order" in data,  # Check if still in order-manager format
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
                    executed_at_str = data["executed_at"]
                    # Handle both Z and +00:00 formats, but avoid double timezone
                    if executed_at_str.endswith("Z"):
                        # Remove Z and add +00:00, but only if not already present
                        if "+00:00" not in executed_at_str:
                            executed_at_str = executed_at_str[:-1] + "+00:00"
                        else:
                            # Already has +00:00, just remove Z
                            executed_at_str = executed_at_str[:-1]
                    data["executed_at"] = datetime.fromisoformat(executed_at_str)
                except ValueError as e:
                    logger.warning("Invalid executed_at format", executed_at=data.get("executed_at"), error=str(e))
                    return None

            if isinstance(data.get("signal_timestamp"), str):
                try:
                    signal_timestamp_str = data["signal_timestamp"]
                    # Handle both Z and +00:00 formats, but avoid double timezone
                    if signal_timestamp_str.endswith("Z"):
                        # Remove Z and add +00:00, but only if not already present
                        if "+00:00" not in signal_timestamp_str:
                            signal_timestamp_str = signal_timestamp_str[:-1] + "+00:00"
                        else:
                            # Already has +00:00, just remove Z
                            signal_timestamp_str = signal_timestamp_str[:-1]
                    data["signal_timestamp"] = datetime.fromisoformat(signal_timestamp_str)
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

            # Extract order execution details first (needed for fallback)
            order_id = order_data.get("order_id")
            asset = order_data.get("asset")
            side = order_data.get("side", "").lower()
            event_type = data.get("event_type", "")
            
            # Log event details for debugging
            logger.debug(
                "Transforming order event",
                order_id=order_id,
                event_type=event_type,
                order_keys=list(order_data.keys()),
                average_price_raw=order_data.get("average_price"),
                filled_quantity_raw=order_data.get("filled_quantity"),
            )
            
            filled_quantity = float(order_data.get("filled_quantity", "0"))
            average_price_raw = order_data.get("average_price")
            average_price = None
            if average_price_raw:
                try:
                    average_price = float(average_price_raw)
                except (ValueError, TypeError):
                    logger.warning("Cannot convert average_price to float", order_id=order_id, average_price_raw=average_price_raw, type=type(average_price_raw).__name__)
                    average_price = None
            
            fees = float(order_data.get("fees", "0")) if order_data.get("fees") else 0.0
            executed_at_str = order_data.get("executed_at")
            created_at_str = order_data.get("created_at")

            if not average_price or average_price <= 0:
                logger.warning(
                    "Order event missing or invalid average_price",
                    order_id=order_id,
                    event_type=event_type,
                    average_price=average_price,
                    average_price_raw=average_price_raw,
                    filled_quantity=filled_quantity,
                )
                return None

            if filled_quantity <= 0:
                logger.warning("Order event has zero or negative filled_quantity", order_id=order_id)
                return None

            # Try to get signal information from event first, then from database
            signal_info = None
            signal_data = data.get("signal", {})
            
            # Check if signal data is in the event
            if signal_data and signal_data.get("strategy_id"):
                signal_price = float(signal_data.get("price", "0")) if signal_data.get("price") else None
                if not signal_price or signal_price <= 0:
                    # Try to get from market_data_snapshot if available
                    market_snapshot = signal_data.get("market_data_snapshot") or {}
                    if isinstance(market_snapshot, dict):
                        signal_price = float(market_snapshot.get("price", "0"))
                
                if signal_price and signal_price > 0:
                    signal_info = {
                        "signal_id": signal_id,
                        "strategy_id": signal_data.get("strategy_id", "unknown"),
                        "price": str(signal_price),
                        "timestamp": signal_data.get("timestamp") or created_at_str or executed_at_str,
                    }
                    logger.debug("Using signal data from event", signal_id=signal_id)
            
            # If not in event, try database
            if not signal_info:
                signal_info = await self._get_signal_info(signal_id)
            
            # If still not found, use fallback from order data
            if not signal_info:
                logger.warning("Signal not found in database or event, using fallback", signal_id=signal_id)
                # Try to extract price from order price field (for limit orders) or use execution price as fallback
                order_price = float(order_data.get("price", "0")) if order_data.get("price") else None
                # For market orders, use execution price as signal price (best approximation)
                # For limit orders, use order price
                signal_price = order_price if (order_price and order_price > 0) else average_price
                
                if not signal_price or signal_price <= 0:
                    logger.warning("Cannot determine signal price", signal_id=signal_id, order_price=order_price, average_price=average_price)
                    return None
                
                # Try to get strategy_id from signal_data if available, otherwise use "unknown"
                strategy_id_fallback = signal_data.get("strategy_id") if signal_data else "unknown"
                
                signal_info = {
                    "signal_id": signal_id,
                    "strategy_id": strategy_id_fallback,
                    "price": str(signal_price),
                    "timestamp": created_at_str or executed_at_str or datetime.now(timezone.utc).isoformat(),
                }
                logger.info("Using fallback signal data", signal_id=signal_id, signal_price=signal_price, strategy_id=strategy_id_fallback)

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
                    executed_at = datetime.now(timezone.utc)
            else:
                executed_at = datetime.now(timezone.utc)

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
        # Ensure all datetime objects are timezone-aware (asyncpg requires this)
        # Convert to UTC timezone-aware datetime
        executed_at = execution_event.executed_at
        if executed_at.tzinfo is None:
            executed_at = executed_at.replace(tzinfo=timezone.utc)
        else:
            # Ensure it's UTC timezone
            executed_at = executed_at.astimezone(timezone.utc)
        
        signal_timestamp = execution_event.signal_timestamp
        if signal_timestamp.tzinfo is None:
            signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)
        else:
            # Ensure it's UTC timezone
            signal_timestamp = signal_timestamp.astimezone(timezone.utc)
        
        # Log datetime info for debugging
        logger.debug(
            "Preparing datetime for database",
            executed_at_tzinfo=str(executed_at.tzinfo),
            signal_timestamp_tzinfo=str(signal_timestamp.tzinfo),
            executed_at_iso=executed_at.isoformat(),
            signal_timestamp_iso=signal_timestamp.isoformat(),
        )
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
                executed_at=executed_at,
                signal_price=execution_event.signal_price,
                signal_timestamp=signal_timestamp,
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

