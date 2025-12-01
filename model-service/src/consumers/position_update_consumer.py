"""
Position update event consumer.

Consumes position update events from RabbitMQ queue position-manager.position_updated,
parses events, invalidates local cache for affected assets, and triggers exit strategy evaluation.
"""

import json
import asyncio
from typing import Optional, Any, Dict

import aio_pika
from aio_pika.abc import AbstractIncomingMessage
import aio_pika.exceptions

from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger
from ..config.settings import settings
from ..services.position_cache import position_cache
from ..services.position_based_signal_generator import position_based_signal_generator
from ..publishers.signal_publisher import signal_publisher

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
        body = None
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            # Validate position update event
            validation_result = self._validate_position_update_event(data)
            if not validation_result["valid"]:
                logger.warning(
                    "Position update event validation failed",
                    queue=self._queue_name,
                    errors=validation_result["errors"],
                    event_keys=list(data.keys()),
                    trace_id=data.get("trace_id"),
                )
                return

            # Extract validated fields
            asset = validation_result["asset"]
            trace_id = data.get("trace_id") or data.get("traceId")

            # Invalidate cache for the affected asset
            await position_cache.invalidate(asset)
            logger.debug(
                "Cache invalidated for position update",
                asset=asset,
                trace_id=trace_id,
            )

            # Trigger exit strategy evaluation if enabled
            if settings.exit_strategy_enabled:
                try:
                    # Extract position data for exit strategy evaluation
                    position_data = self._extract_position_data(data)

                    # Get strategy_id (default to first configured strategy)
                    strategy_id = self._get_strategy_id(asset)

                    # Evaluate exit strategy
                    exit_signal = await position_based_signal_generator.evaluate_position_exit(
                        position_data=position_data,
                        strategy_id=strategy_id,
                        trace_id=trace_id,
                    )

                    if exit_signal:
                        # Publish exit signal
                        await signal_publisher.publish(exit_signal)
                        logger.info(
                            "Exit signal published from position update",
                            asset=asset,
                            signal_id=exit_signal.signal_id,
                            exit_reason=exit_signal.metadata.get("reasoning") if exit_signal.metadata else None,
                            trace_id=trace_id,
                        )

                except Exception as e:
                    logger.error(
                        "Error evaluating exit strategy from position update",
                        asset=asset,
                        error=str(e),
                        exc_info=True,
                        trace_id=trace_id,
                    )
                    # Continue processing - don't let exit strategy errors stop cache invalidation

            logger.info(
                "Position update processed",
                asset=asset,
                trace_id=trace_id,
            )

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse position update JSON",
                queue=self._queue_name,
                error=str(e),
                body_preview=body[:200] if body and len(body) > 200 else body,
            )
        except Exception as e:
            logger.error(
                "Error processing position update message",
                queue=self._queue_name,
                error=str(e),
                exc_info=True,
            )

    def _validate_position_update_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate position update event.

        Args:
            data: Parsed event data

        Returns:
            Dictionary with:
                - valid: bool - whether event is valid
                - errors: List[str] - list of validation errors
                - asset: Optional[str] - validated asset if valid
        """
        errors = []
        asset = data.get("asset") or data.get("symbol") or data.get("trading_pair")

        # Required fields validation
        if not asset:
            errors.append("Missing required field: asset (or symbol or trading_pair)")

        # Validate unrealized_pnl if present (should be numeric)
        unrealized_pnl = data.get("unrealized_pnl")
        if unrealized_pnl is not None:
            try:
                float(unrealized_pnl)
            except (ValueError, TypeError):
                errors.append(f"Invalid unrealized_pnl value: {unrealized_pnl}")

        # Validate position_size if present (should be numeric and non-zero if position exists)
        position_size = data.get("size") or data.get("position_size")
        if position_size is not None:
            try:
                size_float = float(position_size)
                # Size can be zero (no position), but if present should be numeric
            except (ValueError, TypeError):
                errors.append(f"Invalid position size value: {position_size}")

        # Validate timestamp if present
        timestamp = data.get("timestamp")
        if timestamp is not None:
            # Timestamp validation is optional - just log if invalid format
            pass

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "asset": asset if len(errors) == 0 else None,
        }

    def _extract_position_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract position data from position update event.

        Args:
            event_data: Position update event data

        Returns:
            Dictionary with position data for exit strategy evaluation
        """
        asset = event_data.get("asset") or event_data.get("symbol") or event_data.get("trading_pair")

        # Extract all relevant position fields
        position_data = {
            "asset": asset,
            "size": event_data.get("size") or event_data.get("position_size"),
            "unrealized_pnl": event_data.get("unrealized_pnl"),
            "unrealized_pnl_pct": event_data.get("unrealized_pnl_pct"),
            "realized_pnl": event_data.get("realized_pnl"),
            "avg_price": event_data.get("avg_price") or event_data.get("average_price"),
            "mode": event_data.get("mode", "one-way"),
            "time_held_minutes": event_data.get("time_held_minutes"),
            "position_size_norm": event_data.get("position_size_norm"),
        }

        # Remove None values
        position_data = {k: v for k, v in position_data.items() if v is not None}

        return position_data

    def _get_strategy_id(self, asset: str) -> str:
        """
        Get strategy ID for asset.

        Args:
            asset: Trading pair symbol

        Returns:
            Strategy ID (defaults to first configured strategy)
        """
        strategies = settings.trading_strategy_list
        if strategies:
            return strategies[0]  # Use first configured strategy
        return "default"  # Fallback to default strategy


# Global position update consumer instance
position_update_consumer = PositionUpdateConsumer()

