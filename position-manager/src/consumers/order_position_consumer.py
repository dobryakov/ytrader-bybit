"""Order execution event consumer.

Consumes order execution events from `order-manager.order_events` queue and
delegates processing to PositionManager.update_position_from_order_fill().
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

from aio_pika import IncomingMessage

from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..services.position_manager import PositionManager
from ..utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


@dataclass
class OrderExecutionEvent:
    """Typed view over order execution event payload."""

    asset: str
    mode: str
    size_delta: Decimal
    execution_price: Decimal
    execution_fees: Optional[Decimal]
    trace_id: Optional[str]
    execution_timestamp: Optional[datetime]

    @classmethod
    def from_message(cls, payload: Dict[str, Any]) -> "OrderExecutionEvent":
        event_type = payload.get("event_type")
        # Support multiple event types from order-manager
        supported_types = {"position_updated_from_order", "order_executed", "filled", "partially_filled"}
        if event_type not in supported_types:
            raise ValueError(f"Unsupported event_type: {event_type}")

        trace_id = payload.get("trace_id") or get_or_create_trace_id()

        # Extract order data - support both flat structure and nested "order" object
        order_data = payload.get("order", {})
        if not order_data:
            # Fallback to flat structure for backward compatibility
            order_data = payload

        asset = order_data.get("asset") or payload.get("asset")
        if not asset:
            raise ValueError("Missing 'asset' in order execution event")

        mode = payload.get("mode") or order_data.get("mode") or "one-way"

        side = (order_data.get("side") or payload.get("side") or "").lower()
        filled = (
            order_data.get("filled_quantity")
            or payload.get("filled_quantity")
            or order_data.get("execution_quantity")
            or payload.get("execution_quantity")
            or order_data.get("filled_qty")
            or payload.get("filled_qty")
        )
        price = (
            order_data.get("average_price")
            or order_data.get("price")
            or payload.get("execution_price")
            or payload.get("price")
        )
        fees_raw = order_data.get("fees") or payload.get("execution_fees")

        if filled is None or price is None:
            raise ValueError("Missing filled_quantity or execution_price in order event")

        try:
            qty = Decimal(str(filled))
            px = Decimal(str(price))
            fees = Decimal(str(fees_raw)) if fees_raw is not None else None
        except Exception as e:
            raise ValueError(f"Invalid decimal values in order event: {e}") from e

        # Buy increases size, Sell decreases size
        if side == "sell":
            qty = -qty

        # Execution timestamp (Phase 9: used for conflict resolution)
        ts_raw = (
            order_data.get("executed_at")
            or payload.get("executed_at")
            or payload.get("timestamp")
        )
        execution_ts: Optional[datetime] = None
        if isinstance(ts_raw, str):
            try:
                if ts_raw.endswith("Z"):
                    execution_ts = datetime.fromisoformat(ts_raw[:-1])
                else:
                    execution_ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                execution_ts = None

        return cls(
            asset=asset,
            mode=mode,
            size_delta=qty,
            execution_price=px,
            execution_fees=fees,
            trace_id=trace_id,
            execution_timestamp=execution_ts,
        )


class OrderPositionConsumer:
    """Async consumer for `order-manager.order_events` queue."""

    def __init__(self, position_manager: Optional[PositionManager] = None) -> None:
        self._position_manager = position_manager or PositionManager()
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def _handle_message(self, message: IncomingMessage) -> None:
        trace_id = None
        try:
            payload = json.loads(message.body.decode("utf-8"))
            event = OrderExecutionEvent.from_message(payload)
            trace_id = event.trace_id

            logger.info(
                "order_position_event_received",
                asset=event.asset,
                mode=event.mode,
                size_delta=str(event.size_delta),
                execution_price=str(event.execution_price),
                trace_id=trace_id,
            )

            await self._position_manager.update_position_from_order_fill(
                asset=event.asset,
                size_delta=event.size_delta,
                execution_price=event.execution_price,
                execution_fees=event.execution_fees,
                mode=event.mode,
                execution_timestamp=event.execution_timestamp,
            )

            await message.ack()
            logger.debug("order_position_event_acked", trace_id=trace_id)
        except (ValueError, KeyError) as e:
            logger.error(
                "order_position_event_validation_failed",
                error=str(e),
                trace_id=trace_id,
            )
            await message.reject(requeue=False)
        except Exception as e:
            logger.error(
                "order_position_event_processing_failed",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            await message.nack(requeue=True)

    async def start(self) -> None:
        """Start consuming from order-manager.order_events."""
        channel = await RabbitMQConnection.get_channel()
        queue = await channel.declare_queue("order-manager.order_events", durable=True)

        logger.info("order_position_consumer_started", queue=queue.name)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self._stopped.is_set():
                    break
                await self._handle_message(message)

        logger.info("order_position_consumer_stopped")

    async def run_forever(self) -> None:
        """Run the consumer until stop() is called."""
        try:
            await self.start()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            logger.info("order_position_consumer_cancelled")
        except Exception as e:  # pragma: no cover
            logger.error("order_position_consumer_crashed", error=str(e), exc_info=True)

    def spawn(self) -> None:
        """Spawn the consumer in a background task with automatic restart on failure."""
        if self._task is None or self._task.done():
            self._stopped.clear()
            self._task = asyncio.create_task(self._run_with_restart())

    async def _run_with_restart(self) -> None:
        """Run consumer with automatic restart on failure."""
        max_restart_delay = 30.0  # Maximum delay between restarts
        restart_delay = 1.0  # Initial delay
        
        while not self._stopped.is_set():
            try:
                await self.run_forever()
                # If run_forever completes normally (not cancelled), exit loop
                logger.info("order_position_consumer_completed_normally")
                break
            except asyncio.CancelledError:
                logger.info("order_position_consumer_cancelled")
                break
            except Exception as e:
                logger.error(
                    "order_position_consumer_crashed_will_restart",
                    error=str(e),
                    error_type=type(e).__name__,
                    restart_delay=restart_delay,
                    exc_info=True,
                )
                
                # Wait before restarting
                try:
                    await asyncio.sleep(restart_delay)
                except asyncio.CancelledError:
                    break
                
                # Exponential backoff, capped at max_restart_delay
                restart_delay = min(restart_delay * 2.0, max_restart_delay)
                
                # Clear stopped flag to allow restart
                self._stopped.clear()
                
                logger.info(
                    "order_position_consumer_restarting",
                    restart_delay=restart_delay,
                )

    async def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task



