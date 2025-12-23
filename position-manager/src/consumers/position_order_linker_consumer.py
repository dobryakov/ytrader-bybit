"""Position-order linker consumer.

Consumes order execution events from `order-manager.order_events` queue and
creates position_orders relationships (links orders with positions).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

import asyncpg
from aio_pika import IncomingMessage

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..database.repositories.position_order_repo import PositionOrderRepository
from ..services.position_manager import PositionManager
from ..utils.tracing import get_or_create_trace_id
from ..exceptions import DatabaseError

logger = get_logger(__name__)


@dataclass
class OrderExecutionEvent:
    """Typed view over order execution event payload."""

    order_id: str  # bybit_order_id
    signal_id: Optional[str]
    asset: str
    side: str  # "buy" or "sell"
    mode: str
    filled_quantity: Decimal
    execution_price: Decimal
    execution_fees: Optional[Decimal]
    trace_id: Optional[str]
    execution_timestamp: Optional[datetime]

    @classmethod
    def from_message(cls, payload: Dict[str, Any]) -> "OrderExecutionEvent":
        event_type = payload.get("event_type")
        # Support multiple event types from order-manager
        # Only filled/partially_filled affect positions, others are ignored
        supported_types = {"filled", "partially_filled"}
        if event_type not in supported_types:
            raise ValueError(f"Unsupported event_type: {event_type} (expected 'filled' or 'partially_filled')")

        trace_id = payload.get("trace_id") or get_or_create_trace_id()

        # Extract order data - support nested "order" object
        order_data = payload.get("order", {})
        if not order_data:
            # Fallback to flat structure for backward compatibility
            order_data = payload

        order_id = order_data.get("order_id") or payload.get("order_id")
        if not order_id:
            raise ValueError("Missing 'order_id' (bybit_order_id) in order execution event")

        asset = order_data.get("asset") or payload.get("asset")
        if not asset:
            raise ValueError("Missing 'asset' in order execution event")

        side = (order_data.get("side") or payload.get("side") or "").lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {side} (expected 'buy' or 'sell')")

        mode = payload.get("mode") or order_data.get("mode") or "one-way"

        filled = (
            order_data.get("filled_quantity")
            or payload.get("filled_quantity")
        )
        price = (
            order_data.get("average_price")
            or order_data.get("price")
            or payload.get("execution_price")
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

        signal_id = order_data.get("signal_id") or payload.get("signal_id")

        # Execution timestamp
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
            order_id=order_id,
            signal_id=signal_id,
            asset=asset,
            side=side,
            mode=mode,
            filled_quantity=qty,
            execution_price=px,
            execution_fees=fees,
            trace_id=trace_id,
            execution_timestamp=execution_ts,
        )


class PositionOrderLinkerConsumer:
    """Async consumer for `order-manager.order_events` queue.
    
    Creates position_orders relationships (links orders with positions) based on order execution events.
    Does NOT update positions (positions are updated from WebSocket position events).
    """

    def __init__(self, position_manager: Optional[PositionManager] = None) -> None:
        self._position_manager = position_manager or PositionManager()
        self._position_order_repo = PositionOrderRepository()
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def _handle_message(self, message: IncomingMessage) -> None:
        trace_id = None
        try:
            payload = json.loads(message.body.decode("utf-8"))
            event_type = payload.get("event_type")
            
            # Ignore events that don't affect positions (cancelled, rejected, etc.)
            ignored_types = {"cancelled", "rejected", "created", "modified"}
            if event_type in ignored_types:
                trace_id = payload.get("trace_id") or get_or_create_trace_id()
                logger.debug(
                    "position_order_linker_event_ignored",
                    event_type=event_type,
                    order_id=payload.get("order_id"),
                    trace_id=trace_id,
                )
                await message.ack()
                return
            
            event = OrderExecutionEvent.from_message(payload)
            trace_id = event.trace_id

            logger.info(
                "position_order_linker_event_received",
                order_id=event.order_id,
                asset=event.asset,
                mode=event.mode,
                side=event.side,
                filled_quantity=str(event.filled_quantity),
                execution_price=str(event.execution_price),
                trace_id=trace_id,
            )

            await self._create_position_order_relationship(event, trace_id)

            await message.ack()
            logger.debug("position_order_linker_event_acked", trace_id=trace_id)
        except (ValueError, KeyError) as e:
            logger.error(
                "position_order_linker_event_validation_failed",
                error=str(e),
                trace_id=trace_id,
            )
            await message.reject(requeue=False)
        except (asyncio.TimeoutError, asyncpg.PostgresConnectionError, asyncpg.DeadlockDetectedError) as e:
            logger.warning(
                "position_order_linker_db_error_requeue",
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
            )
            await message.nack(requeue=True)
        except Exception as e:
            logger.error(
                "position_order_linker_event_processing_failed",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            await message.nack(requeue=True)

    async def _create_position_order_relationship(
        self,
        event: OrderExecutionEvent,
        trace_id: Optional[str],
    ) -> None:
        """Create position_orders relationship based on order execution event."""
        try:
            # 1. Find order in DB by bybit_order_id (may be NULL if order not yet created)
            order_id = await self._find_order_id_by_bybit_id(event.order_id, trace_id)

            # 2. Calculate size_delta based on order side
            if event.side == "buy":
                size_delta = +event.filled_quantity
            else:  # sell
                size_delta = -event.filled_quantity

            # 3. Find or create position
            position = await self._find_or_create_position(
                asset=event.asset,
                mode=event.mode,
                initial_size=size_delta,
                initial_avg_price=event.execution_price,
                trace_id=trace_id,
            )

            # 4. Determine relationship_type based on position size at order execution time
            # Calculate position size at the moment of order execution by summing all previous size_deltas
            executed_at = event.execution_timestamp or datetime.utcnow()
            position_size_at_execution = await self._calculate_position_size_at_time(
                position_id=position.id,
                execution_time=executed_at,
                trace_id=trace_id,
            )
            
            relationship_type = await self._determine_relationship_type(
                position_size_at_execution=position_size_at_execution,
                side=event.side,
                size_delta=size_delta,
                trace_id=trace_id,
            )

            # 5. Create or update position_orders record (handles partial fills)
            await self._position_order_repo.upsert(
                position_id=position.id,
                bybit_order_id=event.order_id,
                order_id=order_id,  # may be NULL
                relationship_type=relationship_type,
                size_delta=size_delta,
                execution_price=event.execution_price,
                executed_at=event.execution_timestamp or datetime.utcnow(),
            )

            # 6. Update positions.total_fees (if execution_fees provided)
            if event.execution_fees:
                await self._update_position_total_fees(
                    position_id=position.id,
                    fees=event.execution_fees,
                    trace_id=trace_id,
                )

            logger.info(
                "position_order_relationship_created",
                position_id=str(position.id),
                bybit_order_id=event.order_id,
                order_id=str(order_id) if order_id else None,
                relationship_type=relationship_type,
                size_delta=str(size_delta),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "position_order_relationship_creation_failed",
                order_id=event.order_id,
                asset=event.asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise

    async def _find_order_id_by_bybit_id(
        self,
        bybit_order_id: str,
        trace_id: Optional[str],
    ) -> Optional[UUID]:
        """Find internal order UUID by bybit_order_id."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = "SELECT id FROM orders WHERE order_id = $1"
            row = await pool.fetchrow(query, bybit_order_id)
            if row:
                # row["id"] is already a UUID object from asyncpg, convert to string first
                return UUID(str(row["id"]))
            return None
        except Exception as e:
            logger.warning(
                "order_not_found_by_bybit_id",
                bybit_order_id=bybit_order_id,
                error=str(e),
                trace_id=trace_id,
            )
            return None

    async def _find_or_create_position(
        self,
        asset: str,
        mode: str,
        initial_size: Decimal,
        initial_avg_price: Decimal,
        trace_id: Optional[str],
    ) -> Any:  # Position
        """Find existing position or create new one with minimal data."""
        # Try to get existing position
        position = await self._position_manager.get_position(asset, mode)
        if position:
            return position

        # Position doesn't exist - create with minimal data
        # Will be updated properly when position event arrives from WebSocket
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                INSERT INTO positions (
                    asset, mode, size, average_entry_price,
                    unrealized_pnl, realized_pnl, total_fees,
                    current_price, version, last_updated, created_at
                )
                VALUES ($1, $2, $3, $4, 0, 0, 0, NULL, 1, NOW(), NOW())
                ON CONFLICT (asset, mode) DO UPDATE SET
                    asset = EXCLUDED.asset,
                    mode = EXCLUDED.mode
                RETURNING id, asset, mode, size, average_entry_price, current_price,
                          unrealized_pnl, realized_pnl,
                          long_size, short_size, version,
                          last_updated, closed_at, created_at
            """
            row = await pool.fetchrow(
                query,
                asset.upper(),
                mode.lower(),
                str(initial_size),
                str(initial_avg_price),
            )
            if row:
                from ..models import Position
                position_dict = dict(row)
                position = Position.from_db_dict(position_dict)
                logger.info(
                    "position_created_from_order_event",
                    asset=asset,
                    mode=mode,
                    initial_size=str(initial_size),
                    trace_id=trace_id,
                )
                return position

            # Conflict: position was created concurrently, fetch it
            position = await self._position_manager.get_position(asset, mode)
            if position:
                return position

            raise DatabaseError(f"Failed to create or fetch position for {asset}/{mode}")

        except Exception as e:
            logger.error(
                "position_creation_from_order_event_failed",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to create position: {e}") from e

    async def _calculate_position_size_at_time(
        self,
        position_id: UUID,
        execution_time: datetime,
        trace_id: Optional[str],
    ) -> Decimal:
        """Calculate position size at a specific moment in time by summing all size_deltas before that time.
        
        This gives us the accurate position size at the moment of order execution,
        regardless of what the current position size is in the database.
        
        Args:
            position_id: Position UUID
            execution_time: Timestamp to calculate size at
            trace_id: Optional trace ID
            
        Returns:
            Position size at the specified time
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT COALESCE(SUM(size_delta), 0) as total_size
                FROM position_orders
                WHERE position_id = $1
                  AND executed_at < $2
            """
            row = await pool.fetchrow(query, position_id, execution_time)
            if row:
                return Decimal(str(row["total_size"]))
            return Decimal("0")
        except Exception as e:
            logger.warning(
                "position_size_calculation_failed",
                position_id=str(position_id),
                execution_time=execution_time.isoformat(),
                error=str(e),
                trace_id=trace_id,
            )
            # Fallback: return 0 if calculation fails
            return Decimal("0")

    async def _determine_relationship_type(
        self,
        position_size_at_execution: Decimal,
        side: str,
        size_delta: Decimal,
        trace_id: Optional[str],
    ) -> str:
        """Determine relationship_type based on position size at order execution time.
        
        Args:
            position_size_at_execution: Position size at the moment of order execution
            side: Order side ("buy" or "sell")
            size_delta: Size delta from this order
            trace_id: Optional trace ID
            
        Returns:
            Relationship type: "opened", "increased", "decreased", "closed", or "reversed"
        """
        current_size = position_size_at_execution

        if current_size == 0:
            # Position is empty - this order opens it
            return "opened"

        # Check if position increases in same direction
        if (side == "buy" and current_size > 0) or (side == "sell" and current_size < 0):
            # Increasing position in same direction
            return "increased"

        # Position decreases or changes direction
        new_size = current_size + size_delta

        if new_size == 0:
            # Position completely closed
            return "closed"
        elif (current_size > 0 and new_size < 0) or (current_size < 0 and new_size > 0):
            # Position reversed direction
            return "reversed"
        else:
            # Partial decrease (same sign, smaller magnitude)
            return "decreased"

    async def _update_position_total_fees(
        self,
        position_id: UUID,
        fees: Decimal,
        trace_id: Optional[str],
    ) -> None:
        """Update positions.total_fees by adding execution fees."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                UPDATE positions
                SET total_fees = total_fees + $1
                WHERE id = $2
            """
            await pool.execute(query, str(fees), position_id)
            logger.debug(
                "position_total_fees_updated",
                position_id=str(position_id),
                fees_added=str(fees),
                trace_id=trace_id,
            )
        except Exception as e:
            logger.warning(
                "position_total_fees_update_failed",
                position_id=str(position_id),
                fees=str(fees),
                error=str(e),
                trace_id=trace_id,
            )
            # Don't raise - fee update failure shouldn't block position_order creation

    async def start(self) -> None:
        """Start consuming from order-manager.order_events."""
        channel = await RabbitMQConnection.get_channel()
        queue = await channel.declare_queue("order-manager.order_events", durable=True)

        logger.info("position_order_linker_consumer_started", queue=queue.name)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self._stopped.is_set():
                    break
                await self._handle_message(message)

        logger.info("position_order_linker_consumer_stopped")

    async def run_forever(self) -> None:
        """Run the consumer until stop() is called."""
        try:
            await self.start()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            logger.info("position_order_linker_consumer_cancelled")
        except Exception as e:  # pragma: no cover
            logger.error("position_order_linker_consumer_crashed", error=str(e), exc_info=True)

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
                logger.info("position_order_linker_consumer_completed_normally")
                break
            except asyncio.CancelledError:
                logger.info("position_order_linker_consumer_cancelled")
                break
            except Exception as e:
                logger.error(
                    "position_order_linker_consumer_crashed_will_restart",
                    error=str(e),
                    error_type=type(e).__name__,
                    restart_delay=restart_delay,
                    exc_info=True,
                )
                
                try:
                    await asyncio.sleep(restart_delay)
                except asyncio.CancelledError:
                    break
                
                restart_delay = min(restart_delay * 2.0, max_restart_delay)
                self._stopped.clear()
                
                logger.info(
                    "position_order_linker_consumer_restarting",
                    restart_delay=restart_delay,
                )

    async def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

