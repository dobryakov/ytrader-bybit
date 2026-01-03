"""Background task for closing positions when target timestamp is reached."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..models.order import Order
from ..models.position import Position
from ..services.position_manager_client import PositionManagerClient
from ..services.signal_processor import SignalProcessor
from ..utils.tracing import generate_trace_id, set_trace_id

logger = get_logger(__name__)


class TargetHorizonCloseTask:
    """Background task for closing positions when target timestamp is reached."""

    def __init__(self):
        """Initialize target horizon close task."""
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager_client = PositionManagerClient()
        self._signal_processor: Optional[SignalProcessor] = None

    async def start(self) -> None:
        """Start the target horizon close task."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._close_loop())
        logger.info(
            "target_horizon_close_task_started",
            check_interval=60,
        )

    async def stop(self) -> None:
        """Stop the target horizon close task."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("target_horizon_close_task_stopped")

    async def _close_loop(self) -> None:
        """Continuously check and close positions for orders with expired target timestamp."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                # Wait 60 seconds before next check
                await asyncio.sleep(60)

                if not self._should_run:
                    break

                # Query filled/partially_filled orders with expired target_timestamp
                pool = await DatabaseConnection.get_pool()
                # Normalize to timezone-naive UTC for comparison with database TIMESTAMP column
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                
                query = """
                    SELECT id, order_id, signal_id, asset, side, order_type, quantity,
                           filled_quantity, status, target_timestamp
                    FROM orders
                    WHERE target_timestamp IS NOT NULL
                        AND target_timestamp <= $1
                        AND status IN ('filled', 'partially_filled')
                    ORDER BY target_timestamp ASC
                """
                rows = await pool.fetch(query, now_utc)

                if not rows:
                    continue

                logger.info(
                    "target_horizon_check",
                    found_count=len(rows),
                    current_time=now_utc.isoformat(),
                    trace_id=trace_id,
                )

                closed_count = 0
                error_count = 0
                skipped_count = 0

                for row in rows:
                    try:
                        order_id = row["order_id"]
                        asset = row["asset"]
                        target_ts = row["target_timestamp"]
                        order_status = row["status"]
                        
                        # Check if position still exists
                        position = await self._position_manager_client.get_position(
                            asset=asset,
                            mode="one-way",
                            trace_id=trace_id,
                        )
                        
                        if not position or position.size == 0:
                            # Position already closed or doesn't exist
                            skipped_count += 1
                            logger.debug(
                                "target_horizon_skip_no_position",
                                order_id=order_id,
                                asset=asset,
                                target_timestamp=target_ts.isoformat() if target_ts else None,
                                trace_id=trace_id,
                            )
                            continue

                        logger.info(
                            "closing_position_target_horizon_reached",
                            order_id=order_id,
                            asset=asset,
                            target_timestamp=target_ts.isoformat() if target_ts else None,
                            current_time=now_utc.isoformat(),
                            position_size=float(position.size),
                            trace_id=trace_id,
                        )

                        # Close position using SignalProcessor's close position method
                        success = await self._close_position_for_order(
                            asset=asset,
                            position=position,
                            order_id=order_id,
                            trace_id=trace_id,
                        )

                        if success:
                            closed_count += 1
                            logger.info(
                                "position_closed_target_horizon",
                                order_id=order_id,
                                asset=asset,
                                target_timestamp=target_ts.isoformat() if target_ts else None,
                                trace_id=trace_id,
                            )
                        else:
                            error_count += 1
                            logger.warning(
                                "position_close_failed_target_horizon",
                                order_id=order_id,
                                asset=asset,
                                trace_id=trace_id,
                            )

                    except Exception as e:
                        error_count += 1
                        logger.error(
                            "target_horizon_close_error",
                            order_id=row.get("order_id"),
                            asset=row.get("asset"),
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )

                if closed_count > 0 or error_count > 0 or skipped_count > 0:
                    logger.info(
                        "target_horizon_check_completed",
                        closed_count=closed_count,
                        error_count=error_count,
                        skipped_count=skipped_count,
                        total_found=len(rows),
                        trace_id=trace_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "target_horizon_close_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs
                await asyncio.sleep(60)  # Wait before retrying

    async def _close_position_for_order(
        self,
        asset: str,
        position: Position,
        order_id: str,
        trace_id: Optional[str],
    ) -> bool:
        """Close position for an order with expired target timestamp.
        
        Args:
            asset: Trading pair symbol
            position: Position object to close
            order_id: Order ID for logging
            trace_id: Trace ID for logging
            
        Returns:
            True if position was closed successfully, False otherwise
        """
        try:
            # Initialize SignalProcessor if not already initialized
            if self._signal_processor is None:
                self._signal_processor = SignalProcessor()
            
            # Use SignalProcessor's method to close position
            # This creates a close order with reduceOnly=True
            from ..models.trading_signal import TradingSignal, MarketDataSnapshot
            from decimal import Decimal
            from uuid import uuid4
            
            # Create a minimal signal for closing (reuse position data)
            # We need to create a signal-like object for the close operation
            # The actual close happens through OrderExecutor with force_reduce_only=True
            
            # Get current price for the close signal
            current_price = Decimal(str(position.average_entry_price)) if position.average_entry_price else Decimal("50000")
            
            # Create a minimal market data snapshot
            market_data_snapshot = MarketDataSnapshot(
                price=current_price,
                spread=None,
                volume_24h=None,
                volatility=None,
            )
            
            # Determine close side based on position direction
            if position.size > 0:
                close_side = "sell"  # Long position - need SELL order to close
            else:
                close_side = "buy"   # Short position - need BUY order to close
            
            # Create minimal close signal
            close_signal = TradingSignal(
                signal_id=uuid4(),
                asset=asset,
                signal_type=close_side,
                amount=Decimal("1000"),  # Not used for close order
                confidence=Decimal("1.0"),
                strategy_id="target_horizon_close",
                timestamp=datetime.now(timezone.utc),
                market_data_snapshot=market_data_snapshot,
                trace_id=trace_id,
                model_version=None,
                is_warmup=False,
                metadata={
                    "reason": "target_horizon_reached",
                    "original_order_id": order_id,
                },
            )
            
            # Use SignalProcessor's internal method to close position
            # We'll call the close position method directly
            from ..services.order_executor import OrderExecutor
            
            order_executor = OrderExecutor()
            close_quantity = abs(position.size)
            
            logger.info(
                "creating_close_order_target_horizon",
                asset=asset,
                order_id=order_id,
                close_side=close_side,
                close_quantity=float(close_quantity),
                trace_id=trace_id,
            )
            
            # Create close order with reduceOnly=True
            close_order = await order_executor.create_order(
                signal=close_signal,
                order_type="Market",
                quantity=close_quantity,
                price=None,
                trace_id=trace_id,
                force_reduce_only=True,
            )
            
            if close_order:
                logger.info(
                    "close_order_created_target_horizon",
                    asset=asset,
                    original_order_id=order_id,
                    close_order_id=str(close_order.id),
                    trace_id=trace_id,
                )
                return True
            else:
                logger.warning(
                    "close_order_creation_failed_target_horizon",
                    asset=asset,
                    original_order_id=order_id,
                    trace_id=trace_id,
                )
                return False
                
        except Exception as e:
            logger.error(
                "close_position_for_order_error",
                asset=asset,
                order_id=order_id,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return False

