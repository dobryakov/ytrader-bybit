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
            # Check if there's already an active close order for this asset
            # This prevents creating duplicate close orders
            # For long position (size > 0), close order should be SELL
            # For short position (size < 0), close order should be Buy
            expected_close_side = "SELL" if position.size > 0 else "Buy"
            
            pool = await DatabaseConnection.get_pool()
            
            # First check: Look for existing close order for this specific original_order_id
            # by checking signals with metadata containing original_order_id
            check_specific_query = """
                SELECT o.id, o.status, o.created_at, o.side
                FROM orders o
                JOIN signal_order_relationships sor ON sor.order_id = o.id
                JOIN trading_signals ts ON ts.signal_id = sor.signal_id
                WHERE o.asset = $1
                    AND o.side = $2
                    AND ts.metadata->>'original_order_id' = $3
                    AND o.status IN ('pending', 'partially_filled')
                ORDER BY o.created_at DESC
                LIMIT 1
            """
            existing_specific_order = await pool.fetchrow(check_specific_query, asset, expected_close_side, order_id)
            
            if existing_specific_order:
                logger.info(
                    "target_horizon_skip_existing_close_order_for_original",
                    asset=asset,
                    original_order_id=order_id,
                    existing_close_order_id=str(existing_specific_order["id"]),
                    existing_close_order_status=existing_specific_order["status"],
                    existing_close_order_side=existing_specific_order["side"],
                    existing_close_order_created_at=existing_specific_order["created_at"].isoformat() if existing_specific_order["created_at"] else None,
                    trace_id=trace_id,
                )
                return True  # Consider it successful since close order already exists for this original_order_id
            
            # Second check: Look for any active close order for this asset (broader check)
            check_general_query = """
                SELECT id, status, created_at, side
                FROM orders
                WHERE asset = $1
                    AND side = $2
                    AND status IN ('pending', 'partially_filled')
                ORDER BY created_at DESC
                LIMIT 1
            """
            existing_order = await pool.fetchrow(check_general_query, asset, expected_close_side)
            
            if existing_order:
                logger.info(
                    "target_horizon_skip_existing_close_order",
                    asset=asset,
                    original_order_id=order_id,
                    existing_close_order_id=str(existing_order["id"]),
                    existing_close_order_status=existing_order["status"],
                    existing_close_order_side=existing_order["side"],
                    existing_close_order_created_at=existing_order["created_at"].isoformat() if existing_order["created_at"] else None,
                    trace_id=trace_id,
                )
                return True  # Consider it successful since close order already exists
            
            # Initialize SignalProcessor if not already initialized
            if self._signal_processor is None:
                self._signal_processor = SignalProcessor()
            
            # Use SignalProcessor's method to close position
            # This creates a close order with reduceOnly=True
            from ..models.trading_signal import TradingSignal, MarketDataSnapshot
            from decimal import Decimal
            from uuid import uuid4
            
            # Use SignalProcessor's internal method to close position
            # We'll call the close position method directly
            from ..services.order_executor import OrderExecutor
            from ..services.instrument_info_manager import InstrumentInfoManager
            
            order_executor = OrderExecutor()
            instrument_info_manager = InstrumentInfoManager()
            close_quantity = abs(position.size)
            
            # Get current price for balance check (use position entry price or fetch current price)
            current_price = Decimal(str(position.average_entry_price)) if position.average_entry_price else None
            if current_price is None:
                # Fallback: try to get current price from instrument info
                try:
                    instrument_info = await instrument_info_manager.get_instrument(asset)
                    if instrument_info and instrument_info.last_price:
                        current_price = instrument_info.last_price
                    else:
                        current_price = Decimal("50000")  # Fallback price
                except Exception:
                    current_price = Decimal("50000")  # Fallback price
            
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
            
            logger.info(
                "creating_close_order_target_horizon",
                asset=asset,
                order_id=order_id,
                close_side=close_side,
                close_quantity=float(close_quantity),
                current_price=float(current_price),
                position_size=float(position.size),
                trace_id=trace_id,
            )
            
            # Check balance for commission even for reduce-only orders
            # Reduce-only orders don't require margin but still need balance for commission
            if not settings.order_manager_enable_dry_run and settings.order_manager_enable_balance_check:
                from ..services.risk_manager import RiskManager
                risk_manager = RiskManager()
                try:
                    await risk_manager.check_balance(
                        signal=close_signal,
                        order_quantity=close_quantity,
                        order_price=current_price,
                        is_reduce_only=True,
                    )
                    logger.info(
                        "balance_check_passed_target_horizon_reduce_only",
                        asset=asset,
                        order_id=order_id,
                        position_size=float(position.size),
                        close_quantity=float(close_quantity),
                        current_price=float(current_price),
                        reason="Reduce-only order: commission balance sufficient",
                        trace_id=trace_id,
                    )
                except Exception as e:
                    logger.error(
                        "balance_check_failed_target_horizon_reduce_only",
                        asset=asset,
                        order_id=order_id,
                        position_size=float(position.size),
                        close_quantity=float(close_quantity),
                        current_price=float(current_price),
                        error=str(e),
                        reason="Insufficient balance for commission on reduce-only order",
                        trace_id=trace_id,
                    )
                    raise
            else:
                skip_reason = "dry_run" if settings.order_manager_enable_dry_run else "balance_check_disabled"
                logger.info(
                    "balance_check_skipped_target_horizon_reduce_only",
                    asset=asset,
                    order_id=order_id,
                    position_size=float(position.size),
                    reason=f"Balance check skipped: {skip_reason}",
                    trace_id=trace_id,
                )
            
            # Verify position still exists before creating order
            # Position might have been closed between check and order creation
            try:
                current_position = await self._position_manager_client.get_position(
                    asset=asset,
                    mode="one-way",
                    trace_id=trace_id,
                )
                
                if not current_position or abs(current_position.size) < Decimal("0.00000001"):
                    # Position already closed - success
                    logger.info(
                        "target_horizon_position_already_closed",
                        asset=asset,
                        order_id=order_id,
                        trace_id=trace_id,
                        reason="Position already closed before creating close order",
                    )
                    return True
                
                # Update close_quantity if position size changed
                if abs(current_position.size) != close_quantity:
                    logger.info(
                        "target_horizon_position_size_changed",
                        asset=asset,
                        order_id=order_id,
                        original_quantity=float(close_quantity),
                        current_position_size=float(current_position.size),
                        trace_id=trace_id,
                        reason="Position size changed, updating close quantity",
                    )
                    close_quantity = abs(current_position.size)
            except Exception as e:
                logger.warning(
                    "target_horizon_position_check_failed",
                    asset=asset,
                    order_id=order_id,
                    error=str(e),
                    trace_id=trace_id,
                    reason="Failed to verify position before creating order, proceeding anyway",
                )
                # Continue with order creation anyway
            
            # Create close order with reduceOnly=True
            close_order = await order_executor.create_order(
                signal=close_signal,
                order_type="Market",
                quantity=close_quantity,
                price=None,
                trace_id=trace_id,
                force_reduce_only=True,
            )
            
            # None means position was already closed (success case for reduce-only)
            if close_order is None:
                logger.info(
                    "target_horizon_position_closed_during_order",
                    asset=asset,
                    order_id=order_id,
                    trace_id=trace_id,
                    reason="Position was closed during order creation (reduce-only order returned None)",
                )
                return True
            
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

