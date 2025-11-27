"""Signal processor service for processing trading signals and making order decisions."""

import asyncio
from decimal import Decimal
from typing import Optional
from uuid import UUID

from ..config.logging import get_logger
from ..config.settings import settings
from ..models.trading_signal import TradingSignal
from ..models.order import Order
from ..models.signal_order_rel import SignalOrderRelationship
from ..services.order_type_selector import OrderTypeSelector
from ..services.quantity_calculator import QuantityCalculator
from ..services.position_manager import PositionManager
from ..services.risk_manager import RiskManager
from ..publishers.order_event_publisher import OrderEventPublisher
# Import OrderExecutor locally to avoid circular dependency
from ..exceptions import OrderExecutionError, RiskLimitError
from ..utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


class SignalProcessor:
    """Service for processing trading signals and orchestrating order creation.

    Implements per-symbol FIFO queue for signal processing, ensuring signals
    for the same asset are processed sequentially while different assets can
    process in parallel.
    """

    def __init__(self):
        """Initialize signal processor with dependencies."""
        self.order_type_selector = OrderTypeSelector()
        self.quantity_calculator = QuantityCalculator()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager()
        self.event_publisher = OrderEventPublisher()
        # Initialize OrderExecutor lazily to avoid circular dependency
        self._order_executor = None

        # Per-symbol FIFO queues for signal processing
        self._signal_queues: dict[str, asyncio.Queue] = {}
        self._processing_tasks: dict[str, asyncio.Task] = {}

    async def process_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Process a trading signal and create order if valid.

        This method validates the signal, performs risk checks, cancels
        existing orders if needed, and creates a new order.

        Args:
            signal: Trading signal to process

        Returns:
            Created Order object if successful, None if signal was rejected

        Raises:
            OrderExecutionError: If signal processing fails
            RiskLimitError: If risk limits are exceeded
        """
        trace_id = signal.trace_id or get_or_create_trace_id()
        signal_id = signal.signal_id
        asset = signal.asset

        logger.info(
            "signal_processing_started",
            signal_id=str(signal_id),
            asset=asset,
            signal_type=signal.signal_type,
            amount=float(signal.amount),
            trace_id=trace_id,
        )

        # Step 1: Validate signal
        try:
            self._validate_signal(signal)
        except ValueError as e:
            logger.error(
                "signal_validation_failed",
                signal_id=str(signal_id),
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(f"Signal validation failed: {e}") from e

        # Step 2: Get or create queue for this asset
        if asset not in self._signal_queues:
            self._signal_queues[asset] = asyncio.Queue()
            # Start processing task for this asset
            self._processing_tasks[asset] = asyncio.create_task(self._process_asset_queue(asset))

        # Step 3: Add signal to queue (FIFO per asset)
        await self._signal_queues[asset].put(signal)

        logger.debug(
            "signal_queued",
            signal_id=str(signal_id),
            asset=asset,
            queue_size=self._signal_queues[asset].qsize(),
            trace_id=trace_id,
        )

        # Step 4: Wait for processing result (in a real implementation, this would
        # be handled asynchronously via callbacks or events)
        # For now, we'll process immediately
        return await self._process_signal_internal(signal)

    async def _process_signal_internal(self, signal: TradingSignal) -> Optional[Order]:
        """Internal method to process signal after queuing.

        Args:
            signal: Trading signal to process

        Returns:
            Created Order object if successful, None if rejected
        """
        trace_id = signal.trace_id or get_or_create_trace_id()
        signal_id = signal.signal_id
        asset = signal.asset

        try:
            # Step 0: Check for duplicate signal
            duplicate_check = await self._check_duplicate_signal(signal_id, trace_id)
            if duplicate_check is not None:
                # Duplicate signal detected
                if duplicate_check["status"] == "succeeded":
                    logger.warning(
                        "duplicate_signal_rejected",
                        signal_id=str(signal_id),
                        asset=asset,
                        reason="Signal already processed successfully",
                        existing_order_id=str(duplicate_check["order_id"]),
                        trace_id=trace_id,
                    )
                    # Return the existing order instead of creating a new one
                    from ..config.database import DatabaseConnection
                    pool = await DatabaseConnection.get_pool()
                    query = "SELECT * FROM orders WHERE id = $1"
                    row = await pool.fetchrow(query, duplicate_check["order_id"])
                    if row:
                        from ..models.order import Order
                        return Order.from_dict(dict(row))
                    return None
                elif duplicate_check["status"] == "failed":
                    logger.info(
                        "duplicate_signal_retry_allowed",
                        signal_id=str(signal_id),
                        asset=asset,
                        reason="Previous processing failed, allowing retry",
                        trace_id=trace_id,
                    )
                    # Allow retry for failed signals

            # Step 1: Cancel existing orders for this asset if needed
            await self._cancel_existing_orders(asset, signal)

            # Step 2: Calculate order quantity
            quantity = await self.quantity_calculator.calculate_quantity(signal)

            # Step 3: Select order type and price
            order_type, limit_price = self.order_type_selector.select_order_type(signal)
            order_price = limit_price or signal.market_data_snapshot.price

            # Step 4: Get current position
            current_position = await self.position_manager.get_position(asset)

            # Step 5: Risk checks
            # 5a: Balance check (skip in dry-run mode or if disabled)
            # Note: Balance check is optional - Bybit API will reject orders with insufficient balance anyway.
            # This check provides early rejection and better logging, but adds an extra API call.
            if not settings.order_manager_enable_dry_run and settings.order_manager_enable_balance_check:
                await self.risk_manager.check_balance(signal, quantity, order_price)
            else:
                skip_reason = "dry_run" if settings.order_manager_enable_dry_run else "balance_check_disabled"
                logger.info(
                    "balance_check_skipped",
                    signal_id=str(signal_id),
                    reason=skip_reason,
                    trace_id=trace_id,
                )

            # 5b: Order size check
            self.risk_manager.check_order_size(signal, quantity, order_price)

            # 5c: Position size check
            order_side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"
            self.risk_manager.check_position_size(asset, current_position, quantity, order_side)

            # Step 6: Create order via executor
            from ..services.order_executor import OrderExecutor
            if self._order_executor is None:
                self._order_executor = OrderExecutor()
            order = await self._order_executor.create_order(
                signal=signal,
                order_type=order_type,
                quantity=quantity,
                price=limit_price,
                trace_id=trace_id,
            )

            # Step 7: Create signal-order relationship
            if order:
                await self._create_signal_order_relationship(signal_id, order.id, trace_id)
                # Mark signal as succeeded in tracking
                await self._mark_signal_processing_status(signal_id, "succeeded", None, trace_id, order.id)

            logger.info(
                "signal_processing_complete",
                signal_id=str(signal_id),
                asset=asset,
                order_id=str(order.id) if order else None,
                trace_id=trace_id,
            )

            return order

        except RiskLimitError as e:
            error_msg = f"Risk limit exceeded: {str(e)}"
            logger.warning(
                "signal_rejected_risk_limit",
                signal_id=str(signal_id),
                asset=asset,
                signal_type=signal.signal_type,
                amount=float(signal.amount),
                error=str(e),
                error_type="RiskLimitError",
                trace_id=trace_id,
            )
            # Mark signal as failed in tracking
            await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
            # Publish rejection event with signal information
            await self._publish_signal_rejection_event(
                signal=signal,
                rejection_reason=error_msg,
                trace_id=trace_id,
            )
            return None
        except OrderExecutionError as e:
            error_msg = f"Order execution failed: {str(e)}"
            logger.error(
                "signal_processing_failed",
                signal_id=str(signal_id),
                asset=asset,
                signal_type=signal.signal_type,
                amount=float(signal.amount),
                error=str(e),
                error_type="OrderExecutionError",
                trace_id=trace_id,
                exc_info=True,
            )
            # Mark signal as failed in tracking
            await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
            # Publish rejection event with signal information
            await self._publish_signal_rejection_event(
                signal=signal,
                rejection_reason=error_msg,
                trace_id=trace_id,
            )
            raise
        except ValueError as e:
            # Validation errors
            error_msg = f"Signal validation failed: {str(e)}"
            logger.error(
                "signal_validation_failed",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                error_type="ValueError",
                trace_id=trace_id,
            )
            # Mark signal as failed in tracking
            await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
            # Publish rejection event
            await self._publish_signal_rejection_event(
                signal=signal,
                rejection_reason=error_msg,
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg) from e
        except Exception as e:
            # Unexpected errors
            error_msg = f"Unexpected error during signal processing: {str(e)}"
            logger.error(
                "signal_processing_unexpected_error",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
                exc_info=True,
            )
            # Mark signal as failed in tracking
            await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
            # Publish rejection event
            await self._publish_signal_rejection_event(
                signal=signal,
                rejection_reason=error_msg,
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg) from e

    async def _process_asset_queue(self, asset: str) -> None:
        """Process signals from queue for a specific asset (FIFO).

        Implements per-symbol FIFO queue to ensure signals for the same asset
        are processed sequentially, preventing conflicts from simultaneous signals.

        Args:
            asset: Trading pair symbol
        """
        queue = self._signal_queues[asset]
        logger.info("asset_queue_processor_started", asset=asset)

        while True:
            try:
                # Get signal from queue (blocks until available)
                signal = await queue.get()
                queue_size = queue.qsize()
                
                if queue_size > 0:
                    logger.info(
                        "signal_processing_from_queue",
                        asset=asset,
                        signal_id=str(signal.signal_id),
                        queue_size=queue_size,
                        trace_id=signal.trace_id,
                        note="Processing signal from FIFO queue (conflict resolution for simultaneous signals)",
                    )

                # Process signal
                await self._process_signal_internal(signal)

                # Mark task as done
                queue.task_done()

            except asyncio.CancelledError:
                logger.info("asset_queue_processor_cancelled", asset=asset)
                break
            except Exception as e:
                logger.error(
                    "asset_queue_processor_error",
                    asset=asset,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                queue.task_done()

    def _validate_signal(self, signal: TradingSignal) -> None:
        """Validate trading signal parameters with comprehensive validation.

        Args:
            signal: Trading signal to validate

        Raises:
            ValueError: If signal validation fails with detailed error message
        """
        errors = []

        # Required fields validation
        if not signal.signal_id:
            errors.append("Signal ID is required")
        
        if not signal.asset:
            errors.append("Asset is required")
        elif len(signal.asset) < 4:
            errors.append(f"Asset must be a valid trading pair (minimum 4 characters), got: '{signal.asset}'")
        elif not signal.asset.isupper():
            errors.append(f"Asset must be uppercase (e.g., 'BTCUSDT'), got: '{signal.asset}'")
        elif not signal.asset.isalnum():
            errors.append(f"Asset must contain only alphanumeric characters, got: '{signal.asset}'")

        # Amount validation - must be positive and reasonable
        if signal.amount is None:
            errors.append("Amount is required")
        elif signal.amount <= 0:
            errors.append(f"Amount must be positive, got: {signal.amount}")
        elif signal.amount < Decimal("0.01"):
            errors.append(f"Amount must be at least 0.01 USDT, got: {signal.amount}")

        # Confidence validation - must be in valid range
        if signal.confidence is None:
            errors.append("Confidence is required")
        elif signal.confidence < 0:
            errors.append(f"Confidence must be non-negative, got: {signal.confidence}")
        elif signal.confidence > 1:
            errors.append(f"Confidence must not exceed 1.0, got: {signal.confidence}")

        # Market data snapshot validation
        if not signal.market_data_snapshot:
            errors.append("Market data snapshot is required")
        elif not signal.market_data_snapshot.price:
            errors.append("Market data snapshot must include price")
        elif signal.market_data_snapshot.price <= 0:
            errors.append(f"Market data snapshot price must be positive, got: {signal.market_data_snapshot.price}")

        # Signal type validation
        if not signal.signal_type:
            errors.append("Signal type is required")
        elif signal.signal_type.lower() not in {"buy", "sell"}:
            errors.append(f"Signal type must be 'buy' or 'sell', got: '{signal.signal_type}'")

        # Strategy ID validation
        if not signal.strategy_id:
            errors.append("Strategy ID is required")
        elif len(signal.strategy_id) > 100:
            errors.append(f"Strategy ID must not exceed 100 characters, got: {len(signal.strategy_id)}")

        # Timestamp validation
        if not signal.timestamp:
            errors.append("Timestamp is required")

        # Raise comprehensive error if any validation failed
        if errors:
            error_message = "Signal validation failed: " + "; ".join(errors)
            raise ValueError(error_message)

    async def _cancel_existing_orders(self, asset: str, signal: TradingSignal) -> None:
        """Cancel existing orders for asset based on cancellation strategy.

        Args:
            asset: Trading pair symbol
            signal: New trading signal
        """
        trace_id = signal.trace_id or get_or_create_trace_id()

        try:
            # Get pending orders for this asset
            from ..config.database import DatabaseConnection

            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, order_id, side, status
                FROM orders
                WHERE asset = $1 AND status IN ('pending', 'partially_filled')
            """
            rows = await pool.fetch(query, asset)

            if not rows:
                logger.debug("no_orders_to_cancel", asset=asset, trace_id=trace_id)
                return

            # Determine which orders to cancel
            cancel_opposite_only = settings.order_manager_cancel_opposite_orders_only
            signal_side = signal.signal_type.lower()

            orders_to_cancel = []
            for row in rows:
                order_side = row["side"].lower()
                should_cancel = False

                if cancel_opposite_orders_only:
                    # Only cancel opposite direction orders
                    if (signal_side == "buy" and order_side == "sell") or (
                        signal_side == "sell" and order_side == "buy"
                    ):
                        should_cancel = True
                else:
                    # Cancel all pending orders for this asset
                    should_cancel = True

                if should_cancel:
                    orders_to_cancel.append(row)

            # Cancel orders
            from ..services.order_executor import OrderExecutor
            if self._order_executor is None:
                self._order_executor = OrderExecutor()
            for order_row in orders_to_cancel:
                try:
                    await self._order_executor.cancel_order(
                        order_id=order_row["order_id"],
                        asset=asset,
                        trace_id=trace_id,
                    )
                    logger.info(
                        "order_cancelled_for_new_signal",
                        order_id=order_row["order_id"],
                        asset=asset,
                        reason="new_signal",
                        trace_id=trace_id,
                    )
                except Exception as e:
                    logger.error(
                        "order_cancellation_failed",
                        order_id=order_row["order_id"],
                        asset=asset,
                        error=str(e),
                        trace_id=trace_id,
                    )

        except Exception as e:
            logger.error(
                "cancel_existing_orders_error",
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            # Don't fail signal processing if cancellation fails
            pass

    async def _create_signal_order_relationship(
        self, signal_id: UUID, order_id: UUID, trace_id: Optional[str]
    ) -> None:
        """Create signal-order relationship record.

        Args:
            signal_id: Trading signal ID
            order_id: Order ID
            trace_id: Trace ID for logging
        """
        try:
            from ..config.database import DatabaseConnection

            pool = await DatabaseConnection.get_pool()
            query = """
                INSERT INTO signal_order_relationships
                (signal_id, order_id, relationship_type, created_at)
                VALUES ($1, $2, 'one_to_one', NOW())
                ON CONFLICT (signal_id, order_id) DO NOTHING
            """
            await pool.execute(query, str(signal_id), str(order_id))

            logger.debug(
                "signal_order_relationship_created",
                signal_id=str(signal_id),
                order_id=str(order_id),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "signal_order_relationship_failed",
                signal_id=str(signal_id),
                order_id=str(order_id),
                error=str(e),
                trace_id=trace_id,
            )
            # Don't fail order creation if relationship creation fails
            pass

    async def _publish_signal_rejection_event(
        self,
        signal: TradingSignal,
        rejection_reason: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Publish rejection event for a signal that was rejected before order creation.

        Args:
            signal: Trading signal that was rejected
            rejection_reason: Reason for rejection
            trace_id: Optional trace ID
        """
        try:
            # Create a minimal order-like object for event publishing
            # Since no order was created, we'll use signal information
            from uuid import uuid4
            from datetime import datetime
            from decimal import Decimal

            # Create a temporary order object for event publishing
            # This represents a "rejected" order that was never created
            rejected_order = Order(
                id=uuid4(),
                order_id=f"REJECTED-{signal.signal_id}",
                signal_id=signal.signal_id,
                asset=signal.asset,
                side="Buy" if signal.signal_type.lower() == "buy" else "Sell",
                order_type="Market",  # Default, not important for rejection
                quantity=Decimal("0"),  # No quantity since order wasn't created
                price=None,
                status="rejected",
                filled_quantity=Decimal("0"),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                trace_id=trace_id,
                is_dry_run=False,
            )

            # Prepare signal information
            signal_info = {
                "signal_id": str(signal.signal_id),
                "signal_type": signal.signal_type,
                "asset": signal.asset,
                "amount": float(signal.amount),
                "confidence": float(signal.confidence),
                "strategy_id": signal.strategy_id,
                "model_version": signal.model_version,
                "is_warmup": signal.is_warmup,
            }

            # Prepare market conditions from signal snapshot if available
            market_conditions = None
            if signal.market_data_snapshot:
                market_conditions = {
                    "price": float(signal.market_data_snapshot.price),
                    "spread": float(signal.market_data_snapshot.spread) if signal.market_data_snapshot.spread else None,
                    "volume_24h": float(signal.market_data_snapshot.volume_24h) if signal.market_data_snapshot.volume_24h else None,
                    "volatility": float(signal.market_data_snapshot.volatility) if signal.market_data_snapshot.volatility else None,
                    "timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
                }

            # Publish rejection event
            await self.event_publisher.publish_order_event(
                order=rejected_order,
                event_type="rejected",
                trace_id=trace_id,
                rejection_reason=rejection_reason,
                signal_info=signal_info,
                market_conditions=market_conditions,
            )

            logger.info(
                "signal_rejection_event_published",
                signal_id=str(signal.signal_id),
                asset=signal.asset,
                rejection_reason=rejection_reason,
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "signal_rejection_event_publish_failed",
                signal_id=str(signal.signal_id),
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            # Don't raise - event publishing failure shouldn't block rejection processing

    async def _check_duplicate_signal(self, signal_id: UUID, trace_id: Optional[str]) -> Optional[dict]:
        """Check if signal has already been processed.

        Args:
            signal_id: Signal ID to check
            trace_id: Trace ID for logging

        Returns:
            None if signal is new, dict with status and order_id if duplicate
        """
        try:
            from ..config.database import DatabaseConnection
            pool = await DatabaseConnection.get_pool()
            
            # Check if signal has already been processed by looking at signal_order_relationships
            query = """
                SELECT sor.order_id, o.status
                FROM signal_order_relationships sor
                JOIN orders o ON sor.order_id = o.id
                WHERE sor.signal_id = $1
                ORDER BY sor.created_at DESC
                LIMIT 1
            """
            row = await pool.fetchrow(query, str(signal_id))
            
            if row:
                order_id = row["order_id"]
                order_status = row["status"]
                
                # If order was successfully created (not rejected), signal is duplicate
                if order_status not in ("rejected", "dry_run"):
                    return {
                        "status": "succeeded",
                        "order_id": order_id,
                    }
                else:
                    # Previous attempt failed, allow retry
                    return {
                        "status": "failed",
                        "order_id": order_id,
                    }
            
            return None

        except Exception as e:
            logger.error(
                "duplicate_signal_check_error",
                signal_id=str(signal_id),
                error=str(e),
                trace_id=trace_id,
            )
            # On error, allow processing to continue (fail open)
            return None

    async def _mark_signal_processing_status(
        self,
        signal_id: UUID,
        status: str,
        error_message: Optional[str],
        trace_id: Optional[str],
        order_id: Optional[UUID] = None,
    ) -> None:
        """Mark signal processing status for tracking.

        This is a lightweight tracking mechanism. The actual status is tracked
        via signal_order_relationships and orders tables, but we can add additional
        tracking here if needed.

        Args:
            signal_id: Signal ID
            status: Processing status ('succeeded', 'failed', 'processing')
            error_message: Error message if failed
            trace_id: Trace ID for logging
            order_id: Order ID if succeeded
        """
        # For now, we rely on signal_order_relationships table for tracking
        # This method can be extended to add a dedicated signal_processing_status table
        # if more detailed tracking is needed
        logger.debug(
            "signal_processing_status_marked",
            signal_id=str(signal_id),
            status=status,
            order_id=str(order_id) if order_id else None,
            error_message=error_message,
            trace_id=trace_id,
        )

