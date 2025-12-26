"""Signal processor service for processing trading signals and making order decisions."""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID

from ..config.logging import get_logger
from ..config.settings import settings
from ..models.trading_signal import TradingSignal
from ..models.order import Order
from ..models.position import Position
from ..models.signal_order_rel import SignalOrderRelationship
from ..services.order_type_selector import OrderTypeSelector
from ..services.quantity_calculator import QuantityCalculator
from ..services.position_manager_client import PositionManagerClient
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
        self.position_manager_client = PositionManagerClient()
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
        
        # Initialize variables for error handling (will be set during processing)
        order_type = None
        quantity = None
        limit_price = None
        
        # Ensure settings is accessible (import at module level should handle this,
        # but this explicit reference ensures it's available in exception handlers)
        _ = settings

        # Use Redis distributed lock to prevent race condition when processing the same signal concurrently
        # Generate lock key from signal_id
        lock_key = f"signal_lock:{signal_id}"
        
        from ..services.redis_lock import redis_lock
        
        # Acquire Redis lock for this signal_id
        async with redis_lock.acquire(lock_key, timeout=300, blocking_timeout=0.1) as acquired:
            if not acquired:
                # Lock is already held by another process, skip this signal
                logger.debug(
                    "signal_processing_lock_busy",
                    signal_id=str(signal_id),
                    lock_key=lock_key,
                    trace_id=trace_id,
                    message="Signal is already being processed by another worker, skipping",
                )
                return None
            
            logger.info(
                "signal_processing_lock_acquired",
                signal_id=str(signal_id),
                lock_key=lock_key,
                trace_id=trace_id,
            )
            
            try:
                # Step 0: Check for duplicate signal (now protected by lock)
                logger.debug(
                    "checking_duplicate_signal",
                    signal_id=str(signal_id),
                    trace_id=trace_id,
                )
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
                        async with pool.acquire() as conn:
                            query = "SELECT * FROM orders WHERE id = $1"
                            row = await conn.fetchrow(query, duplicate_check["order_id"])
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
                logger.debug(
                    "cancelling_existing_orders",
                    signal_id=str(signal_id),
                    asset=asset,
                    trace_id=trace_id,
                )
                await self._cancel_existing_orders(asset, signal)

                # Step 2: Calculate order quantity
                logger.debug(
                    "calculating_order_quantity",
                    signal_id=str(signal_id),
                    asset=asset,
                    trace_id=trace_id,
                )
                quantity = await self.quantity_calculator.calculate_quantity(signal)

                # Step 3: Select order type and price
                order_type, limit_price = await self.order_type_selector.select_order_type(signal)
                order_price = limit_price or signal.market_data_snapshot.price

                # Step 4: Get current position from Position Manager service
                current_position = await self.position_manager_client.get_position(asset, mode="one-way", trace_id=trace_id)

                # Step 4.1: Check take profit / stop loss exit rules
                # Skip exit check if this is already an exit signal from model-service exit strategy
                # or if this is a close signal created by order-manager (to avoid infinite recursion)
                # to avoid double processing (model-service exit strategy already evaluated PnL)
                is_exit_signal = signal.metadata and signal.metadata.get("exit_strategy", False)
                is_close_signal = signal.metadata and signal.metadata.get("exit_reason") is not None
                if not is_exit_signal and not is_close_signal:
                    exit_check = await self.risk_manager.check_take_profit_stop_loss(asset, current_position, trace_id=trace_id)
                    if exit_check and exit_check.get("should_close"):
                        # Position should be closed due to take profit or stop loss
                        # Create SELL signal to close position
                        exit_reason = exit_check.get("reason", "unknown")
                        logger.info(
                            "position_exit_rule_triggered",
                            asset=asset,
                            exit_reason=exit_reason,
                            unrealized_pnl_pct=exit_check.get("unrealized_pnl_pct"),
                            threshold_pct=exit_check.get("threshold_pct"),
                            trace_id=trace_id,
                        )
                        
                        # Create close signal
                        if current_position and current_position.size != 0:
                            close_signal = await self._create_close_position_signal(
                                signal=signal,
                                position=current_position,
                                exit_reason=exit_reason,
                                exit_check=exit_check,
                                trace_id=trace_id,
                            )
                            if close_signal:
                                # Process close signal instead of original signal
                                logger.info(
                                    "processing_close_signal_instead_of_original",
                                    original_signal_id=str(signal_id),
                                    close_signal_id=str(close_signal.signal_id),
                                    exit_reason=exit_reason,
                                    trace_id=trace_id,
                                )
                                # Recursively process close signal
                                return await self._process_signal_internal(close_signal)
                        # If position is None or size is 0, continue with original signal
                else:
                    # This is an exit signal from model-service exit strategy or a close signal
                    # Skip order-manager exit check to avoid double processing or infinite recursion
                    if is_exit_signal:
                        logger.debug(
                            "skipping_exit_check_for_model_service_exit_signal",
                            asset=asset,
                            signal_id=str(signal_id),
                            trace_id=trace_id,
                            reason="Signal is already an exit signal from model-service exit strategy",
                        )
                    elif is_close_signal:
                        logger.debug(
                            "skipping_exit_check_for_close_signal",
                            asset=asset,
                            signal_id=str(signal_id),
                            exit_reason=signal.metadata.get("exit_reason") if signal.metadata else None,
                            trace_id=trace_id,
                            reason="Signal is a close signal created by order-manager, skipping exit check to avoid recursion",
                        )

                # Step 4.5: Close position if opposite signal and feature enabled
                if settings.order_manager_close_position_before_opposite_signal:
                    current_position = await self._handle_opposite_signal_position_closure(
                        signal=signal,
                        current_position=current_position,
                        asset=asset,
                        trace_id=trace_id,
                    )

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

                # 5b: Order size check and adaptation
                original_quantity = quantity
                adapted_quantity, was_adapted = await self.risk_manager.check_and_adapt_order_size(signal, quantity, order_price)
                quantity = adapted_quantity  # Always use the returned quantity (may be adapted or original)
                if was_adapted:
                    logger.info(
                        "quantity_adapted_for_order_size_limit",
                        signal_id=str(signal_id),
                        asset=asset,
                        original_quantity=float(original_quantity),
                        adapted_quantity=float(adapted_quantity),
                        trace_id=trace_id,
                    )

                # 5c: Position size check
                order_side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
                self.risk_manager.check_position_size(asset, current_position, quantity, order_side, trace_id=trace_id)

                # Step 6: Create order via executor
                logger.info(
                    "creating_order",
                    signal_id=str(signal_id),
                    asset=asset,
                    order_type=order_type,
                    quantity=quantity,
                    price=limit_price,
                    trace_id=trace_id,
                )
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
                logger.info(
                    "order_created",
                    signal_id=str(signal_id),
                    asset=asset,
                    order_id=str(order.id) if order else None,
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
                
                # Save rejected order to database before publishing event
                # Note: quantity, order_type, limit_price are guaranteed to be defined at this point
                # because RiskLimitError is thrown after Step 5c which comes after their calculation
                rejected_order = await self._save_rejected_order(
                    signal=signal,
                    order_type=order_type or "Market",
                    quantity=quantity,
                    price=limit_price,
                    rejection_reason=error_msg,
                    trace_id=trace_id,
                )
                
                # Mark signal as failed in tracking
                await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
                
                # Create signal-order relationship if order was saved
                if rejected_order:
                    await self._create_signal_order_relationship(signal_id, rejected_order.id, trace_id)
                    # Publish rejection event with saved order
                    await self._publish_signal_rejection_event(
                        signal=signal,
                        rejected_order=rejected_order,
                        rejection_reason=error_msg,
                        trace_id=trace_id,
                    )
                else:
                    # Fallback: publish event without saved order
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
                
                # Save rejected order to database before publishing event
                # Use calculated values if available, otherwise use defaults
                rejected_order = await self._save_rejected_order(
                    signal=signal,
                    order_type=order_type or "Market",
                    quantity=quantity,
                    price=limit_price,
                    rejection_reason=error_msg,
                    trace_id=trace_id,
                )
                
                # Mark signal as failed in tracking
                await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
                
                # Create signal-order relationship if order was saved
                if rejected_order:
                    await self._create_signal_order_relationship(signal_id, rejected_order.id, trace_id)
                    # Publish rejection event with saved order
                    await self._publish_signal_rejection_event(
                        signal=signal,
                        rejected_order=rejected_order,
                        rejection_reason=error_msg,
                        trace_id=trace_id,
                    )
                else:
                    # Fallback: try to publish event without saved order
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
                try:
                    await self._mark_signal_processing_status(signal_id, "failed", error_msg, trace_id)
                except Exception as tracking_error:
                    logger.error(
                        "signal_tracking_failed",
                        signal_id=str(signal_id),
                        error=str(tracking_error),
                        trace_id=trace_id,
                        exc_info=True,
                    )
                # Publish rejection event
                try:
                    await self._publish_signal_rejection_event(
                        signal=signal,
                        rejection_reason=error_msg,
                        trace_id=trace_id,
                    )
                except Exception as publish_error:
                    logger.error(
                        "signal_rejection_event_publish_failed_in_exception",
                        signal_id=str(signal_id),
                        error=str(publish_error),
                        trace_id=trace_id,
                        exc_info=True,
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

                if cancel_opposite_only:
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

    async def _save_rejected_order(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Optional[Decimal],
        price: Optional[Decimal],
        rejection_reason: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Save rejected order to database.

        Args:
            signal: Trading signal that was rejected
            order_type: Order type that was attempted
            quantity: Calculated quantity (may be None if calculation failed)
            price: Limit price if applicable
            rejection_reason: Reason for rejection
            trace_id: Optional trace ID

        Returns:
            Saved Order object or None if save failed
        """
        try:
            from ..config.database import DatabaseConnection
            from uuid import uuid4
            from decimal import Decimal

            # Validate rejection_reason is not empty
            if not rejection_reason or not rejection_reason.strip():
                logger.warning(
                    "empty_rejection_reason",
                    signal_id=str(signal.signal_id),
                    trace_id=trace_id,
                )
                rejection_reason = "Unknown rejection reason"

            pool = await DatabaseConnection.get_pool()
            # Database constraint requires 'Buy' or 'SELL' (not 'Sell')
            side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
            
            # Use calculated quantity if available, otherwise estimate from signal amount
            if quantity is None or quantity <= 0:
                # Fallback: estimate quantity from signal amount and current price
                if signal.market_data_snapshot and signal.market_data_snapshot.price:
                    estimated_qty = Decimal(str(float(signal.amount) / float(signal.market_data_snapshot.price)))
                    # Ensure minimum quantity > 0
                    quantity = max(estimated_qty, Decimal("0.000001"))
                else:
                    # Last resort: use minimal quantity
                    quantity = Decimal("0.000001")

            order_uuid = uuid4()
            bybit_order_id = f"REJECTED-{signal.signal_id}"
            
            # Log before insert for debugging
            logger.debug(
                "saving_rejected_order",
                order_id=bybit_order_id,
                signal_id=str(signal.signal_id),
                side=side,
                rejection_reason=rejection_reason,
                trace_id=trace_id,
            )
            
            # Insert or update rejected order
            # Always update rejection_reason on conflict (don't preserve old value)
            query = """
                INSERT INTO orders
                (id, order_id, signal_id, asset, side, order_type, quantity, price,
                 status, filled_quantity, created_at, updated_at, trace_id, is_dry_run, rejection_reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12, $13)
                ON CONFLICT (order_id) DO UPDATE SET
                    rejection_reason = EXCLUDED.rejection_reason,
                    updated_at = NOW()
                RETURNING id, order_id, signal_id, asset, side, order_type, quantity, price,
                          status, filled_quantity, average_price, fees, created_at, updated_at,
                          executed_at, trace_id, is_dry_run, rejection_reason
            """
            row = await pool.fetchrow(
                query,
                str(order_uuid),
                bybit_order_id,
                str(signal.signal_id),
                signal.asset,
                side,
                order_type,
                str(quantity),
                str(price) if price else None,
                "rejected",
                "0",
                trace_id,
                False,
                rejection_reason,
            )
            
            # ON CONFLICT should always return a row (either inserted or updated)
            if row is None:
                logger.error(
                    "rejected_order_insert_returned_none",
                    order_id=bybit_order_id,
                    signal_id=str(signal.signal_id),
                    rejection_reason=rejection_reason,
                    trace_id=trace_id,
                )
                # Fallback: try to update existing order
                query_update_existing = """
                    UPDATE orders
                    SET rejection_reason = $1, updated_at = NOW()
                    WHERE order_id = $2
                    RETURNING id, order_id, signal_id, asset, side, order_type, quantity, price,
                              status, filled_quantity, average_price, fees, created_at, updated_at,
                              executed_at, trace_id, is_dry_run, rejection_reason
                """
                row = await pool.fetchrow(query_update_existing, rejection_reason, bybit_order_id)
                if row is None:
                    logger.error(
                        "rejected_order_update_failed",
                        order_id=bybit_order_id,
                        signal_id=str(signal.signal_id),
                        trace_id=trace_id,
                    )
                    return None

            order_data = dict(row)
            # Verify rejection_reason was saved
            db_rejection_reason = order_data.get("rejection_reason")
            if not db_rejection_reason:
                logger.error(
                    "rejection_reason_not_saved_in_db",
                    order_id=bybit_order_id,
                    signal_id=str(signal.signal_id),
                    rejection_reason_param=rejection_reason,
                    db_rejection_reason=db_rejection_reason,
                    side=side,
                    order_type=order_type,
                    trace_id=trace_id,
                )
                # Try direct update as last resort
                query_direct_update = """
                    UPDATE orders
                    SET rejection_reason = $1, updated_at = NOW()
                    WHERE order_id = $2
                    RETURNING rejection_reason
                """
                update_row = await pool.fetchrow(query_direct_update, rejection_reason, bybit_order_id)
                if update_row:
                    order_data["rejection_reason"] = update_row["rejection_reason"]
                    logger.info(
                        "rejection_reason_updated_via_fallback",
                        order_id=bybit_order_id,
                        signal_id=str(signal.signal_id),
                        trace_id=trace_id,
                    )
                else:
                    logger.error(
                        "rejection_reason_fallback_update_failed",
                        order_id=bybit_order_id,
                        signal_id=str(signal.signal_id),
                        trace_id=trace_id,
                    )
            
            order = Order.from_dict(order_data)

            logger.info(
                "rejected_order_saved_to_database",
                order_id=str(order.id),
                signal_id=str(signal.signal_id),
                asset=signal.asset,
                rejection_reason=rejection_reason,
                db_rejection_reason=db_rejection_reason,
                trace_id=trace_id,
            )

            return order

        except Exception as e:
            logger.error(
                "rejected_order_save_failed",
                signal_id=str(signal.signal_id),
                asset=signal.asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def _publish_signal_rejection_event(
        self,
        signal: TradingSignal,
        rejection_reason: str,
        trace_id: Optional[str] = None,
        rejected_order: Optional[Order] = None,
    ) -> None:
        """
        Publish rejection event for a signal that was rejected before order creation.

        Args:
            signal: Trading signal that was rejected
            rejection_reason: Reason for rejection
            trace_id: Optional trace ID
            rejected_order: Optional rejected order object (if already saved to database)
        """
        try:
            # Use provided order or create a minimal order object for event publishing
            if rejected_order:
                order_for_event = rejected_order
            else:
                # Create a minimal order-like object for event publishing
                from uuid import uuid4
                from datetime import datetime
                from decimal import Decimal

                # Calculate estimated quantity from signal amount
                quantity = Decimal("0.000001")  # Minimum quantity
                if signal.market_data_snapshot and signal.market_data_snapshot.price:
                    estimated_qty = Decimal(str(float(signal.amount) / float(signal.market_data_snapshot.price)))
                    quantity = max(estimated_qty, Decimal("0.000001"))

                order_for_event = Order(
                    id=uuid4(),
                    order_id=f"REJECTED-{signal.signal_id}",
                    signal_id=signal.signal_id,
                    asset=signal.asset,
                    side="Buy" if signal.signal_type.lower() == "buy" else "SELL",
                    order_type="Market",
                    quantity=quantity,
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
                order=order_for_event,
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

    async def _handle_opposite_signal_position_closure(
        self,
        signal: TradingSignal,
        current_position: Optional[Position],
        asset: str,
        trace_id: Optional[str],
    ) -> Optional[Position]:
        """Handle position closure when opposite signal arrives.

        Args:
            signal: Trading signal
            current_position: Current position for asset
            asset: Trading pair symbol
            trace_id: Trace ID for logging

        Returns:
            Updated position (may be None if closed)
        """
        # Check if position exists and is significant
        if not current_position:
            return None

        # Check position data freshness
        current_position = await self._check_and_refresh_position_if_stale(
            current_position=current_position,
            asset=asset,
            trace_id=trace_id,
        )

        if not current_position:
            return None

        position_size = abs(current_position.size)
        min_size_threshold = Decimal(str(settings.order_manager_position_close_min_size_threshold))

        if position_size < min_size_threshold:
            logger.debug(
                "position_too_small_to_close",
                asset=asset,
                position_size=float(position_size),
                min_threshold=float(min_size_threshold),
                trace_id=trace_id,
            )
            return current_position

        # Check if signal is opposite to position
        signal_type = signal.signal_type.lower()
        is_opposite = False

        if current_position.size > 0 and signal_type == "sell":
            # Long position + SELL signal = opposite
            is_opposite = True
        elif current_position.size < 0 and signal_type == "buy":
            # Short position + BUY signal = opposite
            is_opposite = True

        if not is_opposite:
            return current_position

        logger.info(
            "opposite_signal_detected_closing_position",
            asset=asset,
            signal_type=signal_type,
            position_size=float(current_position.size),
            trace_id=trace_id,
        )

        # Close position
        try:
            await self._close_position_before_new_order(
                asset=asset,
                position=current_position,
                signal=signal,
                trace_id=trace_id,
            )

            # Wait for closure if configured
            if settings.order_manager_position_close_wait_mode == "polling":
                current_position = await self._wait_for_position_closure(
                    asset=asset,
                    trace_id=trace_id,
                )
            else:
                # Re-fetch position to get updated state (best effort)
                try:
                    current_position = await self.position_manager_client.get_position(
                        asset, mode="one-way", trace_id=trace_id
                    )
                except Exception as e:
                    logger.warning(
                        "position_refetch_after_close_failed",
                        asset=asset,
                        error=str(e),
                        trace_id=trace_id,
                        reason="Failed to refetch position after close order, continuing",
                    )
                    # Assume position is closed or will be closed
                    current_position = None

        except Exception as e:
            logger.error(
                "position_closure_failed",
                asset=asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
                reason="Failed to close position before opposite signal, continuing with new order",
            )
            # Continue with original position - new order will handle it with reduceOnly

        return current_position

    async def _check_and_refresh_position_if_stale(
        self,
        current_position: Position,
        asset: str,
        trace_id: Optional[str],
    ) -> Optional[Position]:
        """Check position data freshness and refresh from Bybit if stale.

        Args:
            current_position: Current position from Position Manager
            asset: Trading pair symbol
            trace_id: Trace ID for logging

        Returns:
            Fresh position (from Bybit if stale, or original if fresh)
        """
        from datetime import datetime, timezone

        if not settings.order_manager_enable_bybit_position_fallback:
            return current_position

        # Check position age
        now = datetime.now(timezone.utc)
        position_updated = current_position.last_updated
        if position_updated.tzinfo is None:
            # Assume UTC if timezone-naive
            position_updated = position_updated.replace(tzinfo=timezone.utc)

        age_seconds = (now - position_updated).total_seconds()
        max_age = settings.order_manager_position_max_age_seconds

        if age_seconds <= max_age:
            logger.debug(
                "position_data_fresh",
                asset=asset,
                age_seconds=age_seconds,
                max_age=max_age,
                trace_id=trace_id,
            )
            return current_position

        logger.warning(
            "position_data_stale_refreshing_from_bybit",
            asset=asset,
            age_seconds=age_seconds,
            max_age=max_age,
            last_updated=position_updated.isoformat(),
            trace_id=trace_id,
        )

        # Fetch fresh position from Bybit
        try:
            fresh_position = await self.position_manager_client.get_position_from_bybit(
                asset=asset,
                trace_id=trace_id,
            )

            if fresh_position is None:
                logger.info(
                    "position_closed_on_bybit",
                    asset=asset,
                    trace_id=trace_id,
                    reason="Position not found on Bybit, likely closed",
                )
                return None

            # Log discrepancy if sizes differ significantly
            size_diff = abs(abs(fresh_position.size) - abs(current_position.size))
            size_diff_pct = (
                (size_diff / abs(current_position.size) * 100)
                if current_position.size != 0
                else 100.0
            )

            if size_diff > Decimal("0.0001"):  # Significant difference threshold
                logger.warning(
                    "position_size_discrepancy_detected",
                    asset=asset,
                    position_manager_size=float(current_position.size),
                    bybit_size=float(fresh_position.size),
                    size_diff=float(size_diff),
                    size_diff_pct=float(size_diff_pct),
                    age_seconds=age_seconds,
                    trace_id=trace_id,
                    reason="Position Manager data differs from Bybit, using fresh Bybit data",
                )

            logger.info(
                "position_refreshed_from_bybit",
                asset=asset,
                old_size=float(current_position.size),
                new_size=float(fresh_position.size),
                age_seconds=age_seconds,
                trace_id=trace_id,
            )

            # Trigger async sync to update Position Manager database
            if settings.order_manager_auto_sync_position_after_bybit_fetch:
                await self.position_manager_client.trigger_bybit_sync_async(
                    asset=asset,
                    force=True,
                    trace_id=trace_id,
                )

            return fresh_position

        except Exception as e:
            logger.error(
                "position_refresh_from_bybit_failed",
                asset=asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
                reason="Failed to refresh position from Bybit, using stale Position Manager data",
            )
            # Fallback to stale data rather than failing
            return current_position

    async def _create_close_position_signal(
        self,
        signal: TradingSignal,
        position: Position,
        exit_reason: str,
        exit_check: Dict[str, Any],
        trace_id: Optional[str],
    ) -> Optional[TradingSignal]:
        """Create a SELL signal to close position due to take profit or stop loss.

        Args:
            signal: Original trading signal (for context)
            position: Current position to close
            exit_reason: Exit reason ('take_profit' or 'stop_loss')
            exit_check: Exit check result from risk_manager
            trace_id: Trace ID for logging

        Returns:
            TradingSignal for closing position, or None if cannot create
        """
        from uuid import uuid4
        from datetime import datetime, timezone
        from decimal import Decimal

        # Determine close side based on position direction
        if position.size > 0:
            # Long position - need SELL order to close
            close_side = "sell"
        else:
            # Short position - need BUY order to close
            close_side = "buy"

        # Calculate close amount (position size in quote currency)
        # Use current price from signal's market data snapshot
        if signal.market_data_snapshot and signal.market_data_snapshot.price:
            current_price = Decimal(str(signal.market_data_snapshot.price))
            position_size_abs = abs(position.size)
            close_amount = float(position_size_abs * current_price)
        else:
            # Fallback: use position size directly (will be converted later)
            close_amount = float(abs(position.size))

        # Create close signal
        close_signal = TradingSignal(
            signal_id=uuid4(),  # New signal ID for close order
            asset=signal.asset,
            signal_type=close_side,
            amount=Decimal(str(close_amount)),
            confidence=1.0,  # Maximum confidence for exit rules
            strategy_id=signal.strategy_id,
            timestamp=datetime.now(timezone.utc),
            market_data_snapshot=signal.market_data_snapshot,
            trace_id=trace_id,
            model_version=None,  # Not from model
            is_warmup=False,
            metadata={
                "reasoning": f"{exit_reason}_triggered",
                "exit_reason": exit_reason,
                "unrealized_pnl_pct": exit_check.get("unrealized_pnl_pct"),
                "threshold_pct": exit_check.get("threshold_pct"),
                "position_size": float(position.size),
                "original_signal_id": str(signal.signal_id),
            },
        )

        logger.info(
            "close_position_signal_created",
            asset=signal.asset,
            exit_reason=exit_reason,
            close_side=close_side,
            position_size=float(position.size),
            close_amount=close_amount,
            trace_id=trace_id,
        )

        return close_signal

    async def _close_position_before_new_order(
        self,
        asset: str,
        position: Position,
        signal: TradingSignal,
        trace_id: Optional[str],
    ) -> None:
        """Close position by creating reduce-only order.

        Args:
            asset: Trading pair symbol
            position: Current position to close
            signal: Trading signal (for context)
            trace_id: Trace ID for logging
        """
        from ..services.order_executor import OrderExecutor

        # Determine opposite side for closing
        if position.size > 0:
            # Long position - need SELL order to close
            close_side = "sell"
        else:
            # Short position - need BUY order to close
            close_side = "buy"

        close_quantity = abs(position.size)

        logger.info(
            "closing_position_before_opposite_signal",
            asset=asset,
            position_size=float(position.size),
            close_side=close_side,
            close_quantity=float(close_quantity),
            trace_id=trace_id,
        )

        # Create a minimal signal for closing order (reuse signal's market data)
        close_signal = TradingSignal(
            signal_id=signal.signal_id,  # Reuse same signal ID for tracking
            asset=asset,
            signal_type=close_side,
            amount=signal.amount,  # Not used for close order, but required
            confidence=signal.confidence,  # Not used, but required
            strategy_id=signal.strategy_id,
            timestamp=signal.timestamp,
            market_data_snapshot=signal.market_data_snapshot,
            trace_id=trace_id,
            model_version=signal.model_version,
            is_warmup=signal.is_warmup,
            metadata=signal.metadata,
        )

        # Create close order with reduceOnly=True
        if self._order_executor is None:
            self._order_executor = OrderExecutor()

        try:
            close_order = await self._order_executor.create_order(
                signal=close_signal,
                order_type="Market",  # Use Market order for immediate execution
                quantity=close_quantity,
                price=None,  # Market order doesn't need price
                trace_id=trace_id,
                force_reduce_only=True,  # Force reduceOnly flag
            )

            logger.info(
                "position_close_order_created",
                asset=asset,
                close_order_id=str(close_order.id) if close_order else None,
                close_quantity=float(close_quantity),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "position_close_order_creation_failed",
                asset=asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise

    async def _wait_for_position_closure(
        self,
        asset: str,
        trace_id: Optional[str],
    ) -> Optional[Position]:
        """Wait for position closure using polling.

        Args:
            asset: Trading pair symbol
            trace_id: Trace ID for logging

        Returns:
            Updated position (None if closed, Position if still open)
        """
        import asyncio

        timeout_seconds = settings.order_manager_position_close_timeout_seconds
        poll_interval = 1.0  # Check every second
        max_iterations = int(timeout_seconds / poll_interval)

        logger.info(
            "waiting_for_position_closure",
            asset=asset,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            trace_id=trace_id,
        )

        for iteration in range(max_iterations):
            await asyncio.sleep(poll_interval)

            try:
                position = await self.position_manager_client.get_position(
                    asset, mode="one-way", trace_id=trace_id
                )

                if position is None:
                    logger.info(
                        "position_closed_confirmed",
                        asset=asset,
                        iterations=iteration + 1,
                        trace_id=trace_id,
                    )
                    return None

                position_size = abs(position.size)
                min_size_threshold = Decimal(
                    str(settings.order_manager_position_close_min_size_threshold)
                )

                if position_size < min_size_threshold:
                    logger.info(
                        "position_effectively_closed",
                        asset=asset,
                        position_size=float(position_size),
                        min_threshold=float(min_size_threshold),
                        iterations=iteration + 1,
                        trace_id=trace_id,
                    )
                    return None

            except Exception as e:
                logger.warning(
                    "position_closure_poll_error",
                    asset=asset,
                    iteration=iteration + 1,
                    error=str(e),
                    trace_id=trace_id,
                )
                # Continue polling

        # Timeout reached
        logger.warning(
            "position_closure_timeout",
            asset=asset,
            timeout_seconds=timeout_seconds,
            trace_id=trace_id,
            reason="Position closure timeout reached, proceeding with new order",
        )

        # Return current position (may still be open)
        try:
            return await self.position_manager_client.get_position(
                asset, mode="one-way", trace_id=trace_id
            )
        except Exception:
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

