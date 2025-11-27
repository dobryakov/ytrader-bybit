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
            # 5a: Balance check (skip in dry-run mode)
            if not settings.order_manager_enable_dry_run:
                await self.risk_manager.check_balance(signal, quantity, order_price)
            else:
                logger.debug(
                    "balance_check_skipped_dry_run",
                    signal_id=str(signal_id),
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

            logger.info(
                "signal_processing_complete",
                signal_id=str(signal_id),
                asset=asset,
                order_id=str(order.id) if order else None,
                trace_id=trace_id,
            )

            return order

        except RiskLimitError as e:
            logger.warning(
                "signal_rejected_risk_limit",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            return None
        except OrderExecutionError as e:
            logger.error(
                "signal_processing_failed",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            raise

    async def _process_asset_queue(self, asset: str) -> None:
        """Process signals from queue for a specific asset (FIFO).

        Args:
            asset: Trading pair symbol
        """
        queue = self._signal_queues[asset]
        logger.info("asset_queue_processor_started", asset=asset)

        while True:
            try:
                # Get signal from queue (blocks until available)
                signal = await queue.get()

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
                )
                queue.task_done()

    def _validate_signal(self, signal: TradingSignal) -> None:
        """Validate trading signal parameters.

        Args:
            signal: Trading signal to validate

        Raises:
            ValueError: If signal validation fails
        """
        # Required fields
        if not signal.signal_id:
            raise ValueError("Signal ID is required")
        if not signal.asset:
            raise ValueError("Asset is required")
        if not signal.amount or signal.amount <= 0:
            raise ValueError("Amount must be positive")
        if signal.confidence < 0 or signal.confidence > 1:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not signal.market_data_snapshot or not signal.market_data_snapshot.price:
            raise ValueError("Market data snapshot with price is required")

        # Asset format validation
        if len(signal.asset) < 4:
            raise ValueError("Asset must be a valid trading pair")

        # Signal type validation
        if signal.signal_type.lower() not in {"buy", "sell"}:
            raise ValueError("Signal type must be 'buy' or 'sell'")

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

