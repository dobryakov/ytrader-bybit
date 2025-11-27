"""Order executor service for creating, modifying, and cancelling orders on Bybit."""

from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

from ..config.settings import settings
from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..models.order import Order
from ..utils.bybit_client import get_bybit_client
from ..exceptions import OrderExecutionError, BybitAPIError

logger = get_logger(__name__)


class OrderExecutor:
    """Service for executing orders on Bybit exchange via REST API."""

    async def create_order(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
        trace_id: Optional[str] = None,
    ) -> Optional[Order]:
        """Create an order on Bybit exchange.

        Args:
            signal: Trading signal that triggered this order
            order_type: Order type ('Market' or 'Limit')
            quantity: Order quantity in base currency
            price: Limit order price (None for market orders)
            trace_id: Trace ID for request tracking

        Returns:
            Created Order object if successful, None if dry-run mode

        Raises:
            OrderExecutionError: If order creation fails
        """
        trace_id = trace_id or signal.trace_id
        signal_id = signal.signal_id
        asset = signal.asset
        side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

        logger.info(
            "order_creation_started",
            signal_id=str(signal_id),
            asset=asset,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            price=float(price) if price else None,
            trace_id=trace_id,
        )

        # Check if dry-run mode
        if settings.order_manager_enable_dry_run:
            logger.info(
                "order_creation_dry_run",
                signal_id=str(signal_id),
                asset=asset,
                side=side,
                order_type=order_type,
                quantity=float(quantity),
                price=float(price) if price else None,
                trace_id=trace_id,
            )
            # Create dry-run order in database
            return await self._create_dry_run_order(signal, order_type, quantity, price, trace_id)

        try:
            # Prepare order parameters for Bybit API
            bybit_params = self._prepare_bybit_order_params(
                signal=signal,
                order_type=order_type,
                quantity=quantity,
                price=price,
            )

            # Call Bybit API to create order
            bybit_client = get_bybit_client()
            endpoint = "/v5/order/create"
            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)

            # Parse response
            result = response.get("result", {})
            bybit_order_id = result.get("orderId")

            if not bybit_order_id:
                error_msg = "Bybit API did not return order ID"
                logger.error(
                    "order_creation_failed_no_order_id",
                    signal_id=str(signal_id),
                    response=response,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)

            # Create order record in database
            order = await self._save_order_to_database(
                signal=signal,
                bybit_order_id=bybit_order_id,
                order_type=order_type,
                quantity=quantity,
                price=price,
                trace_id=trace_id,
            )

            logger.info(
                "order_creation_complete",
                signal_id=str(signal_id),
                asset=asset,
                order_id=str(order.id),
                bybit_order_id=bybit_order_id,
                trace_id=trace_id,
            )

            return order

        except BybitAPIError as e:
            logger.error(
                "order_creation_bybit_error",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(f"Bybit API error: {e}") from e
        except Exception as e:
            logger.error(
                "order_creation_failed",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(f"Order creation failed: {e}") from e

    async def cancel_order(
        self,
        order_id: str,
        asset: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        """Cancel an order on Bybit exchange.

        Args:
            order_id: Bybit order ID
            asset: Trading pair symbol
            trace_id: Trace ID for request tracking

        Returns:
            True if cancellation successful, False otherwise
        """
        logger.info(
            "order_cancellation_started",
            order_id=order_id,
            asset=asset,
            trace_id=trace_id,
        )

        # Check if dry-run mode
        if settings.order_manager_enable_dry_run:
            logger.info(
                "order_cancellation_dry_run",
                order_id=order_id,
                asset=asset,
                trace_id=trace_id,
            )
            # Update order status in database
            await self._update_order_status_in_db(order_id, "cancelled", trace_id)
            return True

        try:
            # Prepare cancellation parameters
            bybit_params = {
                "category": "linear",
                "symbol": asset,
                "orderId": order_id,
            }

            # Call Bybit API to cancel order
            bybit_client = get_bybit_client()
            endpoint = "/v5/order/cancel"
            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)

            # Update order status in database
            await self._update_order_status_in_db(order_id, "cancelled", trace_id)

            logger.info(
                "order_cancellation_complete",
                order_id=order_id,
                asset=asset,
                trace_id=trace_id,
            )

            return True

        except BybitAPIError as e:
            logger.error(
                "order_cancellation_bybit_error",
                order_id=order_id,
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            # Order might already be filled or cancelled
            return False
        except Exception as e:
            logger.error(
                "order_cancellation_failed",
                order_id=order_id,
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            return False

    def _prepare_bybit_order_params(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
    ) -> dict:
        """Prepare order parameters for Bybit API.

        Args:
            signal: Trading signal
            order_type: Order type ('Market' or 'Limit')
            quantity: Order quantity
            price: Limit order price

        Returns:
            Dictionary with Bybit API parameters
        """
        from ..services.order_type_selector import OrderTypeSelector

        selector = OrderTypeSelector()
        asset = signal.asset
        side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

        params = {
            "category": "linear",
            "symbol": asset,
            "side": side,
            "orderType": order_type,
            "qty": str(quantity),
            "timeInForce": selector.get_time_in_force(order_type),
        }

        # Add price for limit orders
        if order_type == "Limit" and price:
            params["price"] = str(price)

        # Add post_only flag for limit orders
        if selector.should_use_post_only(order_type):
            params["postOnly"] = True

        # Add reduce_only flag if needed
        # TODO: Check position and set reduce_only if appropriate
        # For now, we'll skip this as it requires position lookup

        return params

    async def _create_dry_run_order(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
        trace_id: Optional[str],
    ) -> Order:
        """Create a dry-run order (simulated, not sent to Bybit).

        Args:
            signal: Trading signal
            order_type: Order type
            quantity: Order quantity
            price: Limit order price
            trace_id: Trace ID

        Returns:
            Order object with dry_run status
        """
        order_id = f"DRY-RUN-{uuid4()}"
        side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

        order = Order(
            id=uuid4(),
            order_id=order_id,
            signal_id=signal.signal_id,
            asset=signal.asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="dry_run",
            trace_id=trace_id,
            is_dry_run=True,
        )

        # Save to database
        await self._save_order_to_database(
            signal=signal,
            bybit_order_id=order_id,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trace_id=trace_id,
            is_dry_run=True,
        )

        return order

    async def _save_order_to_database(
        self,
        signal: TradingSignal,
        bybit_order_id: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
        trace_id: Optional[str],
        is_dry_run: bool = False,
    ) -> Order:
        """Save order to database.

        Args:
            signal: Trading signal
            bybit_order_id: Bybit order ID
            order_type: Order type
            quantity: Order quantity
            price: Limit order price
            trace_id: Trace ID
            is_dry_run: Whether this is a dry-run order

        Returns:
            Order object
        """
        try:
            pool = await DatabaseConnection.get_pool()
            side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"
            status = "dry_run" if is_dry_run else "pending"

            query = """
                INSERT INTO orders
                (id, order_id, signal_id, asset, side, order_type, quantity, price,
                 status, filled_quantity, created_at, updated_at, trace_id, is_dry_run)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12)
                RETURNING id, order_id, signal_id, asset, side, order_type, quantity, price,
                          status, filled_quantity, average_price, fees, created_at, updated_at,
                          executed_at, trace_id, is_dry_run
            """
            order_uuid = uuid4()
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
                status,
                "0",
                trace_id,
                is_dry_run,
            )

            order_data = dict(row)
            order = Order.from_dict(order_data)

            logger.debug(
                "order_saved_to_database",
                order_id=str(order.id),
                bybit_order_id=bybit_order_id,
                trace_id=trace_id,
            )

            return order

        except Exception as e:
            logger.error(
                "order_save_to_database_failed",
                bybit_order_id=bybit_order_id,
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(f"Failed to save order to database: {e}") from e

    async def _update_order_status_in_db(
        self,
        bybit_order_id: str,
        status: str,
        trace_id: Optional[str],
    ) -> None:
        """Update order status in database.

        Args:
            bybit_order_id: Bybit order ID
            status: New status
            trace_id: Trace ID
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                UPDATE orders
                SET status = $1, updated_at = NOW()
                WHERE order_id = $2
            """
            await pool.execute(query, status, bybit_order_id)

            logger.debug(
                "order_status_updated",
                bybit_order_id=bybit_order_id,
                status=status,
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "order_status_update_failed",
                bybit_order_id=bybit_order_id,
                status=status,
                error=str(e),
                trace_id=trace_id,
            )

