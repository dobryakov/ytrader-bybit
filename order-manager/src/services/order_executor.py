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
from ..publishers.order_event_publisher import OrderEventPublisher
from ..utils.bybit_client import get_bybit_client
from ..exceptions import OrderExecutionError, BybitAPIError

logger = get_logger(__name__)


class OrderExecutor:
    """Service for executing orders on Bybit exchange via REST API."""

    def __init__(self):
        """Initialize order executor."""
        self.event_publisher = OrderEventPublisher()

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

            # Log order parameters for debugging
            logger.info(
                "order_creation_bybit_params",
                signal_id=str(signal_id),
                asset=asset,
                bybit_params=bybit_params,
                trace_id=trace_id,
            )

            # Call Bybit API to create order
            bybit_client = get_bybit_client()
            endpoint = "/v5/order/create"
            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)

            # Parse response
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            # Handle specific error codes
            if ret_code != 0:
                # Error 30208: "The order price is higher than the maximum buying price"
                # This can happen for Market orders if account settings reject orders outside price limits
                if ret_code == 30208:
                    logger.warning(
                        "order_creation_price_limit_error",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        note="This may be due to account settings. Consider using Limit orders or checking /v5/account/set-limit-px-action",
                        trace_id=trace_id,
                    )
                    # Try to get price limits for debugging
                    try:
                        price_limit_response = await bybit_client.get(
                            "/v5/market/price-limit",
                            params={"category": "linear", "symbol": asset},
                            authenticated=False,
                        )
                        price_limits = price_limit_response.get("result", {})
                        logger.info(
                            "order_creation_price_limits",
                            signal_id=str(signal_id),
                            asset=asset,
                            buy_limit=price_limits.get("buyLmt"),
                            sell_limit=price_limits.get("sellLmt"),
                            trace_id=trace_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "order_creation_price_limit_fetch_failed",
                            signal_id=str(signal_id),
                            error=str(e),
                            trace_id=trace_id,
                        )
                    
                    # Try to enable automatic price adjustment for linear category
                    try:
                        logger.info(
                            "order_creation_enabling_price_adjustment",
                            signal_id=str(signal_id),
                            asset=asset,
                            category="linear",
                            trace_id=trace_id,
                        )
                        adjust_response = await bybit_client.post(
                            "/v5/account/set-limit-px-action",
                            json_data={"category": "linear", "modifyEnable": True},
                            authenticated=True,
                        )
                        logger.info(
                            "order_creation_price_adjustment_enabled",
                            signal_id=str(signal_id),
                            asset=asset,
                            response=adjust_response,
                            trace_id=trace_id,
                        )
                        # Retry order creation after enabling price adjustment
                        logger.info(
                            "order_creation_retrying_after_price_adjustment",
                            signal_id=str(signal_id),
                            asset=asset,
                            trace_id=trace_id,
                        )
                        response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)
                        ret_code = response.get("retCode", 0)
                        ret_msg = response.get("retMsg", "")
                        if ret_code == 0:
                            # Success after enabling price adjustment - continue with normal flow
                            result = response.get("result", {})
                            bybit_order_id = result.get("orderId")
                            if bybit_order_id:
                                logger.info(
                                    "order_creation_success_after_price_adjustment",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    bybit_order_id=bybit_order_id,
                                    trace_id=trace_id,
                                )
                                # Continue with order creation flow
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
                    except Exception as e:
                        logger.warning(
                            "order_creation_price_adjustment_failed",
                            signal_id=str(signal_id),
                            asset=asset,
                            error=str(e),
                            trace_id=trace_id,
                        )
                    
                    # If price adjustment didn't help, try converting Market order to Limit order
                    # This is a workaround for testnet limitations with Market orders
                    if order_type == "Market":
                        logger.info(
                            "order_creation_fallback_to_limit",
                            signal_id=str(signal_id),
                            asset=asset,
                            reason="Market order failed with 30208, trying Limit order as fallback",
                            trace_id=trace_id,
                        )
                        # Get current market price for Limit order
                        try:
                            ticker_response = await bybit_client.get(
                                "/v5/market/tickers",
                                params={"category": "linear", "symbol": asset},
                                authenticated=False,
                            )
                            ticker_data = ticker_response.get("result", {}).get("list", [])
                            if ticker_data:
                                current_price = Decimal(str(ticker_data[0].get("lastPrice", signal.market_data_snapshot.price)))
                                # Use current price for Limit order
                                limit_price = current_price
                                logger.info(
                                    "order_creation_limit_price_from_ticker",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    limit_price=float(limit_price),
                                    trace_id=trace_id,
                                )
                                
                                # Prepare Limit order parameters
                                limit_params = self._prepare_bybit_order_params(
                                    signal=signal,
                                    order_type="Limit",
                                    quantity=quantity,
                                    price=limit_price,
                                )
                                
                                # Try creating Limit order
                                limit_response = await bybit_client.post(endpoint, json_data=limit_params, authenticated=True)
                                limit_ret_code = limit_response.get("retCode", 0)
                                limit_ret_msg = limit_response.get("retMsg", "")
                                
                                if limit_ret_code == 0:
                                    limit_result = limit_response.get("result", {})
                                    limit_order_id = limit_result.get("orderId")
                                    if limit_order_id:
                                        logger.info(
                                            "order_creation_limit_fallback_success",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            bybit_order_id=limit_order_id,
                                            trace_id=trace_id,
                                        )
                                        # Save Limit order to database
                                        order = await self._save_order_to_database(
                                            signal=signal,
                                            bybit_order_id=limit_order_id,
                                            order_type="Limit",
                                            quantity=quantity,
                                            price=limit_price,
                                            trace_id=trace_id,
                                        )
                                        logger.info(
                                            "order_creation_complete",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            order_id=str(order.id),
                                            bybit_order_id=limit_order_id,
                                            trace_id=trace_id,
                                        )
                                        return order
                        except Exception as e:
                            logger.warning(
                                "order_creation_limit_fallback_failed",
                                signal_id=str(signal_id),
                                asset=asset,
                                error=str(e),
                                trace_id=trace_id,
                            )
                
                error_msg = f"Bybit API error: {ret_msg} (code: {ret_code})"
                logger.error(
                    "order_creation_api_error",
                    signal_id=str(signal_id),
                    asset=asset,
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)
            
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

        # Get order from database to capture before state
        order = await self._get_order_by_bybit_id(order_id, trace_id)
        before_state = None
        if order:
            before_state = {
                "status": order.status,
                "filled_quantity": str(order.filled_quantity),
                "average_price": str(order.average_price) if order.average_price else None,
            }

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
            
            # Get updated order and publish event
            if order:
                updated_order = await self._get_order_by_bybit_id(order_id, trace_id)
                if updated_order:
                    after_state = {
                        "status": updated_order.status,
                        "filled_quantity": str(updated_order.filled_quantity),
                        "average_price": str(updated_order.average_price) if updated_order.average_price else None,
                    }
                    await self.event_publisher.publish_order_event(
                        order=updated_order,
                        event_type="cancelled",
                        trace_id=trace_id,
                        before_state=before_state,
                        after_state=after_state,
                    )
            
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

            # Get updated order and publish modification event
            if order:
                updated_order = await self._get_order_by_bybit_id(order_id, trace_id)
                if updated_order:
                    after_state = {
                        "status": updated_order.status,
                        "filled_quantity": str(updated_order.filled_quantity),
                        "average_price": str(updated_order.average_price) if updated_order.average_price else None,
                    }
                    await self.event_publisher.publish_order_event(
                        order=updated_order,
                        event_type="cancelled",
                        trace_id=trace_id,
                        before_state=before_state,
                        after_state=after_state,
                    )

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
        }

        # Add timeInForce - required for Limit orders, optional for Market orders
        if order_type == "Limit":
            params["timeInForce"] = selector.get_time_in_force(order_type)
        # For Market orders, timeInForce is optional - omit it to avoid issues

        # Add price for limit orders only
        if order_type == "Limit" and price:
            params["price"] = str(price)

        # Add post_only flag for limit orders
        if selector.should_use_post_only(order_type):
            params["postOnly"] = True

        # Note: reduce_only flag is not set here as it requires position context
        # and should be determined by the calling service based on current position state
        # This allows for more flexible position management strategies

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
            Order object with dry_run status (saved to database)
        """
        # Generate dry-run order ID
        order_id = f"DRY-RUN-{uuid4()}"

        # Save to database (this will create the order with proper UUID and return it)
        order = await self._save_order_to_database(
            signal=signal,
            bybit_order_id=order_id,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trace_id=trace_id,
            is_dry_run=True,
        )

        logger.info(
            "dry_run_order_created",
            signal_id=str(signal.signal_id),
            order_id=str(order.id),
            bybit_order_id=order_id,
            asset=signal.asset,
            quantity=float(quantity),
            trace_id=trace_id,
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

    async def _get_order_by_bybit_id(
        self, bybit_order_id: str, trace_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Get order from database by Bybit order ID.

        Args:
            bybit_order_id: Bybit order ID
            trace_id: Optional trace ID

        Returns:
            Order object if found, None otherwise
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                       status, filled_quantity, average_price, fees, created_at, updated_at,
                       executed_at, trace_id, is_dry_run
                FROM orders
                WHERE order_id = $1
            """
            row = await pool.fetchrow(query, bybit_order_id)

            if row is None:
                return None

            return Order.from_dict(dict(row))

        except Exception as e:
            logger.error(
                "order_query_by_bybit_id_failed",
                bybit_order_id=bybit_order_id,
                error=str(e),
                trace_id=trace_id,
            )
            return None

