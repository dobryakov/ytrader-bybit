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
                # Error 110007: Insufficient balance
                # Try to reduce order size and retry
                if ret_code == 110007:
                    logger.warning(
                        "order_creation_insufficient_balance_error",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        original_quantity=float(quantity),
                        order_price=float(price) if price else None,
                        trace_id=trace_id,
                    )
                    
                    # Try to reduce order size based on available balance (if enabled)
                    reduced_order = None
                    if settings.order_manager_enable_order_size_reduction:
                        try:
                            reduced_order = await self._handle_insufficient_balance(
                                signal=signal,
                                order_type=order_type,
                                original_quantity=quantity,
                                price=price,
                                ret_msg=ret_msg,
                                trace_id=trace_id,
                            )
                            
                            if reduced_order:
                                # Successfully created reduced order
                                logger.info(
                                    "order_creation_success_with_reduced_size",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    original_quantity=float(quantity),
                                    reduced_quantity=float(reduced_order.quantity),
                                    reduction_percentage=float((1 - reduced_order.quantity / quantity) * 100) if quantity > 0 else 0,
                                    order_id=str(reduced_order.id),
                                    bybit_order_id=reduced_order.order_id,
                                    trace_id=trace_id,
                                )
                                return reduced_order
                        except Exception as e:
                            logger.error(
                                "order_creation_balance_reduction_error",
                                signal_id=str(signal_id),
                                asset=asset,
                                error=str(e),
                                trace_id=trace_id,
                                exc_info=True,
                            )
                            # Continue to rejection flow
                    else:
                        logger.info(
                            "order_size_reduction_disabled",
                            signal_id=str(signal_id),
                            asset=asset,
                            reason="ORDERMANAGER_ENABLE_ORDER_SIZE_REDUCTION is disabled",
                            trace_id=trace_id,
                        )
                    
                    # If reduction failed, wasn't possible, or was disabled, reject the order
                    error_msg = f"Bybit API error: {ret_msg} (code: {ret_code}). Order size reduction attempted but failed."
                    logger.error(
                        "order_creation_api_error_after_reduction",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        trace_id=trace_id,
                    )
                    
                    # Save rejected order to database
                    rejected_order = await self._save_rejected_order(
                        signal=signal,
                        order_type=order_type,
                        quantity=quantity,
                        price=price,
                        rejection_reason=f"Bybit API error 110007: {ret_msg}. Order size reduction attempted but failed.",
                        trace_id=trace_id,
                    )
                    
                    # Publish rejection event
                    if rejected_order:
                        await self.event_publisher.publish_order_event(
                            order=rejected_order,
                            event_type="rejected",
                            trace_id=trace_id,
                            rejection_reason=f"Bybit API error 110007: {ret_msg}",
                        )
                    
                    raise OrderExecutionError(error_msg)
                
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

    def _extract_currencies_from_pair(self, trading_pair: str) -> tuple[str, str]:
        """Extract base and quote currency from trading pair symbol.
        
        Examples:
            BTCUSDT -> ('BTC', 'USDT')
            ETHUSDT -> ('ETH', 'USDT')
            BTCUSDC -> ('BTC', 'USDC')
            ADAUSDT -> ('ADA', 'USDT')
        
        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Tuple of (base_currency, quote_currency)
        """
        # Common quote currencies (ordered by length to match longer ones first)
        quote_currencies = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD', 'EUR', 'GBP', 'JPY']
        
        trading_pair_upper = trading_pair.upper()
        
        # Try to match known quote currencies
        for quote in quote_currencies:
            if trading_pair_upper.endswith(quote):
                base = trading_pair_upper[:-len(quote)]
                return (base, quote)
        
        # Fallback: assume last 4 characters are quote currency
        if len(trading_pair) >= 8:
            return (trading_pair_upper[:-4], trading_pair_upper[-4:])
        
        # Last resort: assume last 3 characters are quote currency
        if len(trading_pair) >= 6:
            return (trading_pair_upper[:-3], trading_pair_upper[-3:])
        
        # If pair is too short, return as-is
        return (trading_pair_upper, "UNKNOWN")

    def _get_required_currency_for_order(self, trading_pair: str, signal_type: str) -> str:
        """Determine which currency is required for an order.
        
        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            signal_type: Signal type ('buy' or 'sell')
            
        Returns:
            Currency symbol needed for the order (e.g., 'USDT' for buy, 'BTC' for sell)
        """
        base_currency, quote_currency = self._extract_currencies_from_pair(trading_pair)
        
        if signal_type.lower() == "buy":
            # Buy orders require quote currency (USDT to buy BTC)
            return quote_currency
        else:
            # Sell orders require base currency (BTC to sell)
            return base_currency

    async def _get_available_balance_from_db(self, coin: str, trace_id: Optional[str] = None) -> Optional[Decimal]:
        """Get latest available balance for a coin from database.
        
        Args:
            coin: Coin symbol (e.g., 'USDT', 'BTC', 'ETH')
            trace_id: Optional trace ID for logging
            
        Returns:
            Available balance as Decimal, or None if not found
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT available_balance
                FROM account_balances
                WHERE coin = $1
                ORDER BY received_at DESC
                LIMIT 1
            """
            row = await pool.fetchrow(query, coin)
            
            if row and row["available_balance"] is not None:
                balance = Decimal(str(row["available_balance"]))
                logger.debug(
                    "balance_retrieved_from_db",
                    coin=coin,
                    available_balance=float(balance),
                    trace_id=trace_id,
                )
                return balance
            
            logger.warning(
                "balance_not_found_in_db",
                coin=coin,
                trace_id=trace_id,
            )
            return None
            
        except Exception as e:
            logger.error(
                "balance_retrieval_from_db_failed",
                coin=coin,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    def _calculate_max_affordable_quantity(
        self,
        available_balance: Decimal,
        order_price: Optional[Decimal],
        signal_type: str,
        trading_pair: str,
        safety_margin: Decimal = Decimal("0.98"),  # 2% safety margin
    ) -> Decimal:
        """Calculate maximum affordable quantity based on available balance.
        
        Args:
            available_balance: Available balance in the required currency
            order_price: Order price (needed for buy orders to calculate quantity)
            signal_type: Signal type ('buy' or 'sell')
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            safety_margin: Safety margin to leave (default: 0.98 = 2% buffer)
            
        Returns:
            Maximum affordable quantity in base currency
        """
        base_currency, quote_currency = self._extract_currencies_from_pair(trading_pair)
        
        if signal_type.lower() == "buy":
            # Buy orders: balance is in quote currency (e.g., USDT)
            # Need to calculate how much base currency we can buy
            if not order_price or order_price <= 0:
                logger.warning(
                    "order_reduction_price_required_for_buy",
                    trading_pair=trading_pair,
                    reason="Order price required to calculate quantity for buy orders",
                )
                return Decimal("0")
            
            # Calculate maximum order value with safety margin
            max_order_value = available_balance * safety_margin
            
            # Calculate maximum quantity in base currency
            max_quantity = max_order_value / order_price
            
            # Ensure quantity is positive
            if max_quantity <= 0:
                return Decimal("0")
            
            return max_quantity
        
        elif signal_type.lower() == "sell":
            # Sell orders: balance is in base currency (e.g., BTC)
            # We can sell as much as we have (with safety margin)
            max_quantity = available_balance * safety_margin
            
            # Ensure quantity is positive
            if max_quantity <= 0:
                return Decimal("0")
            
            return max_quantity
        
        else:
            logger.warning(
                "order_reduction_unknown_signal_type",
                signal_type=signal_type,
                trading_pair=trading_pair,
            )
            return Decimal("0")

    async def _handle_insufficient_balance(
        self,
        signal: TradingSignal,
        order_type: str,
        original_quantity: Decimal,
        price: Optional[Decimal],
        ret_msg: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Order]:
        """Handle insufficient balance error by reducing order size and retrying.
        
        Args:
            signal: Trading signal
            order_type: Order type
            original_quantity: Original order quantity that failed
            price: Order price
            ret_msg: Error message from Bybit
            trace_id: Optional trace ID
            
        Returns:
            Created Order object if successful, None if reduction failed or not possible
        """
        trace_id = trace_id or signal.trace_id
        signal_id = signal.signal_id
        asset = signal.asset
        
        # Determine which currency is required for this order
        required_currency = self._get_required_currency_for_order(asset, signal.signal_type)
        
        logger.info(
            "order_reduction_determining_currency",
            signal_id=str(signal_id),
            asset=asset,
            signal_type=signal.signal_type,
            required_currency=required_currency,
            trace_id=trace_id,
        )
        
        # Get available balance from database for the required currency
        available_balance = await self._get_available_balance_from_db(coin=required_currency, trace_id=trace_id)
        
        if available_balance is None:
            logger.warning(
                "order_reduction_skipped_no_balance_data",
                signal_id=str(signal_id),
                asset=asset,
                required_currency=required_currency,
                reason=f"Balance data for {required_currency} not available in database",
                trace_id=trace_id,
            )
            return None
        
        # Get current price if needed (for buy orders)
        order_price = price
        is_buy_order = signal.signal_type.lower() == "buy"
        
        if is_buy_order and not order_price:
            # For buy orders, we need price to calculate quantity
            if signal.market_data_snapshot and signal.market_data_snapshot.price:
                order_price = signal.market_data_snapshot.price
            else:
                # Try to get current price from Bybit API
                try:
                    bybit_client = get_bybit_client()
                    ticker_response = await bybit_client.get(
                        "/v5/market/tickers",
                        params={"category": "linear", "symbol": asset},
                        authenticated=False,
                    )
                    ticker_data = ticker_response.get("result", {}).get("list", [])
                    if ticker_data:
                        order_price = Decimal(str(ticker_data[0].get("lastPrice", "0")))
                except Exception as e:
                    logger.warning(
                        "order_reduction_price_fetch_failed",
                        signal_id=str(signal_id),
                        asset=asset,
                        error=str(e),
                        trace_id=trace_id,
                    )
                    return None
            
            if not order_price or order_price <= 0:
                logger.warning(
                    "order_reduction_skipped_no_price",
                    signal_id=str(signal_id),
                    asset=asset,
                    reason="Cannot determine order price for buy order reduction calculation",
                    trace_id=trace_id,
                )
                return None
        
        # Calculate maximum affordable quantity
        max_quantity = self._calculate_max_affordable_quantity(
            available_balance=available_balance,
            order_price=order_price if signal.signal_type.lower() == "buy" else None,
            signal_type=signal.signal_type,
            trading_pair=asset,
        )
        
        # Check if reduction is possible and meaningful
        if max_quantity <= 0:
            logger.info(
                "order_reduction_impossible_zero_balance",
                signal_id=str(signal_id),
                asset=asset,
                required_currency=required_currency,
                available_balance=float(available_balance),
                trace_id=trace_id,
            )
            return None
        
        # Check if reduction makes sense (at least 10% of original)
        min_meaningful_quantity = original_quantity * Decimal("0.1")
        if max_quantity < min_meaningful_quantity:
            logger.info(
                "order_reduction_impossible_too_small",
                signal_id=str(signal_id),
                asset=asset,
                required_currency=required_currency,
                available_balance=float(available_balance),
                max_quantity=float(max_quantity),
                original_quantity=float(original_quantity),
                min_meaningful_quantity=float(min_meaningful_quantity),
                trace_id=trace_id,
            )
            return None
        
        # Use the minimum of max_quantity and original_quantity (with safety margin)
        reduced_quantity = min(max_quantity, original_quantity * Decimal("0.95"))  # Max 5% reduction
        
        # Ensure reduced quantity is at least 1% of original
        min_quantity = original_quantity * Decimal("0.01")
        if reduced_quantity < min_quantity:
            reduced_quantity = min_quantity
        
        # Round to valid precision using symbol info (tick_size and lot_size)
        # Import QuantityCalculator to get symbol info and apply proper rounding
        from ..services.quantity_calculator import QuantityCalculator
        quantity_calculator = QuantityCalculator()
        
        try:
            # Get symbol info to apply proper rounding
            symbol_info = await quantity_calculator._get_symbol_info(asset, trace_id)
            # Apply precision rounding using QuantityCalculator's method
            reduced_quantity = quantity_calculator._apply_precision(reduced_quantity, symbol_info, trace_id)
            
            # Validate minimum quantity after rounding
            min_order_qty = Decimal(str(symbol_info.get("min_order_qty", "0")))
            if reduced_quantity < min_order_qty:
                logger.warning(
                    "order_reduction_below_min_after_rounding",
                    signal_id=str(signal_id),
                    asset=asset,
                    reduced_quantity=float(reduced_quantity),
                    min_order_qty=float(min_order_qty),
                    trace_id=trace_id,
                )
                # If reduced quantity is below minimum, try using minimum quantity if it's within balance
                if min_order_qty <= max_quantity:
                    reduced_quantity = min_order_qty
                else:
                    # Cannot create order - reduced quantity below minimum and minimum exceeds balance
                    logger.info(
                        "order_reduction_impossible_below_min",
                        signal_id=str(signal_id),
                        asset=asset,
                        min_order_qty=float(min_order_qty),
                        max_quantity=float(max_quantity),
                        trace_id=trace_id,
                    )
                    return None
        except Exception as e:
            logger.warning(
                "order_reduction_symbol_info_error",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            # Fallback to simple rounding if symbol info fetch fails
            reduced_quantity = reduced_quantity.quantize(Decimal("0.000001"), rounding="ROUND_DOWN")
        
        logger.info(
            "order_reduction_calculated",
            signal_id=str(signal_id),
            asset=asset,
            required_currency=required_currency,
                original_quantity=float(original_quantity),
                reduced_quantity=float(reduced_quantity),
                available_balance=float(available_balance),
                order_price=float(order_price) if order_price else None,
                reduction_percentage=float((1 - reduced_quantity / original_quantity) * 100) if original_quantity > 0 else 0,
                trace_id=trace_id,
            )
        
        # Try to create order with reduced quantity
        try:
            bybit_client = get_bybit_client()
            endpoint = "/v5/order/create"
            
            # Prepare order parameters with reduced quantity
            bybit_params = self._prepare_bybit_order_params(
                signal=signal,
                order_type=order_type,
                quantity=reduced_quantity,
                price=price,
            )
            
            logger.info(
                "order_reduction_retry_attempt",
                signal_id=str(signal_id),
                asset=asset,
                reduced_quantity=float(reduced_quantity),
                trace_id=trace_id,
            )
            
            # Retry order creation with reduced quantity
            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)
            
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            if ret_code == 0:
                # Success!
                result = response.get("result", {})
                bybit_order_id = result.get("orderId")
                
                if bybit_order_id:
                    # Save order to database
                    order = await self._save_order_to_database(
                        signal=signal,
                        bybit_order_id=bybit_order_id,
                        order_type=order_type,
                        quantity=reduced_quantity,
                        price=price,
                        trace_id=trace_id,
                    )
                    
                    logger.info(
                        "order_reduction_success",
                        signal_id=str(signal_id),
                        asset=asset,
                        original_quantity=float(original_quantity),
                        reduced_quantity=float(reduced_quantity),
                        order_id=str(order.id),
                        bybit_order_id=bybit_order_id,
                        trace_id=trace_id,
                    )
                    
                    return order
            
            # If retry also failed with 110007, don't try again
            if ret_code == 110007:
                logger.warning(
                    "order_reduction_retry_failed_insufficient_balance",
                    signal_id=str(signal_id),
                    asset=asset,
                    reduced_quantity=float(reduced_quantity),
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    trace_id=trace_id,
                )
                return None
            
            # Other errors - log and return None
            logger.warning(
                "order_reduction_retry_failed_other_error",
                signal_id=str(signal_id),
                asset=asset,
                reduced_quantity=float(reduced_quantity),
                ret_code=ret_code,
                ret_msg=ret_msg,
                trace_id=trace_id,
            )
            return None
            
        except Exception as e:
            logger.error(
                "order_reduction_retry_exception",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def _save_rejected_order(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Optional[Decimal],
        price: Optional[Decimal],
        rejection_reason: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Order]:
        """Save rejected order to database.
        
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
            pool = await DatabaseConnection.get_pool()
            side = "Buy" if signal.signal_type.lower() == "buy" else "Sell"
            
            # Validate rejection_reason is not empty
            if not rejection_reason or not rejection_reason.strip():
                logger.warning(
                    "empty_rejection_reason",
                    signal_id=str(signal.signal_id),
                    trace_id=trace_id,
                )
                rejection_reason = "Unknown rejection reason"
            
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
            
            # Check if rejection_reason column exists (might not exist in older migrations)
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
            
            if row:
                order_data = dict(row)
                order = Order.from_dict(order_data)
                
                logger.info(
                    "rejected_order_saved",
                    signal_id=str(signal.signal_id),
                    order_id=str(order.id),
                    rejection_reason=rejection_reason,
                    trace_id=trace_id,
                )
                
                return order
            
            return None
            
        except Exception as e:
            logger.error(
                "rejected_order_save_failed",
                signal_id=str(signal.signal_id),
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

