"""Order executor service for creating, modifying, and cancelling orders on Bybit."""

from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime

from decimal import ROUND_DOWN, ROUND_UP

from ..config.settings import settings
from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..models.order import Order
from ..publishers.order_event_publisher import OrderEventPublisher
from ..services.position_manager import PositionManager
from ..services.instrument_info_manager import InstrumentInfoManager
from ..utils.bybit_client import get_bybit_client
from ..exceptions import OrderExecutionError, BybitAPIError

logger = get_logger(__name__)


class OrderExecutor:
    """Service for executing orders on Bybit exchange via REST API."""

    def __init__(self):
        """Initialize order executor."""
        self.event_publisher = OrderEventPublisher()
        self.position_manager = PositionManager()
        self.instrument_info_manager = InstrumentInfoManager()

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
        side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"

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
            # Validate order parameters against instruments-info before creating order
            from ..services.order_validator import OrderValidator
            validator = OrderValidator()
            await validator.validate_order_against_instruments_info(
                signal=signal,
                order_type=order_type,
                quantity=quantity,
                price=price,
                trace_id=trace_id,
            )
            
            # Prepare order parameters for Bybit API
            bybit_params = await self._prepare_bybit_order_params(
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
                                limit_params = await self._prepare_bybit_order_params(
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
                
                # Error 10001: "The number of contracts exceeds minimum limit allowed"
                # This happens when quantity is below Bybit's minimum order size (minOrderValue)
                # Solution: increase quantity to meet minOrderValue and retry
                if ret_code == 10001:
                    logger.warning(
                        "order_creation_min_quantity_exceeded",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        quantity=float(quantity),
                        price=float(price) if price else None,
                        reason="Quantity below minimum limit, will try to increase quantity to meet minOrderValue",
                        trace_id=trace_id,
                    )
                    # Get symbol info to calculate minimum quantity based on minOrderValue
                    try:
                        from ..services.quantity_calculator import QuantityCalculator
                        calculator = QuantityCalculator()
                        symbol_info = await calculator._get_symbol_info(asset, trace_id)
                        min_order_value = symbol_info.get("min_order_value", Decimal("5"))
                        
                        # Get current price if not provided
                        # Try multiple sources: signal snapshot -> ticker API
                        if not price or price == 0:
                            logger.info(
                                "order_creation_fetching_price_for_10001_retry",
                                signal_id=str(signal_id),
                                asset=asset,
                                price_from_signal=float(signal.market_data_snapshot.price) if signal.market_data_snapshot and signal.market_data_snapshot.price else None,
                                trace_id=trace_id,
                            )
                            # First, try signal's market data snapshot
                            if signal.market_data_snapshot and signal.market_data_snapshot.price:
                                price = signal.market_data_snapshot.price
                                logger.info(
                                    "order_creation_using_signal_price_for_10001",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    price=float(price),
                                    trace_id=trace_id,
                                )
                            else:
                                # Fallback: fetch from Bybit ticker API
                                logger.info(
                                    "order_creation_fetching_ticker_price_for_10001",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    trace_id=trace_id,
                                )
                                ticker_response = await bybit_client.get(
                                    "/v5/market/tickers",
                                    params={"category": "linear", "symbol": asset},
                                    authenticated=False,
                                )
                                ticker_data = ticker_response.get("result", {}).get("list", [])
                                if ticker_data and len(ticker_data) > 0:
                                    last_price_str = ticker_data[0].get("lastPrice")
                                    if last_price_str:
                                        price = Decimal(str(last_price_str))
                                        logger.info(
                                            "order_creation_fetched_ticker_price_for_10001",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            price=float(price),
                                            trace_id=trace_id,
                                        )
                                    else:
                                        logger.warning(
                                            "order_creation_no_last_price_in_ticker",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            ticker_data=ticker_data[0],
                                            trace_id=trace_id,
                                        )
                                else:
                                    logger.warning(
                                        "order_creation_no_ticker_data",
                                        signal_id=str(signal_id),
                                        asset=asset,
                                        ticker_response=ticker_response,
                                        trace_id=trace_id,
                                    )
                        
                        if price and price > 0:
                            # Calculate minimum quantity based on minOrderValue
                            min_quantity_by_value = min_order_value / price
                            
                            # Apply precision rounding
                            lot_size = symbol_info.get("lot_size", Decimal("0.001"))
                            min_order_qty = symbol_info.get("min_order_qty", Decimal("0.001"))
                            effective_step = min(lot_size, min_order_qty) if lot_size > 0 and min_order_qty > 0 else (lot_size or min_order_qty)
                            
                            if effective_step > 0:
                                from decimal import ROUND_UP
                                min_quantity_by_value = ((min_quantity_by_value / effective_step).quantize(Decimal("1"), rounding=ROUND_UP) * effective_step)
                            
                            # Use the larger of current quantity and minimum by value
                            # If they're equal but order was still rejected, increase by one step
                            if min_quantity_by_value > quantity:
                                logger.info(
                                    "order_creation_increasing_quantity_for_min_order_value",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    original_quantity=float(quantity),
                                    min_order_value=float(min_order_value),
                                    min_quantity_by_value=float(min_quantity_by_value),
                                    new_quantity=float(min_quantity_by_value),
                                    trace_id=trace_id,
                                )
                                quantity = min_quantity_by_value
                            elif min_quantity_by_value == quantity and effective_step > 0:
                                # If min_quantity_by_value equals current quantity but order was rejected,
                                # increase by one step to ensure we exceed the minimum
                                new_quantity = quantity + effective_step
                                
                                # Round to lot_size to ensure it's valid for Bybit
                                if lot_size > 0:
                                    from decimal import ROUND_UP
                                    new_quantity = ((new_quantity / lot_size).quantize(Decimal("1"), rounding=ROUND_UP) * lot_size)
                                
                                logger.info(
                                    "order_creation_increasing_quantity_by_one_step",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    original_quantity=float(quantity),
                                    min_quantity_by_value=float(min_quantity_by_value),
                                    effective_step=float(effective_step),
                                    lot_size=float(lot_size),
                                    new_quantity=float(new_quantity),
                                    trace_id=trace_id,
                                )
                                quantity = new_quantity
                                
                                # Update bybit_params with new quantity
                                bybit_params["qty"] = str(quantity)
                                
                                # Retry order creation with increased quantity
                                logger.info(
                                    "order_creation_retrying_with_increased_quantity",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    new_quantity=float(quantity),
                                    trace_id=trace_id,
                                )
                                response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)
                                ret_code = response.get("retCode", 0)
                                ret_msg = response.get("retMsg", "")
                                if ret_code == 0:
                                    # Success after increasing quantity
                                    result = response.get("result", {})
                                    bybit_order_id = result.get("orderId")
                                    if bybit_order_id:
                                        logger.info(
                                            "order_creation_success_after_increasing_quantity",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            bybit_order_id=bybit_order_id,
                                            final_quantity=float(quantity),
                                            trace_id=trace_id,
                                        )
                                        # Save order to database
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
                                else:
                                    # Retry with increased quantity failed
                                    logger.warning(
                                        "order_creation_retry_with_increased_quantity_failed",
                                        signal_id=str(signal_id),
                                        asset=asset,
                                        new_quantity=float(quantity),
                                        ret_code=ret_code,
                                        ret_msg=ret_msg,
                                        trace_id=trace_id,
                                    )
                            else:
                                # min_quantity_by_value <= quantity, but order still rejected
                                # This might be due to quantity not being rounded to lot_size/qtyStep correctly
                                # Try rounding quantity to lot_size and retry
                                if lot_size > 0:
                                    from decimal import ROUND_UP
                                    # Round quantity up to next lot_size step
                                    rounded_quantity = ((quantity / lot_size).quantize(Decimal("1"), rounding=ROUND_UP) * lot_size)
                                    
                                    if rounded_quantity != quantity:
                                        logger.info(
                                            "order_creation_rounding_quantity_to_lot_size",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            original_quantity=float(quantity),
                                            lot_size=float(lot_size),
                                            rounded_quantity=float(rounded_quantity),
                                            reason="Quantity not aligned to lot_size, rounding and retrying",
                                            trace_id=trace_id,
                                        )
                                        quantity = rounded_quantity
                                        bybit_params["qty"] = str(quantity)
                                        
                                        # Retry order creation with rounded quantity
                                        logger.info(
                                            "order_creation_retrying_with_rounded_quantity",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            new_quantity=float(quantity),
                                            trace_id=trace_id,
                                        )
                                        try:
                                            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)
                                            ret_code = response.get("retCode", 0)
                                            ret_msg = response.get("retMsg", "")
                                            if ret_code == 0:
                                                # Success after rounding quantity
                                                result = response.get("result", {})
                                                bybit_order_id = result.get("orderId")
                                                if bybit_order_id:
                                                    logger.info(
                                                        "order_creation_success_after_rounding_quantity",
                                                        signal_id=str(signal_id),
                                                        asset=asset,
                                                        bybit_order_id=bybit_order_id,
                                                        final_quantity=float(quantity),
                                                        trace_id=trace_id,
                                                    )
                                                    # Save order to database
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
                                            else:
                                                logger.warning(
                                                    "order_creation_retry_with_rounded_quantity_failed",
                                                    signal_id=str(signal_id),
                                                    asset=asset,
                                                    rounded_quantity=float(quantity),
                                                    ret_code=ret_code,
                                                    ret_msg=ret_msg,
                                                    trace_id=trace_id,
                                                )
                                        except Exception as e:
                                            logger.error(
                                                "order_creation_retry_with_rounded_quantity_exception",
                                                signal_id=str(signal_id),
                                                asset=asset,
                                                error=str(e),
                                                trace_id=trace_id,
                                                exc_info=True,
                                            )
                                    else:
                                        # Quantity is already rounded to lot_size
                                        logger.warning(
                                            "order_creation_quantity_already_above_min_and_rounded",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            current_quantity=float(quantity),
                                            min_quantity_by_value=float(min_quantity_by_value),
                                            min_order_value=float(min_order_value),
                                            price=float(price),
                                            lot_size=float(lot_size),
                                            reason="Quantity is above minimum and already rounded to lot_size, but order still rejected",
                                            trace_id=trace_id,
                                        )
                                else:
                                    logger.warning(
                                        "order_creation_quantity_already_above_min",
                                        signal_id=str(signal_id),
                                        asset=asset,
                                        current_quantity=float(quantity),
                                        min_quantity_by_value=float(min_quantity_by_value),
                                        min_order_value=float(min_order_value),
                                        price=float(price),
                                        reason="Quantity above minimum but lot_size not available for rounding",
                                        trace_id=trace_id,
                                    )
                        else:
                            # Price not available, cannot calculate min quantity
                            logger.error(
                                "order_creation_cannot_get_price_for_10001_retry",
                                signal_id=str(signal_id),
                                asset=asset,
                                quantity=float(quantity),
                                price=float(price) if price else None,
                                has_signal_snapshot=signal.market_data_snapshot is not None,
                                snapshot_price=float(signal.market_data_snapshot.price) if signal.market_data_snapshot and signal.market_data_snapshot.price else None,
                                trace_id=trace_id,
                            )
                    except Exception as e:
                        logger.error(
                            "order_creation_10001_retry_failed",
                            signal_id=str(signal_id),
                            asset=asset,
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )
                    # If retry failed or didn't help, skip further error handling for 10001
                    # and continue to next error code checks
                    # (110017, etc.) - but if none match, we'll reach the general error handler
                
                # Error 110017: "current position is zero, cannot fix reduce-only order qty"
                # This happens when reduceOnly=True but position was closed between check and order creation
                # Solution: retry without reduceOnly flag
                if ret_code == 110017:
                    logger.warning(
                        "order_creation_reduce_only_position_zero",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        reason="Position became zero after reduceOnly was set, retrying without reduceOnly",
                        trace_id=trace_id,
                    )
                    # Remove reduceOnly from params and retry
                    if "reduceOnly" in bybit_params:
                        del bybit_params["reduceOnly"]
                        logger.info(
                            "order_creation_retrying_without_reduce_only",
                            signal_id=str(signal_id),
                            asset=asset,
                            trace_id=trace_id,
                        )
                        try:
                            response = await bybit_client.post(endpoint, json_data=bybit_params, authenticated=True)
                            ret_code = response.get("retCode", 0)
                            ret_msg = response.get("retMsg", "")
                            if ret_code == 0:
                                # Success after removing reduceOnly
                                result = response.get("result", {})
                                bybit_order_id = result.get("orderId")
                                if bybit_order_id:
                                    logger.info(
                                        "order_creation_success_after_removing_reduce_only",
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
                            # Handle 30208 error (price limit) after retry without reduceOnly
                            elif ret_code == 30208 and order_type == "Market":
                                logger.warning(
                                    "order_creation_price_limit_error_after_110017_retry",
                                    signal_id=str(signal_id),
                                    asset=asset,
                                    ret_code=ret_code,
                                    ret_msg=ret_msg,
                                    reason="Market order failed with 30208 after removing reduceOnly, trying Limit order as fallback",
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
                                        current_price = Decimal(str(ticker_data[0].get("lastPrice", signal.market_data_snapshot.price if signal.market_data_snapshot else "0")))
                                        # Use current price for Limit order
                                        limit_price = current_price
                                        logger.info(
                                            "order_creation_limit_price_from_ticker_after_110017",
                                            signal_id=str(signal_id),
                                            asset=asset,
                                            limit_price=float(limit_price),
                                            trace_id=trace_id,
                                        )
                                        
                                        # Prepare Limit order parameters
                                        # Note: We're creating Limit order as fallback after 110017 error,
                                        # which means position might have become zero, so we should NOT use reduceOnly
                                        limit_params = await self._prepare_bybit_order_params(
                                            signal=signal,
                                            order_type="Limit",
                                            quantity=quantity,
                                            price=limit_price,
                                        )
                                        # Force remove reduceOnly for Limit fallback after 110017
                                        # Position may have closed between Market order attempt and Limit fallback
                                        if "reduceOnly" in limit_params:
                                            logger.info(
                                                "order_creation_removing_reduce_only_for_limit_fallback",
                                                signal_id=str(signal_id),
                                                asset=asset,
                                                reason="Removing reduceOnly for Limit fallback after 110017, position may be zero",
                                                trace_id=trace_id,
                                            )
                                            del limit_params["reduceOnly"]
                                        
                                        # Try creating Limit order
                                        limit_response = await bybit_client.post(endpoint, json_data=limit_params, authenticated=True)
                                        limit_ret_code = limit_response.get("retCode", 0)
                                        limit_ret_msg = limit_response.get("retMsg", "")
                                        
                                        if limit_ret_code == 0:
                                            limit_result = limit_response.get("result", {})
                                            limit_order_id = limit_result.get("orderId")
                                            if limit_order_id:
                                                logger.info(
                                                    "order_creation_limit_fallback_success_after_110017",
                                                    signal_id=str(signal_id),
                                                    asset=asset,
                                                    bybit_order_id=limit_order_id,
                                                    limit_price=float(limit_price),
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
                                        else:
                                            logger.warning(
                                                "order_creation_limit_fallback_failed_after_110017",
                                                signal_id=str(signal_id),
                                                asset=asset,
                                                ret_code=limit_ret_code,
                                                ret_msg=limit_ret_msg,
                                                trace_id=trace_id,
                                            )
                                except Exception as limit_fallback_error:
                                    logger.error(
                                        "order_creation_limit_fallback_exception_after_110017",
                                        signal_id=str(signal_id),
                                        asset=asset,
                                        error=str(limit_fallback_error),
                                        trace_id=trace_id,
                                        exc_info=True,
                                    )
                        except Exception as e:
                            logger.error(
                                "order_creation_retry_without_reduce_only_failed",
                                signal_id=str(signal_id),
                                asset=asset,
                                error=str(e),
                                trace_id=trace_id,
                                exc_info=True,
                            )
                    
                    # If retry failed or reduceOnly was not in params, continue to error handling
                    error_msg = f"Bybit API error: {ret_msg} (code: {ret_code})"
                    logger.error(
                        "order_creation_api_error_after_110017_retry",
                        signal_id=str(signal_id),
                        asset=asset,
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        trace_id=trace_id,
                    )
                    raise OrderExecutionError(error_msg)
                
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

    def _format_decimal_to_string(self, value: Decimal, decimal_places: int) -> str:
        """Format Decimal to string with exact number of decimal places.
        
        This method formats a Decimal value to a string with exactly the specified
        number of decimal places, without converting to float to avoid precision errors.
        
        Args:
            value: Decimal value to format
            decimal_places: Exact number of decimal places required
            
        Returns:
            Formatted string with exact decimal places (e.g., "3026.18" for 2 decimal places)
        """
        # Handle zero
        if value == 0:
            if decimal_places > 0:
                return f"0.{'0' * decimal_places}"
            return "0"
        
        # Get sign and absolute value
        is_negative = value < 0
        abs_value = abs(value)
        
        # Use Decimal's tuple representation to build string directly
        # This avoids any float conversion precision issues
        sign, digits, exponent = abs_value.as_tuple()
        
        # Convert digits tuple to list of strings
        digits_list = [str(d) for d in digits]
        
        # Build the number parts based on exponent
        if exponent >= 0:
            # Integer part only (or with trailing zeros)
            integer_part = ''.join(digits_list + ['0'] * exponent) if digits_list else '0'
            if decimal_places > 0:
                return f"{'-' if is_negative else ''}{integer_part}.{'0' * decimal_places}"
            return f"{'-' if is_negative else ''}{integer_part}"
        else:
            # Has decimal part
            abs_exponent = abs(exponent)
            
            # Determine integer and decimal parts
            if abs_exponent <= len(digits_list):
                # Decimal point is within the digits
                integer_part = ''.join(digits_list[:-abs_exponent]) if len(digits_list) > abs_exponent else '0'
                decimal_part_raw = ''.join(digits_list[-abs_exponent:])
            else:
                # Decimal point is before all digits (e.g., 0.0123)
                integer_part = '0'
                decimal_part_raw = '0' * (abs_exponent - len(digits_list)) + ''.join(digits_list)
            
            # Ensure exactly the required decimal places
            if len(decimal_part_raw) > decimal_places:
                decimal_part = decimal_part_raw[:decimal_places]
            elif len(decimal_part_raw) < decimal_places:
                decimal_part = decimal_part_raw.ljust(decimal_places, '0')
            else:
                decimal_part = decimal_part_raw
            
            return f"{'-' if is_negative else ''}{integer_part}.{decimal_part}"

    async def _prepare_bybit_order_params(
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
        # Side for Bybit API: "Buy" or "Sell" (not "SELL")
        side_api = "Buy" if signal.signal_type.lower() == "buy" else "Sell"
        # Side for database: "Buy" or "SELL" (uppercase for SELL per constraint)
        side_db = "Buy" if signal.signal_type.lower() == "buy" else "SELL"

        params = {
            "category": "linear",
            "symbol": asset,
            "side": side_api,
            "orderType": order_type,
            "qty": str(quantity),
        }

        # Add timeInForce - required for Limit orders, optional for Market orders
        if order_type == "Limit":
            params["timeInForce"] = selector.get_time_in_force(order_type)
        # For Market orders, timeInForce is optional - omit it to avoid issues

        # Add price for limit orders only
        if order_type == "Limit" and price:
            # Get tick size first
            try:
                instrument_info = await self.instrument_info_manager.get_instrument(asset)
                if instrument_info and instrument_info.price_tick_size > 0:
                    tick_size = instrument_info.price_tick_size
                else:
                    # Use default tick size based on asset (BTCUSDT and ETHUSDT use 0.01)
                    if "BTC" in asset.upper() or "ETH" in asset.upper():
                        tick_size = Decimal("0.01")
                    else:
                        tick_size = Decimal("0.01")
            except Exception:
                tick_size = Decimal("0.01")
            
            # Round price to tick size using quantize
            # This ensures price matches Bybit's tick size requirements
            if side_api.lower() == "buy":
                # For buy orders, round down to get better price
                quantized_price = price.quantize(tick_size, rounding=ROUND_DOWN)
            else:
                # For sell orders, round up to get better price
                quantized_price = price.quantize(tick_size, rounding=ROUND_UP)
            
            # Determine decimal places from tick size
            tick_str = str(tick_size).rstrip('0').rstrip('.')
            if '.' in tick_str:
                decimal_places = len(tick_str.split('.')[1])
            else:
                decimal_places = 0
            
            # Format Decimal directly as string with exact decimal places
            # Avoid float conversion to prevent precision errors
            # Use Decimal's quantize to ensure exact precision matching tick size
            precision = Decimal('0.1') ** decimal_places
            final_price = quantized_price.quantize(precision)
            
            # Format Decimal to string with exact decimal places
            # Use helper method to ensure precise formatting
            price_str = self._format_decimal_to_string(final_price, decimal_places)
            
            # Validate price against price_limit_ratio before setting
            validated_price_str = await self._validate_and_adjust_price_for_limit_ratio(
                asset=asset,
                price_str=price_str,
                price_decimal=final_price,
                side=side_api,
                trace_id=signal.trace_id,
            )
            params["price"] = validated_price_str
            
            logger.info(
                "price_prepared_for_bybit",
                asset=asset,
                original_price=float(price),
                quantized_price=float(quantized_price),
                tick_size=float(tick_size),
                decimal_places=decimal_places,
                price_string=params["price"],
                validated_price_string=validated_price_str,
                side=side_api,
            )

        # Add post_only flag for limit orders
        if selector.should_use_post_only(order_type):
            params["postOnly"] = True

        # Set reduce_only flag based on current position
        # IMPORTANT: reduce_only should only be set when order would reduce existing position
        # - Sell order + Long position (positive size) = close/reduce long
        # - Buy order + Short position (negative size) = close/reduce short
        try:
            position = await self.position_manager.get_position(asset)
            if position and abs(position.size) > Decimal("0.00000001"):  # Position exists and is significant
                is_reducing_long = side_api == "Sell" and position.size > 0
                is_reducing_short = side_api == "Buy" and position.size < 0
                
                if is_reducing_long or is_reducing_short:
                    params["reduceOnly"] = True
                    logger.info(
                        "reduce_only_set",
                        asset=asset,
                        side=side_api,
                        order_type=order_type,
                        position_size=float(position.size),
                        reason=f"{'Reducing long' if is_reducing_long else 'Reducing short'} position, setting reduce_only",
                    )
                else:
                    logger.debug(
                        "reduce_only_not_set",
                        asset=asset,
                        side=side_api,
                        position_size=float(position.size),
                        reason="Order side does not reduce position, not setting reduce_only",
                    )
            else:
                logger.debug(
                    "reduce_only_not_set_no_position",
                    asset=asset,
                    side=side_api,
                    position_size=float(position.size) if position else 0.0,
                    reason="No significant position exists, not setting reduce_only",
                )
        except Exception as e:
            logger.warning(
                "reduce_only_check_failed",
                asset=asset,
                side=side_api,
                error=str(e),
                reason="Failed to check position for reduce_only, continuing without it",
            )
            # Continue without reduce_only if position check fails

        # Add TP/SL orders if enabled
        if settings.order_manager_tp_sl_enabled:
            # For Market orders, get current market price to ensure TP/SL are calculated correctly
            # For Limit orders, use the limit price
            if order_type == "Market" and price is None:
                # Get current market price for accurate TP/SL calculation
                entry_price_decimal = await selector._get_current_market_price(
                    asset=asset,
                    fallback_price=signal.market_data_snapshot.price,
                    trace_id=signal.trace_id,
                )
                logger.debug(
                    "tp_sl_using_current_market_price",
                    asset=asset,
                    current_price=float(entry_price_decimal),
                    snapshot_price=float(signal.market_data_snapshot.price),
                    trace_id=signal.trace_id,
                )
            else:
                # For Limit orders, use limit price; fallback to snapshot price if not available
                entry_price_decimal = price if price else signal.market_data_snapshot.price
            
            tp_price, sl_price = await self._calculate_tp_sl(
                signal=signal,
                entry_price=entry_price_decimal,
                trace_id=signal.trace_id,
            )
            
            # Validate TP/SL prices before adding to params
            # For buy orders: TP > entry_price, SL < entry_price
            # For sell orders: TP < entry_price, SL > entry_price
            if tp_price and settings.order_manager_tp_enabled:
                if side_api.lower() == "buy":
                    if tp_price <= entry_price_decimal:
                        logger.warning(
                            "tp_price_invalid_for_buy_order",
                            asset=asset,
                            tp_price=float(tp_price),
                            entry_price=float(entry_price_decimal),
                            reason="TP must be higher than entry price for buy orders",
                            trace_id=signal.trace_id,
                        )
                        tp_price = None
                else:  # sell
                    if tp_price >= entry_price_decimal:
                        logger.warning(
                            "tp_price_invalid_for_sell_order",
                            asset=asset,
                            tp_price=float(tp_price),
                            entry_price=float(entry_price_decimal),
                            reason="TP must be lower than entry price for sell orders",
                            trace_id=signal.trace_id,
                        )
                        tp_price = None
                
                if tp_price:
                    params["takeProfit"] = str(tp_price)
                    params["tpTriggerBy"] = settings.order_manager_tp_sl_trigger_by
                    logger.info(
                        "take_profit_added_to_order",
                        asset=asset,
                        tp_price=float(tp_price),
                        entry_price=float(entry_price_decimal),
                        trigger_by=settings.order_manager_tp_sl_trigger_by,
                        trace_id=signal.trace_id,
                    )
            
            if sl_price and settings.order_manager_sl_enabled:
                if side_api.lower() == "buy":
                    if sl_price >= entry_price_decimal:
                        logger.warning(
                            "sl_price_invalid_for_buy_order",
                            asset=asset,
                            sl_price=float(sl_price),
                            entry_price=float(entry_price_decimal),
                            reason="SL must be lower than entry price for buy orders",
                            trace_id=signal.trace_id,
                        )
                        sl_price = None
                else:  # sell
                    if sl_price <= entry_price_decimal:
                        logger.warning(
                            "sl_price_invalid_for_sell_order",
                            asset=asset,
                            sl_price=float(sl_price),
                            entry_price=float(entry_price_decimal),
                            reason="SL must be higher than entry price for sell orders",
                            trace_id=signal.trace_id,
                        )
                        sl_price = None
                
                if sl_price:
                    params["stopLoss"] = str(sl_price)
                    params["slTriggerBy"] = settings.order_manager_tp_sl_trigger_by
                    logger.info(
                        "stop_loss_added_to_order",
                        asset=asset,
                        sl_price=float(sl_price),
                        entry_price=float(entry_price_decimal),
                        trigger_by=settings.order_manager_tp_sl_trigger_by,
                        trace_id=signal.trace_id,
                    )

        return params

    async def _calculate_tp_sl(
        self,
        signal: TradingSignal,
        entry_price: Decimal,
        trace_id: Optional[str] = None,
    ) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate TP/SL prices using hybrid approach (metadata priority, then settings).

        Args:
            signal: Trading signal
            entry_price: Entry price for the order
            trace_id: Optional trace ID for logging

        Returns:
            Tuple of (tp_price, sl_price) or (None, None) if disabled
        """
        priority = settings.order_manager_tp_sl_priority.lower()
        
        # Try metadata first if priority is 'metadata' or 'both'
        if priority in ("metadata", "both"):
            if signal.metadata:
                tp_price_meta = signal.metadata.get("take_profit_price")
                sl_price_meta = signal.metadata.get("stop_loss_price")
                
                # If at least one value is in metadata, use metadata (and calculate missing from settings)
                if tp_price_meta is not None or sl_price_meta is not None:
                    tp_price = Decimal(str(tp_price_meta)) if tp_price_meta is not None else None
                    sl_price = Decimal(str(sl_price_meta)) if sl_price_meta is not None else None
                    
                    # If one is missing, calculate it from settings
                    if tp_price is None or sl_price is None:
                        tp_from_settings, sl_from_settings = await self._calculate_tp_sl_from_settings(
                            signal, entry_price, trace_id
                        )
                        if tp_price is None:
                            tp_price = tp_from_settings
                        if sl_price is None:
                            sl_price = sl_from_settings
                    
                    logger.info(
                        "tp_sl_calculated_from_metadata",
                        asset=signal.asset,
                        tp_price=float(tp_price) if tp_price else None,
                        sl_price=float(sl_price) if sl_price else None,
                        entry_price=float(entry_price),
                        trace_id=trace_id,
                    )
                    return tp_price, sl_price
            # If priority is 'metadata' but metadata is None or doesn't contain TP/SL, fallback to settings
            if priority == "metadata":
                logger.debug(
                    "tp_sl_metadata_not_available_fallback_to_settings",
                    asset=signal.asset,
                    metadata_exists=signal.metadata is not None,
                    trace_id=trace_id,
                )
                return await self._calculate_tp_sl_from_settings(signal, entry_price, trace_id)
        
        # Use settings if priority is 'settings' or 'both' with no metadata values
        if priority == "settings" or (priority == "both" and (signal.metadata is None or 
            (signal.metadata.get("take_profit_price") is None and signal.metadata.get("stop_loss_price") is None))):
            return await self._calculate_tp_sl_from_settings(signal, entry_price, trace_id)
        
        # Fallback: no TP/SL (should not reach here with valid priority values)
        logger.warning(
            "tp_sl_calculation_fallback_to_none",
            asset=signal.asset,
            priority=priority,
            trace_id=trace_id,
        )
        return None, None

    async def _calculate_tp_sl_from_settings(
        self,
        signal: TradingSignal,
        entry_price: Decimal,
        trace_id: Optional[str] = None,
    ) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculate TP/SL prices from configuration settings.

        Args:
            signal: Trading signal
            entry_price: Entry price for the order
            trace_id: Optional trace ID for logging

        Returns:
            Tuple of (tp_price, sl_price)
        """
        tp_price = None
        sl_price = None
        side = signal.signal_type.lower()
        
        # Calculate TP if enabled
        if settings.order_manager_tp_enabled:
            tp_threshold = Decimal(str(settings.order_manager_tp_threshold_pct))
            if side == "buy":
                tp_price = entry_price * (Decimal("1") + tp_threshold / Decimal("100"))
            else:  # sell
                tp_price = entry_price * (Decimal("1") - tp_threshold / Decimal("100"))
            
            # Round TP price to tick size
            try:
                instrument_info = await self.instrument_info_manager.get_instrument(signal.asset)
                if instrument_info and instrument_info.price_tick_size > 0:
                    tick_size = instrument_info.price_tick_size
                else:
                    tick_size = Decimal("0.01")
                
                # Round TP: for buy orders, round up (better price for seller); for sell orders, round down
                if side == "buy":
                    tp_price = tp_price.quantize(tick_size, rounding=ROUND_UP)
                else:
                    tp_price = tp_price.quantize(tick_size, rounding=ROUND_DOWN)
            except Exception as e:
                logger.warning(
                    "tp_price_rounding_failed",
                    asset=signal.asset,
                    tp_price=float(tp_price),
                    error=str(e),
                    trace_id=trace_id,
                )
        
        # Calculate SL if enabled
        if settings.order_manager_sl_enabled:
            sl_threshold = abs(Decimal(str(settings.order_manager_sl_threshold_pct)))
            if side == "buy":
                sl_price = entry_price * (Decimal("1") - sl_threshold / Decimal("100"))
            else:  # sell
                sl_price = entry_price * (Decimal("1") + sl_threshold / Decimal("100"))
            
            # Round SL price to tick size
            try:
                instrument_info = await self.instrument_info_manager.get_instrument(signal.asset)
                if instrument_info and instrument_info.price_tick_size > 0:
                    tick_size = instrument_info.price_tick_size
                else:
                    tick_size = Decimal("0.01")
                
                # Round SL: for buy orders, round down (better price for buyer); for sell orders, round up
                if side == "buy":
                    sl_price = sl_price.quantize(tick_size, rounding=ROUND_DOWN)
                else:
                    sl_price = sl_price.quantize(tick_size, rounding=ROUND_UP)
            except Exception as e:
                logger.warning(
                    "sl_price_rounding_failed",
                    asset=signal.asset,
                    sl_price=float(sl_price),
                    error=str(e),
                    trace_id=trace_id,
                )
        
        logger.info(
            "tp_sl_calculated_from_settings",
            asset=signal.asset,
            tp_price=float(tp_price) if tp_price else None,
            sl_price=float(sl_price) if sl_price else None,
            entry_price=float(entry_price),
            tp_threshold_pct=settings.order_manager_tp_threshold_pct if settings.order_manager_tp_enabled else None,
            sl_threshold_pct=settings.order_manager_sl_threshold_pct if settings.order_manager_sl_enabled else None,
            trace_id=trace_id,
        )
        
        return tp_price, sl_price

    async def _round_price_to_tick_size(self, asset: str, price: Decimal, side: str) -> Decimal:
        """Round price to tick size for the given asset.

        Args:
            asset: Trading pair symbol
            price: Price to round
            side: Order side ('Buy' or 'Sell')

        Returns:
            Rounded price
        """
        try:
            instrument_info = await self.instrument_info_manager.get_instrument_info(asset)
            if instrument_info and instrument_info.price_tick_size > 0:
                tick_size = instrument_info.price_tick_size
                # Round down for buy (to get better price), round up for sell
                if side.lower() == "buy":
                    rounded_price = (price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
                else:
                    rounded_price = (price / tick_size).quantize(Decimal("1"), rounding=ROUND_UP) * tick_size
                
                # Normalize to remove trailing zeros and unnecessary precision
                rounded_price = rounded_price.normalize()
                
                logger.debug(
                    "price_rounded_to_tick_size",
                    asset=asset,
                    original_price=float(price),
                    rounded_price=float(rounded_price),
                    tick_size=float(tick_size),
                    side=side,
                )
                return rounded_price
            else:
                # Fallback: use reasonable default tick sizes based on asset price
                # BTCUSDT typically has tick size 0.01, ETHUSDT has 0.01
                # For other assets, use 0.01 as default
                default_tick_size = Decimal("0.01")
                if "BTC" in asset.upper():
                    default_tick_size = Decimal("0.01")  # BTCUSDT tick size is usually 0.01
                elif "ETH" in asset.upper():
                    default_tick_size = Decimal("0.01")  # ETHUSDT tick size is usually 0.01
                
                rounded_price = (price / default_tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN if side.lower() == "buy" else ROUND_UP) * default_tick_size
                # Normalize to remove trailing zeros and unnecessary precision
                rounded_price = rounded_price.normalize()
                logger.warning(
                    "tick_size_not_found_using_default",
                    asset=asset,
                    original_price=float(price),
                    rounded_price=float(rounded_price),
                    default_tick_size=float(default_tick_size),
                    reason="Instrument info not found, using default tick size",
                )
                return rounded_price
        except Exception as e:
            logger.warning(
                "price_rounding_failed",
                asset=asset,
                price=float(price),
                error=str(e),
                reason="Failed to round price to tick size, using original price",
            )
            return price

    async def _validate_and_adjust_price_for_limit_ratio(
        self,
        asset: str,
        price_str: str,
        price_decimal: Decimal,
        side: str,
        trace_id: Optional[str] = None,
    ) -> str:
        """Validate price against price_limit_ratio and adjust if needed.
        
        This method ensures that the limit order price is within the allowed
        deviation from current market price according to Bybit's price_limit_ratio.
        
        Args:
            asset: Trading pair symbol
            price_str: Formatted price string
            price_decimal: Price as Decimal
            side: Order side ('Buy' or 'Sell')
            trace_id: Optional trace ID for logging
            
        Returns:
            Validated and potentially adjusted price string
        """
        try:
            # Get instrument info to check price_limit_ratio
            instrument_info = await self.instrument_info_manager.get_instrument(asset)
            if not instrument_info:
                logger.warning(
                    "price_limit_ratio_validation_skipped_no_instrument_info",
                    asset=asset,
                    price=price_str,
                    trace_id=trace_id,
                    reason="Instrument info not found, skipping price_limit_ratio validation",
                )
                return price_str
            
            # Get price_limit_ratio (use X for buy orders, Y for sell orders)
            # Default to 0.1 (10%) if not available
            if side.lower() == "buy":
                price_limit_ratio = instrument_info.price_limit_ratio_x or Decimal("0.1")
            else:
                price_limit_ratio = instrument_info.price_limit_ratio_y or Decimal("0.1")
            
            # Get current market price
            try:
                bybit_client = get_bybit_client()
                ticker_response = await bybit_client.get(
                    "/v5/market/tickers",
                    params={"category": "linear", "symbol": asset},
                    authenticated=False,
                )
                ticker_data = ticker_response.get("result", {}).get("list", [])
                if not ticker_data or not ticker_data[0].get("lastPrice"):
                    logger.warning(
                        "price_limit_ratio_validation_skipped_no_market_price",
                        asset=asset,
                        price=price_str,
                        trace_id=trace_id,
                        reason="Could not fetch current market price",
                    )
                    return price_str
                
                current_market_price = Decimal(str(ticker_data[0].get("lastPrice")))
            except Exception as e:
                logger.warning(
                    "price_limit_ratio_validation_skipped_api_error",
                    asset=asset,
                    price=price_str,
                    error=str(e),
                    trace_id=trace_id,
                    reason="Failed to fetch current market price from API",
                )
                return price_str
            
            # Calculate maximum allowed deviation
            max_deviation = current_market_price * price_limit_ratio
            price_deviation = abs(price_decimal - current_market_price)
            
            # Check if price is within allowed deviation
            if price_deviation <= max_deviation:
                logger.debug(
                    "price_limit_ratio_validation_passed",
                    asset=asset,
                    price=price_str,
                    current_market_price=float(current_market_price),
                    price_deviation=float(price_deviation),
                    max_deviation=float(max_deviation),
                    price_limit_ratio=float(price_limit_ratio),
                    side=side,
                    trace_id=trace_id,
                )
                return price_str
            
            # Price exceeds allowed deviation - adjust it
            logger.warning(
                "price_limit_ratio_validation_failed_adjusting",
                asset=asset,
                original_price=price_str,
                current_market_price=float(current_market_price),
                price_deviation=float(price_deviation),
                max_deviation=float(max_deviation),
                price_limit_ratio=float(price_limit_ratio),
                side=side,
                trace_id=trace_id,
                reason="Price exceeds maximum allowed deviation, adjusting to limit",
            )
            
            # Adjust price to be within allowed deviation
            # For buy orders, adjust down (closer to market price)
            # For sell orders, adjust up (closer to market price)
            if side.lower() == "buy":
                # Buy: price should be at most max_deviation below market price
                adjusted_price = current_market_price - max_deviation
                # Ensure adjusted price doesn't exceed original price (only move closer to market)
                if adjusted_price > price_decimal:
                    adjusted_price = price_decimal
            else:
                # Sell: price should be at most max_deviation above market price
                adjusted_price = current_market_price + max_deviation
                # Ensure adjusted price doesn't go below original price (only move closer to market)
                if adjusted_price < price_decimal:
                    adjusted_price = price_decimal
            
            # Round adjusted price to tick size and format
            tick_size = instrument_info.price_tick_size or Decimal("0.01")
            if side.lower() == "buy":
                adjusted_price = adjusted_price.quantize(tick_size, rounding=ROUND_DOWN)
            else:
                adjusted_price = adjusted_price.quantize(tick_size, rounding=ROUND_UP)
            
            # Format adjusted price
            tick_str = str(tick_size).rstrip('0').rstrip('.')
            if '.' in tick_str:
                decimal_places = len(tick_str.split('.')[1])
            else:
                decimal_places = 0
            
            adjusted_price_str = self._format_decimal_to_string(adjusted_price, decimal_places)
            
            logger.info(
                "price_adjusted_for_limit_ratio",
                asset=asset,
                original_price=price_str,
                adjusted_price=adjusted_price_str,
                current_market_price=float(current_market_price),
                price_limit_ratio=float(price_limit_ratio),
                side=side,
                trace_id=trace_id,
            )
            
            return adjusted_price_str
            
        except Exception as e:
            logger.error(
                "price_limit_ratio_validation_error",
                asset=asset,
                price=price_str,
                error=str(e),
                trace_id=trace_id,
                reason="Error during price_limit_ratio validation, using original price",
                exc_info=True,
            )
            # On error, return original price to avoid blocking order creation
            return price_str

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
            side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
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

    async def _get_available_margin_from_db(
        self, base_currency: str = "USDT", trace_id: Optional[str] = None
    ) -> Optional[Decimal]:
        """Get available margin from database for account-level trading.
        
        Args:
            base_currency: Base currency for margin (USDT, USD, etc.)
            trace_id: Optional trace ID for logging
            
        Returns:
            Available margin in base currency, or None if not found
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT total_available_balance, base_currency, received_at
                FROM account_margin_balances
                ORDER BY received_at DESC
                LIMIT 1
            """
            row = await pool.fetchrow(query)
            
            if row is None:
                logger.debug(
                    "margin_balance_not_found_in_db",
                    base_currency=base_currency,
                    trace_id=trace_id,
                )
                return None
            
            # Check if base currency matches (or use any if base_currency is provided)
            stored_base = row.get("base_currency", "")
            available_margin = Decimal(str(row["total_available_balance"]))
            
            logger.info(
                "margin_balance_retrieved_from_db",
                base_currency=stored_base,
                available_margin=float(available_margin),
                received_at=str(row["received_at"]),
                trace_id=trace_id,
            )
            
            return available_margin
            
        except Exception as e:
            logger.error(
                "margin_balance_retrieval_from_db_failed",
                base_currency=base_currency,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def _get_margin_from_bybit_api(
        self, base_currency: str = "USDT", trace_id: Optional[str] = None
    ) -> Optional[Decimal]:
        """Get available margin directly from Bybit API.
        
        This method is used as a fallback when database margin data is unavailable.
        
        Args:
            base_currency: Base currency for margin (USDT, USD, etc.)
            trace_id: Optional trace ID for logging
            
        Returns:
            Available margin in base currency, or None if not found or API error
        """
        try:
            bybit_client = get_bybit_client()
            endpoint = "/v5/account/wallet-balance"
            params = {"accountType": "UNIFIED"}
            
            response = await bybit_client.get(endpoint, params=params, authenticated=True)
            
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            if ret_code != 0:
                logger.warning(
                    "margin_api_fetch_failed",
                    base_currency=base_currency,
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    trace_id=trace_id,
                )
                return None
            
            result = response.get("result", {})
            list_data = result.get("list", []) if result else []
            
            if not list_data:
                logger.warning(
                    "margin_api_no_account_data",
                    base_currency=base_currency,
                    trace_id=trace_id,
                )
                return None
            
            account = list_data[0]
            account_type = account.get("accountType", "")
            
            # For unified accounts, try to get margin in the requested base currency
            if account_type == "UNIFIED":
                # First, try to get balance for the specific base currency from coin array
                coins = account.get("coin", [])
                for coin_data in coins:
                    if isinstance(coin_data, dict) and coin_data.get("coin") == base_currency:
                        # Check if this coin can be used for margin (marginCollateral = true)
                        margin_collateral = coin_data.get("marginCollateral", False)
                        if margin_collateral:
                            # Try to get available balance for this coin
                            available_value = coin_data.get("availableToWithdraw", "")
                            if not available_value or available_value == "":
                                available_value = coin_data.get("walletBalance", "0")
                            
                            if available_value and available_value != "":
                                try:
                                    margin = Decimal(str(available_value))
                                    logger.info(
                                        "margin_retrieved_from_api_coin",
                                        base_currency=base_currency,
                                        margin=float(margin),
                                        source=f"coin.{base_currency}",
                                        trace_id=trace_id,
                                    )
                                    return margin
                                except (ValueError, TypeError) as e:
                                    logger.warning(
                                        "margin_api_invalid_coin_value",
                                        base_currency=base_currency,
                                        value=available_value,
                                        error=str(e),
                                        trace_id=trace_id,
                                    )
                        break
                
                # Fallback: use totalAvailableBalance (this is usually in USDT, but may be in other currencies)
                # Note: totalAvailableBalance is the total available margin across all currencies in unified account
                total_available = account.get("totalAvailableBalance", "0")
                if total_available and total_available != "":
                    try:
                        margin = Decimal(str(total_available))
                        logger.info(
                            "margin_retrieved_from_api_total",
                            base_currency=base_currency,
                            margin=float(margin),
                            source="totalAvailableBalance",
                            note="totalAvailableBalance may be in different currency than requested",
                            trace_id=trace_id,
                        )
                        return margin
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "margin_api_invalid_value",
                            base_currency=base_currency,
                            value=total_available,
                            error=str(e),
                            trace_id=trace_id,
                        )
            
            logger.warning(
                "margin_api_not_found",
                base_currency=base_currency,
                account_type=account_type,
                trace_id=trace_id,
            )
            return None
            
        except Exception as e:
            logger.error(
                "margin_api_fetch_exception",
                base_currency=base_currency,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

    async def _get_base_currency_for_margin(self, trace_id: Optional[str] = None) -> str:
        """Determine base currency for margin from database.
        
        For unified accounts this is usually USDT, but can be another currency.
        
        Args:
            trace_id: Optional trace ID for logging
            
        Returns:
            Base currency for margin (default: "USDT")
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT base_currency, received_at
                FROM account_margin_balances
                ORDER BY received_at DESC
                LIMIT 1
            """
            row = await pool.fetchrow(query)
            
            if row and row.get("base_currency"):
                base_currency = str(row["base_currency"])
                logger.debug(
                    "base_currency_retrieved_from_db",
                    base_currency=base_currency,
                    trace_id=trace_id,
                )
                return base_currency
            
            # Fallback to USDT if not found
            logger.debug(
                "base_currency_not_found_defaulting_to_usdt",
                trace_id=trace_id,
            )
            return "USDT"
            
        except Exception as e:
            logger.warning(
                "base_currency_retrieval_failed_defaulting_to_usdt",
                error=str(e),
                trace_id=trace_id,
            )
            return "USDT"

    async def _get_balance_from_bybit_api(self, coin: str, trace_id: Optional[str] = None) -> Optional[Decimal]:
        """Get available balance for a coin directly from Bybit API.
        
        This method is used as a fallback when database balance data is unavailable.
        
        Args:
            coin: Coin symbol (e.g., 'USDT', 'BTC', 'ETH')
            trace_id: Optional trace ID for logging
            
        Returns:
            Available balance as Decimal, or None if not found or API error
        """
        try:
            bybit_client = get_bybit_client()
            endpoint = "/v5/account/wallet-balance"
            params = {"accountType": "UNIFIED"}
            
            response = await bybit_client.get(endpoint, params=params, authenticated=True)
            
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            if ret_code != 0:
                logger.warning(
                    "balance_api_fetch_failed",
                    coin=coin,
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    trace_id=trace_id,
                )
                return None
            
            result = response.get("result", {})
            list_data = result.get("list", []) if result else []
            
            if not list_data:
                logger.warning(
                    "balance_api_no_account_data",
                    coin=coin,
                    trace_id=trace_id,
                )
                return None
            
            account = list_data[0]
            account_type = account.get("accountType", "")
            
            # For unified accounts, check if coin is USDT (use totalAvailableBalance)
            if account_type == "UNIFIED" and coin == "USDT":
                total_available = account.get("totalAvailableBalance", "0")
                if total_available and total_available != "":
                    try:
                        balance = Decimal(str(total_available))
                        logger.info(
                            "balance_retrieved_from_api_total",
                            coin=coin,
                            balance=float(balance),
                            trace_id=trace_id,
                        )
                        return balance
                    except (ValueError, TypeError):
                        pass
            
            # Get coin-specific balance
            coins = account.get("coin", [])
            for coin_data in coins:
                if coin_data.get("coin") == coin:
                    # Try availableToWithdraw first, then walletBalance
                    available_value = coin_data.get("availableToWithdraw", "")
                    if not available_value or available_value == "":
                        available_value = coin_data.get("walletBalance", "0")
                    
                    if available_value and available_value != "":
                        try:
                            balance = Decimal(str(available_value))
                            logger.info(
                                "balance_retrieved_from_api",
                                coin=coin,
                                balance=float(balance),
                                source="availableToWithdraw" if coin_data.get("availableToWithdraw") else "walletBalance",
                                trace_id=trace_id,
                            )
                            return balance
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                "balance_api_invalid_value",
                                coin=coin,
                                value=available_value,
                                error=str(e),
                                trace_id=trace_id,
                            )
                    break
            
            logger.warning(
                "balance_api_coin_not_found",
                coin=coin,
                trace_id=trace_id,
            )
            return None
            
        except Exception as e:
            logger.error(
                "balance_api_fetch_exception",
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
        safety_margin: Decimal = Decimal("0.95"),  # 5% safety margin (more conservative when balance check is disabled)
        has_position: bool = False,
        position_size: Decimal = Decimal("0"),
        available_margin: Optional[Decimal] = None,
        base_currency: str = "USDT",
    ) -> Decimal:
        """Calculate maximum affordable quantity based on available balance or margin.
        
        Args:
            available_balance: Available balance in the required currency
            order_price: Order price (needed for buy orders to calculate quantity)
            signal_type: Signal type ('buy' or 'sell')
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            safety_margin: Safety margin to leave (default: 0.95 = 5% buffer)
            has_position: Whether there is a current position
            position_size: Current position size (positive = long, negative = short)
            available_margin: Available margin in base currency for trading (for sell orders without position)
            base_currency: Base currency for margin (USDT, USD, etc.)
            
        Returns:
            Maximum affordable quantity in base currency
        """
        base_currency_pair, quote_currency_pair = self._extract_currencies_from_pair(trading_pair)
        
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
            # Sell orders: check if we have a long position (can use reduce_only)
            if has_position and position_size > 0:
                # Have long position - can use reduce_only (no margin needed)
                # Return a very large number (practically unlimited)
                logger.debug(
                    "order_reduction_sell_with_long_position",
                    trading_pair=trading_pair,
                    position_size=float(position_size),
                    reason="Can use reduce_only, margin not required",
                )
                return Decimal("999999999")  # Practically unlimited
            
            # No position or short position - need margin in base currency to open/increase short position
            # Note: SELL with short position would INCREASE the short (negative) position, not reduce it
            if available_margin is None or available_margin <= 0:
                logger.warning(
                    "order_reduction_no_margin_available_for_sell",
                    trading_pair=trading_pair,
                    available_margin=float(available_margin) if available_margin else None,
                    reason="No margin available for opening short position",
                )
                return Decimal("0")
            
            if not order_price or order_price <= 0:
                logger.warning(
                    "order_reduction_price_required_for_sell",
                    trading_pair=trading_pair,
                    reason="Order price required to calculate quantity for sell orders without position",
                )
                return Decimal("0")
            
            # Calculate maximum order value with safety margin
            max_order_value = available_margin * safety_margin
            
            # Calculate maximum quantity in base currency
            max_quantity = max_order_value / order_price
            
            # Ensure quantity is positive
            if max_quantity <= 0:
                return Decimal("0")
            
            logger.debug(
                "order_reduction_sell_calculated_from_margin",
                trading_pair=trading_pair,
                available_margin=float(available_margin),
                order_price=float(order_price),
                max_quantity=float(max_quantity),
                base_currency=base_currency,
            )
            
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
        signal_type = signal.signal_type.lower()
        
        # Get current position from database
        position = await self.position_manager.get_position(asset)
        has_position = position is not None
        position_size = position.size if position else Decimal("0")
        has_long_position = has_position and position_size > 0
        has_short_position = has_position and position_size < 0
        
        logger.info(
            "order_reduction_checking_position",
            signal_id=str(signal_id),
            asset=asset,
            signal_type=signal_type,
            has_position=has_position,
            position_size=float(position_size) if position else 0,
            has_long_position=has_long_position,
            has_short_position=has_short_position,
            trace_id=trace_id,
        )
        
        # For sell orders with long position, we can use reduce_only (no margin needed)
        # For sell orders with short position, we should NOT use reduce_only (it would increase position)
        # Instead, we need margin to open a new short position
        if signal_type == "sell" and has_long_position:
            logger.info(
                "order_reduction_sell_with_long_position_reduce_only",
                signal_id=str(signal_id),
                asset=asset,
                position_size=float(position_size),
                reason="Can use reduce_only flag to close long position, margin not required",
                trace_id=trace_id,
            )
            # Will try to create order with reduce_only=True, don't reduce size
            # The order creation will be retried with reduce_only flag set
            # For now, continue with original quantity (will be handled in _prepare_bybit_order_params)
            max_quantity = Decimal("999999999")  # Practically unlimited
            available_balance = Decimal("0")  # Not used for reduce_only
        elif signal_type == "sell" and has_short_position:
            # SELL order with SHORT position: This would OPEN a new short or increase existing short
            # We don't want to use reduce_only here (it's for closing positions)
            # Need margin to open/increase short position (same as sell without position)
            logger.info(
                "order_reduction_sell_with_short_position",
                signal_id=str(signal_id),
                asset=asset,
                position_size=float(position_size),
                reason="SELL with short position would increase position, need margin, not using reduce_only",
                trace_id=trace_id,
            )
            # Use margin logic (same as sell without position)
            # Get base currency for margin and available margin from DB
            base_currency = await self._get_base_currency_for_margin(trace_id)
            available_balance = await self._get_available_margin_from_db(base_currency, trace_id)
            
            # Fallback: get margin from Bybit API if database data is unavailable
            if available_balance is None:
                logger.info(
                    "order_reduction_margin_not_in_db",
                    signal_id=str(signal_id),
                    asset=asset,
                    base_currency=base_currency,
                    reason="Margin data not available in database, fetching from Bybit API",
                    trace_id=trace_id,
                )
                available_balance = await self._get_margin_from_bybit_api(base_currency, trace_id)
                
                if available_balance is None:
                    logger.warning(
                        "order_reduction_skipped_no_margin_data",
                        signal_id=str(signal_id),
                        asset=asset,
                        base_currency=base_currency,
                        reason=f"Margin data for {base_currency} not available in database or API",
                        trace_id=trace_id,
                    )
                    return None
                
                logger.info(
                    "order_reduction_margin_fetched_from_api",
                    signal_id=str(signal_id),
                    asset=asset,
                    base_currency=base_currency,
                    available_margin=float(available_balance),
                    trace_id=trace_id,
                )
            else:
                    logger.info(
                        "order_reduction_margin_retrieved_from_db",
                        signal_id=str(signal_id),
                        asset=asset,
                        base_currency=base_currency,
                        available_margin=float(available_balance),
                        trace_id=trace_id,
                    )
            # Will calculate max_quantity from margin below (using _calculate_max_affordable_quantity)
            # Set required_currency for logging
            required_currency = base_currency
            # max_quantity will be calculated later from margin using _calculate_max_affordable_quantity
            # Skip the else block below - we already handled SELL with short position
            pass
        else:
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
            
            # For sell orders without position, need margin (not coin balance)
            if signal_type == "sell":
                # Get base currency for margin and available margin from DB
                base_currency = await self._get_base_currency_for_margin(trace_id)
                available_balance = await self._get_available_margin_from_db(base_currency, trace_id)
                
                # Fallback: get margin from Bybit API if database data is unavailable
                if available_balance is None:
                    logger.info(
                        "order_reduction_margin_not_in_db",
                        signal_id=str(signal_id),
                        asset=asset,
                        base_currency=base_currency,
                        reason="Margin data not available in database, fetching from Bybit API",
                        trace_id=trace_id,
                    )
                    available_balance = await self._get_margin_from_bybit_api(base_currency, trace_id)
                    
                    if available_balance is None:
                        logger.warning(
                            "order_reduction_skipped_no_margin_data",
                            signal_id=str(signal_id),
                            asset=asset,
                            base_currency=base_currency,
                            reason=f"Margin data for {base_currency} not available in database or API",
                            trace_id=trace_id,
                        )
                        return None
                    
                    logger.info(
                        "order_reduction_margin_fetched_from_api",
                        signal_id=str(signal_id),
                        asset=asset,
                        base_currency=base_currency,
                        available_margin=float(available_balance),
                        trace_id=trace_id,
                    )
                else:
                    logger.info(
                        "order_reduction_margin_retrieved_from_db",
                        signal_id=str(signal_id),
                        asset=asset,
                        base_currency=base_currency,
                        available_margin=float(available_balance),
                        trace_id=trace_id,
                    )
            else:
                # For buy orders, get coin balance from database
                available_balance = await self._get_available_balance_from_db(coin=required_currency, trace_id=trace_id)
                
                # Fallback: get balance from Bybit API if database data is unavailable
                if available_balance is None:
                    logger.info(
                        "order_reduction_balance_not_in_db",
                        signal_id=str(signal_id),
                        asset=asset,
                        required_currency=required_currency,
                        reason="Balance data not available in database, fetching from Bybit API",
                        trace_id=trace_id,
                    )
                    available_balance = await self._get_balance_from_bybit_api(required_currency, trace_id)
                    
                    if available_balance is None:
                        logger.warning(
                            "order_reduction_skipped_no_balance_data",
                            signal_id=str(signal_id),
                            asset=asset,
                            required_currency=required_currency,
                            reason=f"Balance data for {required_currency} not available in database or API",
                            trace_id=trace_id,
                        )
                        return None
                    
                    logger.info(
                        "order_reduction_balance_fetched_from_api",
                        signal_id=str(signal_id),
                        asset=asset,
                        required_currency=required_currency,
                        available_balance=float(available_balance),
                        trace_id=trace_id,
                    )
        
        # Get current price if needed (for Market orders or when price is None)
        order_price = price
        is_buy_order = signal_type == "buy"
        
        # For sell orders with short position (need margin calculation), we need price
        # For buy orders, we also need price for quantity calculation
        if (signal_type == "sell" and (not has_long_position)) or (is_buy_order and not order_price):
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
        # For sell orders, get base currency and margin
        base_currency_margin = "USDT"
        available_margin = None
        if signal_type == "sell" and not has_long_position:
            base_currency_margin = await self._get_base_currency_for_margin(trace_id)
            available_margin = available_balance  # Already fetched above
            available_balance = Decimal("0")  # Not used for sell without position
        
        max_quantity = self._calculate_max_affordable_quantity(
            available_balance=available_balance,
            order_price=order_price if is_buy_order else order_price,  # Need price for sell without position
            signal_type=signal.signal_type,
            trading_pair=asset,
            has_position=has_position,
            position_size=position_size,
            available_margin=available_margin,
            base_currency=base_currency_margin,
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
        
        # Get symbol info to check minimum order quantity
        from ..services.quantity_calculator import QuantityCalculator
        quantity_calculator = QuantityCalculator()
        
        try:
            symbol_info = await quantity_calculator._get_symbol_info(asset, trace_id)
            min_order_qty = Decimal(str(symbol_info.get("min_order_qty", "0")))
        except Exception as e:
            logger.warning(
                "order_reduction_min_qty_check_failed",
                signal_id=str(signal_id),
                asset=asset,
                error=str(e),
                trace_id=trace_id,
            )
            # Fallback: use 1% of original as minimum
            min_order_qty = original_quantity * Decimal("0.01")
        
        # Check if reduction is possible - need at least minimum order quantity
        if max_quantity < min_order_qty:
            logger.info(
                "order_reduction_impossible_below_minimum",
                signal_id=str(signal_id),
                asset=asset,
                required_currency=required_currency,
                available_balance=float(available_balance),
                max_quantity=float(max_quantity),
                min_order_qty=float(min_order_qty),
                original_quantity=float(original_quantity),
                trace_id=trace_id,
            )
            return None
        
        # Try iterative reduction with multiple attempts
        max_reduction_attempts = 3
        reduction_factors = [Decimal("0.7"), Decimal("0.5"), Decimal("0.3")]  # 70%, 50%, 30% of original quantity
        
        previous_reduced_quantity = None
        
        for attempt in range(max_reduction_attempts):
            # Calculate reduced quantity for this attempt
            if attempt == 0:
                # First attempt: use max_quantity or 95% of original (whichever is smaller)
                reduced_quantity = min(max_quantity, original_quantity * Decimal("0.95"))
            else:
                # Subsequent attempts: reduce from original using reduction factor
                # Then ensure it's within max_quantity limit
                reduction_factor = reduction_factors[min(attempt - 1, len(reduction_factors) - 1)]
                # Calculate reduced quantity as percentage of original
                reduced_quantity = original_quantity * reduction_factor
                # But don't exceed what we can afford (max_quantity)
                reduced_quantity = min(reduced_quantity, max_quantity)
                # Also ensure we're reducing from previous attempt (not increasing)
                if previous_reduced_quantity is not None:
                    reduced_quantity = min(reduced_quantity, previous_reduced_quantity)
            
            # Store for next iteration
            previous_reduced_quantity = reduced_quantity
            
            # Ensure reduced quantity is at least minimum order quantity
            if reduced_quantity < min_order_qty:
                # If we can't even meet minimum, we can't create order
                if attempt == 0:
                    logger.info(
                        "order_reduction_impossible_below_minimum_after_calculation",
                        signal_id=str(signal_id),
                        asset=asset,
                        calculated_quantity=float(reduced_quantity),
                        min_order_qty=float(min_order_qty),
                        trace_id=trace_id,
                    )
                    return None
                else:
                    # Try minimum quantity as last resort
                    reduced_quantity = min_order_qty
            
            # Apply proper rounding using symbol info
            try:
                # Apply precision rounding using QuantityCalculator's method
                reduced_quantity = quantity_calculator._apply_precision(reduced_quantity, symbol_info, trace_id)
                
                # Double-check minimum after rounding
                if reduced_quantity < min_order_qty:
                    # If below minimum after rounding, try minimum if it's within balance
                    if min_order_qty <= max_quantity:
                        reduced_quantity = min_order_qty
                    else:
                        logger.info(
                            "order_reduction_attempt_failed_below_min_after_rounding",
                            signal_id=str(signal_id),
                            asset=asset,
                            attempt=attempt + 1,
                            reduced_quantity_after_rounding=float(reduced_quantity),
                            min_order_qty=float(min_order_qty),
                            max_quantity=float(max_quantity),
                            trace_id=trace_id,
                        )
                        # Try next attempt if available
                        if attempt < max_reduction_attempts - 1:
                            continue
                        return None
            except Exception as e:
                logger.warning(
                    "order_reduction_precision_error",
                    signal_id=str(signal_id),
                    asset=asset,
                    attempt=attempt + 1,
                    error=str(e),
                    trace_id=trace_id,
                )
                # Fallback to simple rounding
                reduced_quantity = reduced_quantity.quantize(Decimal("0.000001"), rounding="ROUND_DOWN")
            
            logger.info(
                "order_reduction_attempt",
                signal_id=str(signal_id),
                asset=asset,
                attempt=attempt + 1,
                max_attempts=max_reduction_attempts,
                original_quantity=float(original_quantity),
                reduced_quantity=float(reduced_quantity),
                max_quantity=float(max_quantity),
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
                bybit_params = await self._prepare_bybit_order_params(
                    signal=signal,
                    order_type=order_type,
                    quantity=reduced_quantity,
                    price=price,
                )
                
                logger.info(
                    "order_reduction_retry_attempt",
                    signal_id=str(signal_id),
                    asset=asset,
                    attempt=attempt + 1,
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
                            attempt=attempt + 1,
                            original_quantity=float(original_quantity),
                            reduced_quantity=float(reduced_quantity),
                            order_id=str(order.id),
                            bybit_order_id=bybit_order_id,
                            trace_id=trace_id,
                        )
                        
                        return order
                
                # If retry also failed with 110007, try next reduction if available
                if ret_code == 110007:
                    logger.info(
                        "order_reduction_attempt_failed_insufficient_balance",
                        signal_id=str(signal_id),
                        asset=asset,
                        attempt=attempt + 1,
                        reduced_quantity=float(reduced_quantity),
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        has_more_attempts=(attempt < max_reduction_attempts - 1),
                        trace_id=trace_id,
                    )
                    # Continue to next attempt if available
                    if attempt < max_reduction_attempts - 1:
                        continue
                    # Last attempt failed, return None
                    return None
                
                # Other errors - log and return None
                logger.warning(
                    "order_reduction_retry_failed_other_error",
                    signal_id=str(signal_id),
                    asset=asset,
                    attempt=attempt + 1,
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
                    attempt=attempt + 1,
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue to next attempt if available
                if attempt < max_reduction_attempts - 1:
                    continue
                return None
        
        # All attempts exhausted
        logger.info(
            "order_reduction_all_attempts_exhausted",
            signal_id=str(signal_id),
            asset=asset,
            max_attempts=max_reduction_attempts,
            original_quantity=float(original_quantity),
            max_quantity=float(max_quantity),
            trace_id=trace_id,
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
            side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
            
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

