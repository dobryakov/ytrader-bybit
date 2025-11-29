"""Order validator service for validating orders against instruments-info."""

from decimal import Decimal
from typing import Optional

from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..services.quantity_calculator import QuantityCalculator
from ..utils.bybit_client import get_bybit_client
from ..exceptions import OrderExecutionError

logger = get_logger(__name__)


class OrderValidator:
    """Service for validating order parameters against instruments-info."""
    
    async def validate_order_against_instruments_info(
        self,
        signal: TradingSignal,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
        trace_id: Optional[str] = None,
    ) -> None:
        """Validate order parameters against instruments-info before creating order.
        
        Validates:
        - minOrderValue: minimum order value in USDT
        - maxOrderQty: maximum order quantity
        - priceLimitRatio: for market orders, ensures price is within allowed deviation
        
        Args:
            signal: Trading signal
            order_type: Order type ('Market' or 'Limit')
            quantity: Order quantity
            price: Order price (for limit orders) or current market price (for market orders)
            trace_id: Trace ID for logging
            
        Raises:
            OrderExecutionError: If validation fails
        """
        trace_id = trace_id or signal.trace_id
        asset = signal.asset
        
        # Get symbol info (includes instruments-info data)
        calculator = QuantityCalculator()
        symbol_info = await calculator._get_symbol_info(asset, trace_id)
        
        # Validate minimum order value
        min_order_value = symbol_info.get("min_order_value", Decimal("5"))
        if price:
            order_value = quantity * price
            if order_value < min_order_value:
                error_msg = (
                    f"Order value {order_value} USDT is below minimum {min_order_value} USDT for {asset}. "
                    f"Required: {min_order_value} USDT, actual: {order_value} USDT"
                )
                logger.error(
                    "order_validation_min_value_failed",
                    signal_id=str(signal.signal_id),
                    asset=asset,
                    order_value=float(order_value),
                    min_order_value=float(min_order_value),
                    quantity=float(quantity),
                    price=float(price),
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)
        
        # Validate maximum order quantity
        max_order_qty = symbol_info.get("max_order_qty", Decimal("999999999"))
        if quantity > max_order_qty:
            error_msg = (
                f"Order quantity {quantity} exceeds maximum {max_order_qty} for {asset}. "
                f"Required: <= {max_order_qty}, actual: {quantity}"
            )
            logger.error(
                "order_validation_max_qty_failed",
                signal_id=str(signal.signal_id),
                asset=asset,
                quantity=float(quantity),
                max_order_qty=float(max_order_qty),
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg)
        
        # For market orders, validate price deviation using priceLimitRatio
        if order_type == "Market" and price:
            # Get current market price from signal snapshot or fetch from API
            current_price = signal.market_data_snapshot.price if signal.market_data_snapshot else None
            if not current_price:
                # Try to get from Bybit API
                try:
                    bybit_client = get_bybit_client()
                    ticker_response = await bybit_client.get(
                        "/v5/market/tickers",
                        params={"category": "linear", "symbol": asset},
                        authenticated=False,
                    )
                    ticker_data = ticker_response.get("result", {}).get("list", [])
                    if ticker_data:
                        current_price = Decimal(str(ticker_data[0].get("lastPrice", "0")))
                except Exception as e:
                    logger.warning(
                        "order_validation_price_fetch_failed",
                        signal_id=str(signal.signal_id),
                        asset=asset,
                        error=str(e),
                        trace_id=trace_id,
                    )
                    # Skip price validation if we can't get current price
                    current_price = None
            
            if current_price and current_price > 0:
                price_limit_ratio = symbol_info.get("price_limit_ratio_x", Decimal("0.1"))
                max_deviation = current_price * price_limit_ratio
                price_deviation = abs(price - current_price)
                
                if price_deviation > max_deviation:
                    error_msg = (
                        f"Market order price {price} deviates {price_deviation} from current price {current_price}, "
                        f"exceeding maximum allowed deviation {max_deviation} (ratio: {price_limit_ratio}) for {asset}"
                    )
                    logger.error(
                        "order_validation_price_deviation_failed",
                        signal_id=str(signal.signal_id),
                        asset=asset,
                        order_price=float(price),
                        current_price=float(current_price),
                        deviation=float(price_deviation),
                        max_deviation=float(max_deviation),
                        price_limit_ratio=float(price_limit_ratio),
                        trace_id=trace_id,
                    )
                    raise OrderExecutionError(error_msg)
        
        logger.debug(
            "order_validation_passed",
            signal_id=str(signal.signal_id),
            asset=asset,
            order_type=order_type,
            quantity=float(quantity),
            price=float(price) if price else None,
            trace_id=trace_id,
        )

