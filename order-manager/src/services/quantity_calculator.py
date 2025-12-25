"""Quantity calculator service for converting amount to quantity with precision."""

from decimal import Decimal, ROUND_DOWN, ROUND_UP

from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..utils.bybit_client import get_bybit_client
from ..exceptions import OrderExecutionError

logger = get_logger(__name__)


class QuantityCalculator:
    """Service for converting quote currency amount to base currency quantity with proper precision."""

    def __init__(self):
        """Initialize quantity calculator."""
        # Cache for symbol info (tick size, lot size, min quantity)
        self._symbol_info_cache: dict[str, dict] = {}

    async def calculate_quantity(self, signal: TradingSignal) -> Decimal:
        """Calculate order quantity from signal amount with proper precision.

        Uses signal snapshot price for conversion, fetches tick size/lot size from
        Bybit API, and rounds down to nearest valid quantity.

        Args:
            signal: Trading signal with amount and market data

        Returns:
            Calculated quantity in base currency

        Raises:
            OrderExecutionError: If quantity calculation fails or quantity is too small
        """
        trace_id = signal.trace_id
        amount = signal.amount
        market_data = signal.market_data_snapshot
        snapshot_price = market_data.price
        asset = signal.asset

        # Step 1: Convert amount to quantity using snapshot price
        quantity = amount / snapshot_price

        logger.debug(
            "quantity_calculation_initial",
            signal_id=str(signal.signal_id),
            amount=float(amount),
            price=float(snapshot_price),
            quantity=float(quantity),
            trace_id=trace_id,
        )

        # Step 2: Fetch symbol info (tick size, lot size, min quantity)
        symbol_info = await self._get_symbol_info(asset, trace_id)

        # Step 3: Apply precision rounding
        quantity = self._apply_precision(quantity, symbol_info, trace_id)

        # Step 4: Ensure quantity meets minOrderValue requirement
        # Bybit requires order value (quantity * price) to be >= minOrderValue
        min_order_value = symbol_info.get("min_order_value", Decimal("5"))
        order_value = quantity * snapshot_price
        
        if order_value < min_order_value:
            # Increase quantity to meet minOrderValue requirement
            min_quantity_by_value = min_order_value / snapshot_price
            # Round up to next effective step
            effective_step = symbol_info.get("lot_size", Decimal("0.001"))
            min_order_qty = symbol_info.get("min_order_qty", Decimal("0.001"))
            effective_step = min(effective_step, min_order_qty) if effective_step > 0 and min_order_qty > 0 else (effective_step or min_order_qty)
            
            if effective_step > 0:
                min_quantity_by_value = ((min_quantity_by_value / effective_step).quantize(Decimal("1"), rounding=ROUND_UP) * effective_step)
            
            quantity = max(quantity, min_quantity_by_value)
            
            logger.info(
                "quantity_adjusted_for_min_order_value",
                signal_id=str(signal.signal_id),
                original_quantity=float(quantity),
                min_order_value=float(min_order_value),
                order_value=float(order_value),
                adjusted_quantity=float(quantity),
                trace_id=trace_id,
            )

        # Step 5: Validate minimum quantity (after minOrderValue adjustment)
        min_quantity = Decimal(str(symbol_info.get("min_order_qty", "0")))
        if quantity < min_quantity:
            error_msg = f"Calculated quantity {quantity} is below minimum {min_quantity} for {asset}"
            logger.error(
                "quantity_below_minimum",
                signal_id=str(signal.signal_id),
                quantity=float(quantity),
                min_quantity=float(min_quantity),
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg)

        logger.info(
            "quantity_calculation_complete",
            signal_id=str(signal.signal_id),
            amount=float(amount),
            price=float(snapshot_price),
            final_quantity=float(quantity),
            min_quantity=float(min_quantity),
            trace_id=trace_id,
        )

        return quantity

    async def _get_symbol_info(self, asset: str, trace_id: str | None = None) -> dict:
        """Get symbol information from Bybit API or cache.

        Args:
            asset: Trading pair symbol
            trace_id: Trace ID for logging

        Returns:
            Dictionary with symbol info (tick_size, lot_size, min_order_qty)
        """
        # Check cache first
        if asset in self._symbol_info_cache:
            logger.debug("symbol_info_cache_hit", asset=asset, trace_id=trace_id)
            return self._symbol_info_cache[asset]

            # Fetch from Bybit API
            try:
                from ..config.settings import settings
                bybit_client = get_bybit_client()
                # Bybit v5 API endpoint for instrument info
                endpoint = "/v5/market/instruments-info"
                params = {"category": settings.bybit_market_category, "symbol": asset}

            response = await bybit_client.get(endpoint, params=params, authenticated=False)
            result = response.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                raise OrderExecutionError(f"Symbol {asset} not found in Bybit API")

            symbol_data = list_data[0]
            
            # Extract all required fields from instruments-info
            lot_size_filter = symbol_data.get("lotSizeFilter", {})
            price_filter = symbol_data.get("priceFilter", {})
            
            symbol_info = {
                "lot_size": Decimal(str(lot_size_filter.get("qtyStep", "0.001"))),  # qtyStep is the step size for quantity
                "min_order_qty": Decimal(str(lot_size_filter.get("minQty", "0.001"))),
                "max_order_qty": Decimal(str(lot_size_filter.get("maxQty", "999999999"))),
                "min_order_value": Decimal(str(symbol_data.get("minOrderValue", "5"))),  # Default 5 USDT
                "price_tick_size": Decimal(str(price_filter.get("tickSize", "0.01"))),
                "price_limit_ratio_x": Decimal(str(price_filter.get("priceLimitRatioX", "0.1"))),  # Default 10% deviation
                "price_limit_ratio_y": Decimal(str(price_filter.get("priceLimitRatioY", "0.1"))),
                "raw_data": symbol_data,  # Store full response for future use
            }

            # Cache the result
            self._symbol_info_cache[asset] = symbol_info

            logger.info(
                "symbol_info_fetched",
                asset=asset,
                lot_size=float(symbol_info["lot_size"]),
                min_order_qty=float(symbol_info["min_order_qty"]),
                trace_id=trace_id,
            )

            return symbol_info

        except Exception as e:
            logger.error("symbol_info_fetch_failed", asset=asset, error=str(e), trace_id=trace_id)
            # Fallback to default values if API call fails
            default_info = {
                "tick_size": Decimal("0.001"),
                "lot_size": Decimal("0.001"),
                "min_order_qty": Decimal("0.001"),
            }
            return default_info

    def _apply_precision(self, quantity: Decimal, symbol_info: dict, trace_id: str | None = None) -> Decimal:
        """Apply lot size precision to quantity.

        Note: tick_size is for price precision, not quantity. Only lot_size (qtyStep) is used for quantity.

        Args:
            quantity: Raw quantity value
            symbol_info: Symbol information with lot_size (qtyStep)
            trace_id: Trace ID for logging

        Returns:
            Rounded quantity
        """
        # Use lot_size (qtyStep) for quantity precision - this is the step size for order quantity
        lot_size = symbol_info.get("lot_size", Decimal("0.001"))
        min_order_qty = symbol_info.get("min_order_qty", Decimal("0.001"))
        
        # Use the smaller of qtyStep and minQty to avoid rounding small quantities to zero
        # If qtyStep is larger than minQty, we'll use minQty to ensure we can place small orders
        effective_step = min(lot_size, min_order_qty) if lot_size > 0 and min_order_qty > 0 else (lot_size or min_order_qty)
        
        # Round down to nearest effective step multiple
        if effective_step > 0:
            quantity = (quantity / effective_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * effective_step

        logger.debug(
            "quantity_precision_applied",
            before_rounding=float(quantity),
            lot_size=float(lot_size),
            after_rounding=float(quantity),
            trace_id=trace_id,
        )

        return quantity

