"""Quantity calculator service for converting amount to quantity with precision."""

from decimal import Decimal, ROUND_DOWN

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

        # Step 4: Validate minimum quantity
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
            bybit_client = get_bybit_client()
            # Bybit v5 API endpoint for instrument info
            endpoint = "/v5/market/instruments-info"
            params = {"category": "linear", "symbol": asset}

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
                "tick_size": Decimal(str(lot_size_filter.get("qtyStep", "0.001"))),
                "lot_size": Decimal(str(lot_size_filter.get("minQty", "0.001"))),
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
                tick_size=float(symbol_info["tick_size"]),
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
        """Apply tick size and lot size precision to quantity.

        Args:
            quantity: Raw quantity value
            symbol_info: Symbol information with tick_size and lot_size
            trace_id: Trace ID for logging

        Returns:
            Rounded quantity
        """
        tick_size = symbol_info.get("tick_size", Decimal("0.001"))
        lot_size = symbol_info.get("lot_size", Decimal("0.001"))

        # Round down to nearest tick size
        if tick_size > 0:
            quantity = (quantity / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size

        # Round down to nearest lot size multiple
        if lot_size > 0:
            quantity = (quantity / lot_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * lot_size

        logger.debug(
            "quantity_precision_applied",
            before=float(quantity),
            tick_size=float(tick_size),
            lot_size=float(lot_size),
            after=float(quantity),
            trace_id=trace_id,
        )

        return quantity

