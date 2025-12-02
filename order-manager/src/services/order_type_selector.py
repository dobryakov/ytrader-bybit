"""Order type selector service for determining market vs limit orders."""

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..services.instrument_info_manager import InstrumentInfoManager

logger = get_logger(__name__)


class OrderTypeSelector:
    """Service for selecting order type (Market vs Limit) based on signal characteristics."""

    def __init__(self):
        """Initialize order type selector with configuration."""
        self.confidence_threshold = settings.order_manager_market_order_confidence_threshold
        self.spread_threshold = settings.order_manager_market_order_spread_threshold
        self.price_offset_ratio = settings.order_manager_limit_order_price_offset_ratio
        self.instrument_info_manager = InstrumentInfoManager()

    async def select_order_type(self, signal: TradingSignal) -> tuple[str, Decimal | None]:
        """Select order type and calculate limit price if needed.

        Decision logic:
        - Market Orders: Use when signal confidence > threshold OR spread < threshold OR explicit urgency
        - Limit Orders: Default choice for most signals to control execution price
        - Limit Price Calculation: Use signal snapshot price with offset

        Args:
            signal: Trading signal to process

        Returns:
            Tuple of (order_type, limit_price)
            - order_type: 'Market' or 'Limit'
            - limit_price: Decimal price for limit orders, None for market orders
        """
        trace_id = signal.trace_id
        market_data = signal.market_data_snapshot
        snapshot_price = market_data.price
        spread = market_data.spread or Decimal("0")

        # Convert spread to percentage if needed (assuming it's already a percentage)
        spread_pct = spread

        # Decision logic: Market order conditions
        use_market_order = False

        # Condition 1: High confidence (> threshold)
        if signal.confidence >= self.confidence_threshold:
            logger.info(
                "order_type_selected_market_confidence",
                signal_id=str(signal.signal_id),
                confidence=float(signal.confidence),
                threshold=float(self.confidence_threshold),
                trace_id=trace_id,
            )
            use_market_order = True

        # Condition 2: Low spread (< threshold)
        elif spread_pct < self.spread_threshold:
            logger.info(
                "order_type_selected_market_spread",
                signal_id=str(signal.signal_id),
                spread=float(spread_pct),
                threshold=float(self.spread_threshold),
                trace_id=trace_id,
            )
            use_market_order = True

        # Condition 3: Explicit urgency flag (if present in metadata)
        if signal.metadata and signal.metadata.get("urgency") is True:
            logger.info(
                "order_type_selected_market_urgency",
                signal_id=str(signal.signal_id),
                trace_id=trace_id,
            )
            use_market_order = True

        if use_market_order:
            return ("Market", None)

        # Default: Limit order with price offset
        limit_price = await self._calculate_limit_price(signal, snapshot_price, spread_pct)
        logger.info(
            "order_type_selected_limit",
            signal_id=str(signal.signal_id),
            limit_price=float(limit_price),
            snapshot_price=float(snapshot_price),
            trace_id=trace_id,
        )
        return ("Limit", limit_price)

    async def _calculate_limit_price(self, signal: TradingSignal, snapshot_price: Decimal, spread: Decimal) -> Decimal:
        """Calculate limit price with offset based on order side.

        Args:
            signal: Trading signal
            snapshot_price: Market price from signal snapshot
            spread: Current spread percentage

        Returns:
            Calculated limit price rounded to tick size
        """
        signal_type = signal.signal_type.lower()

        if signal_type == "buy":
            # Buy orders: limit price = snapshot_price - (spread * offset_ratio)
            # Slightly below market to improve fill probability
            price_offset = snapshot_price * (spread / Decimal("100")) * Decimal(str(self.price_offset_ratio))
            limit_price = snapshot_price - price_offset
        elif signal_type == "sell":
            # Sell orders: limit price = snapshot_price + (spread * offset_ratio)
            # Slightly above market to improve fill probability
            price_offset = snapshot_price * (spread / Decimal("100")) * Decimal(str(self.price_offset_ratio))
            limit_price = snapshot_price + price_offset
        else:
            # Fallback: use snapshot price
            limit_price = snapshot_price

        # Ensure price is positive
        if limit_price <= 0:
            limit_price = snapshot_price

        # Round to tick size
        limit_price = await self._round_price_to_tick_size(signal.asset, limit_price, signal_type)

        return limit_price

    async def _round_price_to_tick_size(self, asset: str, price: Decimal, side: str) -> Decimal:
        """Round price to tick size for the given asset.

        Args:
            asset: Trading pair symbol
            price: Price to round
            side: Order side ('buy' or 'sell')

        Returns:
            Rounded price
        """
        try:
            instrument_info = await self.instrument_info_manager.get_instrument_info(asset)
            if instrument_info and instrument_info.price_tick_size > 0:
                tick_size = instrument_info.price_tick_size
            else:
                # Fallback: use reasonable default tick sizes based on asset
                if "BTC" in asset.upper():
                    tick_size = Decimal("0.01")  # BTCUSDT tick size is usually 0.01
                elif "ETH" in asset.upper():
                    tick_size = Decimal("0.01")  # ETHUSDT tick size is usually 0.01
                else:
                    tick_size = Decimal("0.01")
            
            # Use quantize directly with tick_size to ensure proper rounding
            if side.lower() == "buy":
                rounded_price = price.quantize(tick_size, rounding=ROUND_DOWN)
            else:
                rounded_price = price.quantize(tick_size, rounding=ROUND_UP)
            
            logger.debug(
                "price_rounded_to_tick_size",
                asset=asset,
                original_price=float(price),
                rounded_price=float(rounded_price),
                tick_size=float(tick_size),
                side=side,
            )
            return rounded_price
        except Exception as e:
            logger.warning(
                "price_rounding_failed",
                asset=asset,
                price=float(price),
                error=str(e),
                reason="Failed to round price to tick size, using quantize with default",
            )
            # Fallback: use default tick size
            default_tick_size = Decimal("0.01")
            try:
                if side.lower() == "buy":
                    return price.quantize(default_tick_size, rounding=ROUND_DOWN)
                else:
                    return price.quantize(default_tick_size, rounding=ROUND_UP)
            except Exception:
                return price

    def get_time_in_force(self, order_type: str) -> str:
        """Get time-in-force value for order type.

        Args:
            order_type: 'Market' or 'Limit'

        Returns:
            Time-in-force string: 'GTC' (Good Till Cancel) for limit, 'IOC' (Immediate or Cancel) for market
        """
        if order_type == "Market":
            return "IOC"  # Immediate or Cancel
        return "GTC"  # Good Till Cancel

    def should_use_post_only(self, order_type: str) -> bool:
        """Determine if post_only flag should be set.

        Args:
            order_type: 'Market' or 'Limit'

        Returns:
            True for limit orders (to ensure maker fees), False for market orders
        """
        return order_type == "Limit"

    def should_use_reduce_only(self, signal: TradingSignal, has_position: bool, position_size: Decimal) -> bool:
        """Determine if reduce_only flag should be set.

        Args:
            signal: Trading signal
            has_position: Whether position exists for this asset
            position_size: Current position size (positive = long, negative = short)

        Returns:
            True if order should reduce position, False otherwise
        """
        if not has_position:
            return False

        signal_type = signal.signal_type.lower()

        # For sell signals: reduce_only if we have a long position (positive size)
        if signal_type == "sell" and position_size > 0:
            return True

        # For buy signals: reduce_only if we have a short position (negative size)
        if signal_type == "buy" and position_size < 0:
            return True

        return False

