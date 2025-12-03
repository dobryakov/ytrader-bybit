"""
Warm-up signal generation service.

Generates trading signals using simple heuristics or controlled random generation
when no trained model exists, allowing trading to begin immediately.
"""

import random
from typing import Optional, List
from datetime import datetime

from ..models.signal import TradingSignal, MarketDataSnapshot
from ..consumers.market_data_consumer import market_data_cache
from ..services.balance_calculator import balance_calculator
from ..config.settings import settings
from ..config.logging import get_logger, bind_context

logger = get_logger(__name__)


class WarmUpSignalGenerator:
    """Generates trading signals using warm-up heuristics."""

    def __init__(
        self,
        min_amount: float = 100.0,
        max_amount: float = 1000.0,
        randomness_level: float = 0.5,
    ):
        """
        Initialize warm-up signal generator.

        Args:
            min_amount: Minimum order amount in quote currency
            max_amount: Maximum order amount in quote currency
            randomness_level: Level of randomness (0.0 = deterministic, 1.0 = fully random)
        """
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.randomness_level = randomness_level

    async def generate_signal(
        self,
        asset: str,
        strategy_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a warm-up trading signal.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            TradingSignal or None if market data unavailable
        """
        bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

        logger.info("Generating warm-up signal", asset=asset, strategy_id=strategy_id)

        # Retrieve market data snapshot with freshness checks
        market_data = market_data_cache.get_market_data(
            asset,
            max_age_seconds=settings.market_data_max_age_seconds,
            stale_warning_threshold_seconds=settings.market_data_stale_warning_threshold_seconds,
        )
        if not market_data:
            logger.warning(
                "Market data unavailable, skipping signal generation",
                asset=asset,
                strategy_id=strategy_id,
            )
            return None

        # Generate signal using heuristics or random generation
        signal_type = self._determine_signal_type(asset, market_data)
        base_amount = self._determine_amount()
        
        # Check available balance and adapt amount
        # For SELL signals, pass current_price to convert quote currency to base currency
        current_price = float(market_data.price) if market_data.price else None
        adapted_amount = await balance_calculator.calculate_affordable_amount(
            trading_pair=asset,
            signal_type=signal_type,
            requested_amount=base_amount,
            current_price=current_price,
        )
        
        if adapted_amount is None:
            logger.warning(
                "Insufficient balance, skipping signal generation",
                asset=asset,
                strategy_id=strategy_id,
                signal_type=signal_type,
                requested_amount=base_amount,
                trace_id=trace_id,
            )
            return None
        
        # Use adapted amount
        amount = adapted_amount
        if amount != base_amount:
            logger.info(
                "Adapted signal amount to available balance",
                asset=asset,
                strategy_id=strategy_id,
                original_amount=base_amount,
                adapted_amount=amount,
                trace_id=trace_id,
            )
        
        confidence = self._calculate_confidence(market_data)

        # Create market data snapshot
        market_snapshot = MarketDataSnapshot(
            price=market_data["price"],
            spread=market_data["spread"],
            volume_24h=market_data["volume_24h"],
            volatility=market_data["volatility"],
            orderbook_depth=market_data.get("orderbook_depth"),
        )

        # Create signal
        signal = TradingSignal(
            signal_type=signal_type,
            asset=asset,
            amount=amount,
            confidence=confidence,
            strategy_id=strategy_id,
            model_version=None,
            is_warmup=True,
            market_data_snapshot=market_snapshot,
            metadata={
                "reasoning": f"Warm-up signal: {signal_type} based on heuristics",
                "risk_score": self._calculate_risk_score(market_data),
                "randomness_level": self.randomness_level,
            },
            trace_id=trace_id,
        )

        logger.info(
            "Generated warm-up signal",
            asset=asset,
            strategy_id=strategy_id,
            signal_type=signal_type,
            amount=amount,
            confidence=confidence,
            trace_id=trace_id,
        )

        return signal

    def _determine_signal_type(self, asset: str, market_data: dict) -> str:
        """
        Determine signal type (buy/sell) using heuristics.

        Args:
            asset: Trading pair symbol
            market_data: Current market data

        Returns:
            'buy' or 'sell'
        """
        # Use randomness level to mix heuristics with random decisions
        if random.random() < self.randomness_level:
            # Fully random
            return random.choice(["buy", "sell"])
        else:
            # Heuristic-based: simple momentum strategy
            # If volatility is high and volume is increasing, consider buying
            # If price is high relative to recent volatility, consider selling
            volatility = market_data.get("volatility", 0.0)
            volume_24h = market_data.get("volume_24h", 0.0)

            # Simple heuristic: high volume + moderate volatility = buy signal
            if volume_24h > 1000000 and 0.01 < volatility < 0.05:
                return "buy"
            # High volatility = sell signal (risk reduction)
            elif volatility > 0.05:
                return "sell"
            # Default: random
            else:
                return random.choice(["buy", "sell"])

    def _determine_amount(self) -> float:
        """
        Determine order amount.

        Returns:
            Order amount in quote currency
        """
        if self.randomness_level > 0.5:
            # More random: wider range
            amount = random.uniform(self.min_amount, self.max_amount)
        else:
            # Less random: closer to middle
            mid_amount = (self.min_amount + self.max_amount) / 2
            range_size = (self.max_amount - self.min_amount) * (1 - self.randomness_level)
            amount = random.uniform(
                mid_amount - range_size / 2,
                mid_amount + range_size / 2,
            )

        return round(amount, 2)

    def _calculate_confidence(self, market_data: dict) -> float:
        """
        Calculate confidence score based on market data quality.

        Args:
            market_data: Current market data

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence for warm-up signals (lower than model-based)
        base_confidence = 0.5

        # Adjust based on data freshness
        last_updated = market_data.get("last_updated")
        if last_updated:
            age_seconds = (datetime.utcnow() - last_updated).total_seconds()
            max_age = settings.market_data_max_age_seconds
            warning_age = settings.market_data_stale_warning_threshold_seconds
            # Reduce confidence if data is stale or approaching staleness
            if age_seconds > max_age:
                base_confidence *= 0.7
            elif age_seconds > warning_age:
                base_confidence *= 0.9

        # Add some randomness based on randomness level
        if self.randomness_level > 0.5:
            confidence = base_confidence + random.uniform(-0.2, 0.2)
        else:
            confidence = base_confidence + random.uniform(-0.1, 0.1)

        # Clamp to valid range
        return max(0.1, min(0.9, confidence))

    def _calculate_risk_score(self, market_data: dict) -> float:
        """
        Calculate risk score based on market conditions.

        Args:
            market_data: Current market data

        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk)
        """
        volatility = market_data.get("volatility", 0.0)
        spread = market_data.get("spread", 0.0)
        price = market_data.get("price", 1.0)

        # Risk increases with volatility and spread
        volatility_risk = min(1.0, volatility * 10)  # Scale volatility to 0-1
        spread_risk = min(1.0, (spread / price) * 100) if price > 0 else 0.5  # Spread as % of price

        # Combined risk score
        risk_score = (volatility_risk * 0.7 + spread_risk * 0.3)

        return round(min(1.0, max(0.0, risk_score)), 2)

    async def generate_signals_for_strategies(
        self,
        assets: List[str],
        strategy_ids: List[str],
        trace_id: Optional[str] = None,
    ) -> List[TradingSignal]:
        """
        Generate warm-up signals for multiple strategies and assets.

        Args:
            assets: List of trading pair symbols
            strategy_ids: List of strategy identifiers
            trace_id: Trace ID for request flow tracking

        Returns:
            List of generated signals
        """
        signals = []
        for strategy_id in strategy_ids:
            for asset in assets:
                signal = await self.generate_signal(asset, strategy_id, trace_id)
                if signal:
                    signals.append(signal)
        return signals


# Initialize warm-up signal generator with settings
def get_warmup_generator():
    """Get warm-up signal generator instance with current settings."""
    return WarmUpSignalGenerator(
        min_amount=settings.warmup_min_amount,
        max_amount=settings.warmup_max_amount,
        randomness_level=settings.warmup_randomness_level,
    )

