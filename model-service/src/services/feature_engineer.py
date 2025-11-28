"""
Feature engineering service.

Processes execution events and market data into features for model training.
MUST use market_data_snapshot from trading signals for features describing
market state at decision time. Uses execution event market_conditions only
for performance metrics.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..models.execution_event import OrderExecutionEvent
from ..models.signal import MarketDataSnapshot
from ..config.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Engineers features from execution events and market data."""

    def __init__(self):
        """Initialize feature engineer."""
        pass

    def engineer_features(
        self,
        execution_events: List[OrderExecutionEvent],
        signal_market_data: Optional[Dict[str, MarketDataSnapshot]] = None,
    ) -> pd.DataFrame:
        """
        Engineer features from execution events and market data.

        Args:
            execution_events: List of order execution events
            signal_market_data: Optional dictionary mapping signal_id to MarketDataSnapshot
                               (for market state at decision time)

        Returns:
            DataFrame with engineered features (one row per execution event)
        """
        if not execution_events:
            logger.warning("No execution events provided for feature engineering")
            return pd.DataFrame()

        features_list = []

        for event in execution_events:
            # Get market data snapshot from signal (preferred) or fallback to execution event
            market_snapshot = None
            if signal_market_data and event.signal_id in signal_market_data:
                market_snapshot = signal_market_data[event.signal_id]
                logger.debug("Using signal market data snapshot", signal_id=event.signal_id)
            else:
                # Fallback: construct from execution event market_conditions
                # Note: This is less ideal as it represents execution time, not decision time
                logger.warning(
                    "Signal market data not available, using execution event market_conditions",
                    signal_id=event.signal_id,
                )
                # Create a minimal market snapshot from execution event
                # We'll use execution price as a proxy for price at decision time
                market_snapshot = MarketDataSnapshot(
                    price=event.signal_price,  # Use signal price as proxy
                    spread=event.market_conditions.spread,
                    volume_24h=event.market_conditions.volume_24h,
                    volatility=event.market_conditions.volatility,
                )

            # Engineer features for this event
            features = self._engineer_event_features(event, market_snapshot)
            features_list.append(features)

        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        logger.info("Engineered features", event_count=len(execution_events), feature_count=len(features_df.columns))
        return features_df

    def _engineer_event_features(
        self, event: OrderExecutionEvent, market_snapshot: MarketDataSnapshot
    ) -> Dict[str, Any]:
        """
        Engineer features for a single execution event.

        Args:
            event: Order execution event
            market_snapshot: Market data snapshot at decision time

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Price features (from market snapshot at decision time)
        features["price"] = market_snapshot.price
        features["price_log"] = np.log(market_snapshot.price) if market_snapshot.price > 0 else 0.0

        # Spread features
        features["spread"] = market_snapshot.spread
        features["spread_percent"] = (
            (market_snapshot.spread / market_snapshot.price * 100) if market_snapshot.price > 0 else 0.0
        )

        # Volume features
        features["volume_24h"] = market_snapshot.volume_24h
        features["volume_24h_log"] = np.log(market_snapshot.volume_24h + 1)  # +1 to avoid log(0)

        # Volatility features
        features["volatility"] = market_snapshot.volatility
        features["volatility_squared"] = market_snapshot.volatility**2

        # Order book depth features (if available)
        if market_snapshot.orderbook_depth:
            features["bid_depth"] = market_snapshot.orderbook_depth.get("bid_depth", 0.0)
            features["ask_depth"] = market_snapshot.orderbook_depth.get("ask_depth", 0.0)
            features["depth_imbalance"] = (
                (features["bid_depth"] - features["ask_depth"]) / (features["bid_depth"] + features["ask_depth"] + 1e-10)
            )
        else:
            features["bid_depth"] = 0.0
            features["ask_depth"] = 0.0
            features["depth_imbalance"] = 0.0

        # Technical indicators (if available)
        if market_snapshot.technical_indicators:
            indicators = market_snapshot.technical_indicators
            features["rsi"] = indicators.get("rsi", 50.0)  # Default to neutral
            features["macd"] = indicators.get("macd", 0.0)
            features["macd_signal"] = indicators.get("macd_signal", 0.0)
            features["macd_histogram"] = indicators.get("macd_histogram", 0.0)
            features["moving_average_20"] = indicators.get("moving_average_20", market_snapshot.price)
            features["moving_average_50"] = indicators.get("moving_average_50", market_snapshot.price)
            features["bollinger_upper"] = indicators.get("bollinger_upper", market_snapshot.price)
            features["bollinger_lower"] = indicators.get("bollinger_lower", market_snapshot.price)
            features["bollinger_width"] = (
                (features["bollinger_upper"] - features["bollinger_lower"]) / market_snapshot.price
                if market_snapshot.price > 0
                else 0.0
            )
        else:
            # Default values when technical indicators not available
            features["rsi"] = 50.0
            features["macd"] = 0.0
            features["macd_signal"] = 0.0
            features["macd_histogram"] = 0.0
            features["moving_average_20"] = market_snapshot.price
            features["moving_average_50"] = market_snapshot.price
            features["bollinger_upper"] = market_snapshot.price
            features["bollinger_lower"] = market_snapshot.price
            features["bollinger_width"] = 0.0

        # Execution features (from execution event)
        features["execution_price"] = event.execution_price
        features["execution_quantity"] = event.execution_quantity
        features["execution_fees"] = event.execution_fees
        features["execution_value"] = event.execution_price * event.execution_quantity

        # Slippage features (price difference between signal and execution)
        features["slippage"] = event.performance.slippage
        features["slippage_percent"] = event.performance.slippage_percent
        features["price_change"] = event.execution_price - event.signal_price
        features["price_change_percent"] = (
            ((event.execution_price - event.signal_price) / event.signal_price * 100)
            if event.signal_price > 0
            else 0.0
        )

        # Time features
        time_diff = (event.executed_at - event.signal_timestamp).total_seconds()
        features["execution_delay_seconds"] = time_diff
        features["execution_delay_minutes"] = time_diff / 60.0

        # Side features (categorical encoding)
        features["side_buy"] = 1 if event.side == "buy" else 0
        features["side_sell"] = 1 if event.side == "sell" else 0

        # Asset features (categorical - could be one-hot encoded if multiple assets)
        # For now, we'll use a simple hash-based encoding
        features["asset_hash"] = hash(event.asset) % 1000  # Simple hash encoding

        # Strategy features (categorical)
        features["strategy_hash"] = hash(event.strategy_id) % 100  # Simple hash encoding

        # Market context features (from execution event market_conditions - for performance metrics only)
        # These represent market state at execution time, not decision time
        features["execution_spread"] = event.market_conditions.spread
        features["execution_volume_24h"] = event.market_conditions.volume_24h
        features["execution_volatility"] = event.market_conditions.volatility

        # Market change features (difference between decision time and execution time)
        features["spread_change"] = event.market_conditions.spread - market_snapshot.spread
        features["volume_change"] = event.market_conditions.volume_24h - market_snapshot.volume_24h
        features["volatility_change"] = event.market_conditions.volatility - market_snapshot.volatility

        return features

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names that will be generated.

        Returns:
            List of feature names
        """
        return [
            "price",
            "price_log",
            "spread",
            "spread_percent",
            "volume_24h",
            "volume_24h_log",
            "volatility",
            "volatility_squared",
            "bid_depth",
            "ask_depth",
            "depth_imbalance",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "moving_average_20",
            "moving_average_50",
            "bollinger_upper",
            "bollinger_lower",
            "bollinger_width",
            "execution_price",
            "execution_quantity",
            "execution_fees",
            "execution_value",
            "slippage",
            "slippage_percent",
            "price_change",
            "price_change_percent",
            "execution_delay_seconds",
            "execution_delay_minutes",
            "side_buy",
            "side_sell",
            "asset_hash",
            "strategy_hash",
            "execution_spread",
            "execution_volume_24h",
            "execution_volatility",
            "spread_change",
            "volume_change",
            "volatility_change",
        ]


# Global feature engineer instance
feature_engineer = FeatureEngineer()

