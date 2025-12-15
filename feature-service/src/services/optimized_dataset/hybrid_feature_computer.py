"""
Hybrid Feature Computer for optimized dataset building.

Combines vectorized computation (where possible) with streaming computation
(where needed, e.g., for orderbook features).
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import structlog

from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from .requirements_analyzer import DataRequirements
from .vectorized_features import (
    compute_technical_indicators_vectorized,
    compute_orderflow_features_vectorized,
    compute_price_features_vectorized,
)
from src.features.orderbook_features import compute_all_orderbook_features
from src.features.perpetual_features import compute_all_perpetual_features
from src.features.temporal_features import compute_all_temporal_features
from src.features.candle_patterns import (
    compute_all_candle_patterns_3m,
    compute_all_candle_patterns_5m,
    compute_all_candle_patterns_15m,
)

logger = structlog.get_logger(__name__)


class HybridFeatureComputer:
    """
    Hybrid feature computer combining vectorization and streaming computation.
    
    Uses vectorized computation for:
    - Technical indicators (EMA, RSI)
    - Price features (returns, volatility, VWAP)
    - Orderflow features (signed volume, etc.)
    
    Uses streaming computation for:
    - Orderbook features (requires orderbook state per timestamp)
    - Perpetual features (funding rate)
    - Temporal features (time of day)
    - Candle pattern features (complex pattern matching)
    """
    
    def __init__(
        self,
        requirements: DataRequirements,
        feature_registry_version: str = "1.0.0",
    ):
        """
        Initialize hybrid feature computer.
        
        Args:
            requirements: DataRequirements from Feature Registry analysis
            feature_registry_version: Feature Registry version
        """
        self.requirements = requirements
        self.feature_registry_version = feature_registry_version
        
        logger.info(
            "hybrid_feature_computer_initialized",
            needs_orderbook=requirements.needs_orderbook,
            needs_trades=requirements.needs_trades,
            needs_klines=requirements.needs_klines,
            feature_groups=list(requirements.feature_groups.keys()),
        )
    
    def compute_features_batch(
        self,
        timestamps: pd.Series,
        rolling_window: "OptimizedRollingWindow",
        klines_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        orderbook_states: Optional[Dict[datetime, OrderbookState]] = None,
        funding_rates: Optional[Dict[datetime, float]] = None,
        next_funding_times: Optional[Dict[datetime, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute features for multiple timestamps using hybrid approach.
        
        Args:
            timestamps: Series of timestamps to compute features for
            rolling_window: OptimizedRollingWindow instance
            klines_df: DataFrame with klines
            trades_df: DataFrame with trades
            orderbook_states: Optional dict mapping timestamp to OrderbookState
            funding_rates: Optional dict mapping timestamp to funding rate
            next_funding_times: Optional dict mapping timestamp to next funding time
            
        Returns:
            DataFrame with features for all timestamps
        """
        if timestamps.empty:
            return pd.DataFrame()
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps, utc=True)
        
        # Initialize result DataFrame
        result = pd.DataFrame({"timestamp": timestamps})
        
        # Step 1: Vectorized computation for technical indicators
        if "technical" in self.requirements.feature_groups:
            technical_df = compute_technical_indicators_vectorized(
                klines_df=klines_df,
                timestamps=timestamps,
                period_ema=21,
                period_rsi=14,
            )
            # Merge technical indicators
            for col in ["ema_21", "rsi_14"]:
                if col in technical_df.columns:
                    result[col] = technical_df[col]
        
        # Step 2: Vectorized computation for orderflow features
        if "orderflow" in self.requirements.feature_groups and not trades_df.empty:
            orderflow_df = compute_orderflow_features_vectorized(
                trades_df=trades_df,
                timestamps=timestamps,
                windows=[1, 3, 15, 60],
            )
            # Merge orderflow features
            for col in orderflow_df.columns:
                if col != "timestamp":
                    result[col] = orderflow_df[col]
        
        # Step 3: Vectorized computation for price features
        if "price" in self.requirements.feature_groups:
            # Get current prices from klines
            current_prices = self._get_current_prices(timestamps, klines_df)
            
            price_df = compute_price_features_vectorized(
                klines_df=klines_df,
                trades_df=trades_df,
                timestamps=timestamps,
                current_prices=current_prices,
            )
            # Merge price features
            for col in price_df.columns:
                if col != "timestamp":
                    result[col] = price_df[col]
        
        # Step 4: Streaming computation for orderbook features (if needed)
        if self.requirements.needs_orderbook and orderbook_states:
            orderbook_features = self._compute_orderbook_features_streaming(
                timestamps=timestamps,
                orderbook_states=orderbook_states,
            )
            # Merge orderbook features
            for col in orderbook_features.columns:
                if col != "timestamp":
                    result[col] = orderbook_features[col]
        
        # Step 5: Streaming computation for perpetual features
        if "perpetual" in self.requirements.feature_groups:
            perpetual_features = self._compute_perpetual_features_streaming(
                timestamps=timestamps,
                funding_rates=funding_rates or {},
                next_funding_times=next_funding_times or {},
            )
            # Merge perpetual features
            for col in perpetual_features.columns:
                if col != "timestamp":
                    result[col] = perpetual_features[col]
        
        # Step 6: Streaming computation for temporal features
        if "temporal" in self.requirements.feature_groups:
            temporal_features = self._compute_temporal_features_streaming(
                timestamps=timestamps,
            )
            # Merge temporal features
            for col in temporal_features.columns:
                if col != "timestamp":
                    result[col] = temporal_features[col]
        
        # Step 7: Streaming computation for candle pattern features
        if "candle_patterns" in self.requirements.feature_groups:
            candle_pattern_features = self._compute_candle_patterns_streaming(
                timestamps=timestamps,
                rolling_window=rolling_window,
            )
            # Merge candle pattern features
            for col in candle_pattern_features.columns:
                if col != "timestamp":
                    result[col] = candle_pattern_features[col]
        
        return result
    
    def _get_current_prices(
        self, timestamps: pd.Series, klines_df: pd.DataFrame
    ) -> pd.Series:
        """
        Get current prices for each timestamp from klines.
        
        Args:
            timestamps: Series of timestamps
            klines_df: DataFrame with klines
            
        Returns:
            Series of current prices
        """
        current_prices = pd.Series([None] * len(timestamps))
        
        if klines_df.empty or "timestamp" not in klines_df.columns or "close" not in klines_df.columns:
            return current_prices
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(klines_df["timestamp"]):
            klines_df = klines_df.copy()
            klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], utc=True)
        
        klines_sorted = klines_df.sort_values("timestamp").reset_index(drop=True)
        
        for idx, ts in enumerate(timestamps):
            klines_before = klines_sorted[klines_sorted["timestamp"] <= ts]
            if not klines_before.empty:
                latest_close = klines_before.iloc[-1]["close"]
                current_prices.iloc[idx] = pd.to_numeric(latest_close, errors='coerce')
        
        return current_prices
    
    def _compute_orderbook_features_streaming(
        self,
        timestamps: pd.Series,
        orderbook_states: Dict[datetime, OrderbookState],
    ) -> pd.DataFrame:
        """
        Compute orderbook features for each timestamp (streaming).
        
        Args:
            timestamps: Series of timestamps
            orderbook_states: Dict mapping timestamp to OrderbookState
            
        Returns:
            DataFrame with orderbook features
        """
        result = pd.DataFrame({"timestamp": timestamps})
        
        # Initialize orderbook feature columns
        orderbook_feature_names = [
            "mid_price", "spread_abs", "spread_rel",
            "depth_bid_top5", "depth_bid_top10",
            "depth_ask_top5", "depth_ask_top10",
            "depth_imbalance_top5", "depth_imbalance_top10",
        ]
        
        for feature_name in orderbook_feature_names:
            result[feature_name] = None
        
        # Compute features for each timestamp
        for idx, ts in enumerate(timestamps):
            orderbook = orderbook_states.get(ts)
            if orderbook:
                features = compute_all_orderbook_features(orderbook)
                for feature_name, value in features.items():
                    if feature_name in result.columns:
                        result.at[idx, feature_name] = value
        
        return result
    
    def _compute_perpetual_features_streaming(
        self,
        timestamps: pd.Series,
        funding_rates: Dict[datetime, float],
        next_funding_times: Dict[datetime, int],
    ) -> pd.DataFrame:
        """
        Compute perpetual features for each timestamp (streaming).
        
        Args:
            timestamps: Series of timestamps
            funding_rates: Dict mapping timestamp to funding rate
            next_funding_times: Dict mapping timestamp to next funding time
            
        Returns:
            DataFrame with perpetual features
        """
        result = pd.DataFrame({"timestamp": timestamps})
        result["funding_rate"] = None
        result["time_to_funding"] = None
        
        for idx, ts in enumerate(timestamps):
            funding_rate = funding_rates.get(ts)
            next_funding_time = next_funding_times.get(ts)
            
            time_to_funding = None
            if next_funding_time and isinstance(ts, datetime):
                time_to_funding = (next_funding_time / 1000) - ts.timestamp()
            
            features = compute_all_perpetual_features(
                funding_rate=funding_rate,
                time_to_funding=time_to_funding,
            )
            
            for feature_name, value in features.items():
                if feature_name in result.columns:
                    result.at[idx, feature_name] = value
        
        return result
    
    def _compute_temporal_features_streaming(
        self,
        timestamps: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute temporal features for each timestamp (streaming).
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            DataFrame with temporal features
        """
        result = pd.DataFrame({"timestamp": timestamps})
        result["time_of_day_sin"] = None
        result["time_of_day_cos"] = None
        
        for idx, ts in enumerate(timestamps):
            if isinstance(ts, datetime):
                features = compute_all_temporal_features(ts)
                for feature_name, value in features.items():
                    if feature_name in result.columns:
                        result.at[idx, feature_name] = value
        
        return result
    
    def _compute_candle_patterns_streaming(
        self,
        timestamps: pd.Series,
        rolling_window: "OptimizedRollingWindow",
    ) -> pd.DataFrame:
        """
        Compute candle pattern features for each timestamp (streaming).
        
        Args:
            timestamps: Series of timestamps
            rolling_window: OptimizedRollingWindow instance
            
        Returns:
            DataFrame with candle pattern features
        """
        result = pd.DataFrame({"timestamp": timestamps})
        
        # Determine which version to use
        if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
            compute_patterns = compute_all_candle_patterns_15m
        elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
            compute_patterns = compute_all_candle_patterns_5m
        else:
            compute_patterns = compute_all_candle_patterns_3m
        
        # Get feature names from first computation
        first_patterns = None
        
        for idx, ts in enumerate(timestamps):
            # Get rolling windows for this timestamp
            rolling_windows = rolling_window.get_window(ts)
            
            if rolling_windows:
                patterns = compute_patterns(rolling_windows)
                
                # Initialize columns on first iteration
                if first_patterns is None:
                    first_patterns = patterns
                    for feature_name in patterns.keys():
                        result[feature_name] = None
                
                # Set values
                for feature_name, value in patterns.items():
                    if feature_name in result.columns:
                        result.at[idx, feature_name] = value
        
        return result

