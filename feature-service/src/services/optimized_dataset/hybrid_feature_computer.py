"""
Hybrid Feature Computer for optimized dataset building.

Combines vectorized computation (where possible) with streaming computation
(where needed, e.g., for orderbook features).
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
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
        feature_registry: Optional[Any] = None,
        target_horizon_minutes: int = 0,
    ):
        """
        Initialize hybrid feature computer.
        
        Args:
            requirements: DataRequirements from Feature Registry analysis
            feature_registry_version: Feature Registry version
            feature_registry: Optional FeatureRegistry instance for lookback info
            target_horizon_minutes: Target horizon in minutes
        """
        self.requirements = requirements
        self.feature_registry_version = feature_registry_version
        self.feature_registry = feature_registry
        self.target_horizon_minutes = target_horizon_minutes
        
        logger.info(
            "hybrid_feature_computer_initialized",
            needs_orderbook=requirements.needs_orderbook,
            needs_trades=requirements.needs_trades,
            needs_klines=requirements.needs_klines,
            feature_groups=list(requirements.feature_groups.keys()),
            target_horizon_minutes=target_horizon_minutes,
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
        
        # Ensure timestamps are datetime and timezone-aware UTC
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps, utc=True)
        else:
            # Normalize timezone to UTC
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize(timezone.utc)
            else:
                timestamps = timestamps.dt.tz_convert(timezone.utc)
        
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
            # Both DataFrames should have the same length (same timestamps), but verify
            if len(candle_pattern_features) != len(result):
                raise ValueError(
                    f"Length mismatch when merging candle pattern features: "
                    f"result has {len(result)} rows, candle_pattern_features has {len(candle_pattern_features)} rows. "
                    f"This should not happen - both should be computed from the same timestamps. "
                    f"Timestamps length: {len(timestamps)}, result shape: {result.shape}, "
                    f"candle_pattern_features shape: {candle_pattern_features.shape}, "
                    f"result index: {result.index[:5].tolist()}, candle_pattern_features index: {candle_pattern_features.index[:5].tolist()}"
                )
            
            # Reset indices to ensure alignment (both should have default RangeIndex starting from 0)
            # This is a safety measure in case indices somehow don't align
            result_aligned = result.reset_index(drop=True)
            candle_aligned = candle_pattern_features.reset_index(drop=True)
            
            # Verify lengths match after reset
            if len(candle_aligned) != len(result_aligned):
                raise ValueError(
                    f"Length mismatch after index reset: "
                    f"result_aligned has {len(result_aligned)} rows, candle_aligned has {len(candle_aligned)} rows."
                )
            
            # Assign columns - use .values to assign by position, not by index
            for col in candle_aligned.columns:
                if col != "timestamp":
                    # Convert to numpy array to ensure proper length and avoid index alignment issues
                    result_aligned[col] = candle_aligned[col].values
            
            # Return aligned result
            result = result_aligned
        
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
        
        # Ensure timestamp is datetime and timezone-aware UTC
        klines_df = klines_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(klines_df["timestamp"]):
            klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], utc=True)
        else:
            # Normalize timezone to UTC
            if klines_df["timestamp"].dt.tz is None:
                klines_df["timestamp"] = klines_df["timestamp"].dt.tz_localize(timezone.utc)
            else:
                klines_df["timestamp"] = klines_df["timestamp"].dt.tz_convert(timezone.utc)
        
        # Normalize timestamps Series to timezone-aware UTC
        timestamps_normalized = timestamps.copy()
        if not pd.api.types.is_datetime64_any_dtype(timestamps_normalized):
            timestamps_normalized = pd.to_datetime(timestamps_normalized, utc=True)
        else:
            if timestamps_normalized.dt.tz is None:
                timestamps_normalized = timestamps_normalized.dt.tz_localize(timezone.utc)
            else:
                timestamps_normalized = timestamps_normalized.dt.tz_convert(timezone.utc)
        
        klines_sorted = klines_df.sort_values("timestamp").reset_index(drop=True)
        
        for idx, ts in enumerate(timestamps_normalized):
            # Ensure ts is timezone-aware UTC datetime for comparison
            if isinstance(ts, pd.Timestamp):
                ts_dt = ts.to_pydatetime()
            else:
                ts_dt = ts
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            else:
                ts_dt = ts_dt.astimezone(timezone.utc)
            
            klines_before = klines_sorted[klines_sorted["timestamp"] <= ts_dt]
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
            
        Raises:
            ValueError: If required data is missing for any feature computation
        """
        # Create result DataFrame with explicit reset of index to ensure RangeIndex starting from 0
        # This prevents issues when timestamps Series has non-zero starting index (e.g., from iloc slicing in batches)
        # Reset index first to ensure timestamps has RangeIndex, then create DataFrame
        timestamps_reset = timestamps.reset_index(drop=True) if hasattr(timestamps, 'reset_index') else timestamps
        result = pd.DataFrame({"timestamp": timestamps_reset})
        
        # Determine which version to use
        if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
            compute_patterns = compute_all_candle_patterns_15m
        elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
            compute_patterns = compute_all_candle_patterns_5m
        else:
            compute_patterns = compute_all_candle_patterns_3m
        
        # Get feature names and lookback windows from registry
        feature_lookbacks = {}
        candle_pattern_feature_names = []
        if self.feature_registry and self.feature_registry.features:
            for feature in self.feature_registry.features:
                if feature.name.startswith(("candle_", "pattern_")):
                    lookback_str = feature.lookback_window or "0m"
                    lookback_minutes = self._parse_lookback_window(lookback_str)
                    feature_lookbacks[feature.name] = lookback_minutes or 0
                    candle_pattern_feature_names.append(feature.name)
        
        # Initialize all candle pattern feature columns with None
        # This ensures all rows have these columns, even if computation fails for some rows
        # Initialize columns directly in DataFrame (don't create separate Series to avoid length issues)
        for feature_name in candle_pattern_feature_names:
            result[feature_name] = None
        
        # Get feature names from first computation (will be set on first successful computation)
        first_patterns = None
        
        for idx, ts in enumerate(timestamps):
            # Normalize timestamp
            if isinstance(ts, pd.Timestamp):
                ts_dt = ts.to_pydatetime()
            else:
                ts_dt = ts
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            else:
                ts_dt = ts_dt.astimezone(timezone.utc)
            
            # Calculate window boundaries
            max_lookback = self.requirements.max_lookback_minutes
            window_start = ts_dt - timedelta(minutes=max_lookback)
            window_end = ts_dt
            target_timestamp = ts_dt + timedelta(minutes=self.target_horizon_minutes)
            
            # Get rolling windows for this timestamp
            rolling_windows = rolling_window.get_window(ts_dt)
            
            # Check data availability
            missing_data = []
            has_klines = False
            has_trades = False
            klines_count = 0
            trades_count = 0
            klines_in_last_15m_count = 0
            klines_in_last_15m = pd.Series(dtype='datetime64[ns, UTC]')  # Initialize to avoid NameError
            
            # Determine required minutes and count for candle patterns based on version
            if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                # compute_all_candle_patterns_15m: need at least 5 1-minute klines in last 15 minutes
                required_minutes = 15
                required_count = 5
            elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                # compute_all_candle_patterns_5m: need at least 5 1-minute klines in last 5 minutes
                required_minutes = 5
                required_count = 5
            else:
                # compute_all_candle_patterns_3m: need at least 3 1-minute klines in last 3 minutes
                required_minutes = 3
                required_count = 3
            
            if rolling_windows:
                # Check klines availability
                # Use get_klines_for_window to check data in the actual time window needed for candle patterns
                # This matches what compute_all_candle_patterns_15m will use
                klines_1m = rolling_windows.get_klines_for_window(
                    "1m",
                    ts_dt - timedelta(minutes=required_minutes + 1),  # Add 1 minute buffer
                    ts_dt
                )
                if klines_1m is not None and not klines_1m.empty:
                    has_klines = True
                    klines_count = len(klines_1m)
                    
                    # Check if we have klines in the required time window for candle patterns
                    # This is more important than general max_lookback for candle patterns
                    if klines_count > 0 and "timestamp" in klines_1m.columns:
                        klines_timestamps = klines_1m["timestamp"]
                        # Normalize timestamps
                        if not pd.api.types.is_datetime64_any_dtype(klines_timestamps):
                            klines_timestamps = pd.to_datetime(klines_timestamps, utc=True)
                        else:
                            if klines_timestamps.dt.tz is None:
                                klines_timestamps = klines_timestamps.dt.tz_localize(timezone.utc)
                            else:
                                klines_timestamps = klines_timestamps.dt.tz_convert(timezone.utc)
                        
                        # Check if we have at least required_count klines in the last required_minutes minutes
                        lookback_start = ts_dt - timedelta(minutes=required_minutes)
                        klines_in_window = klines_timestamps[klines_timestamps >= lookback_start]
                        klines_in_window_count = len(klines_in_window)
                        
                        # Store for logging (use last 15m count for compatibility)
                        if required_minutes == 15:
                            klines_in_last_15m = klines_in_window
                            klines_in_last_15m_count = klines_in_window_count
                        
                        if klines_in_window_count < required_count:
                            missing_data.append(
                                f"klines_1m_in_last_{required_minutes}m (only {klines_in_window_count}, "
                                f"need at least {required_count} for candle patterns)"
                            )
                    elif klines_count > 0:
                        # Have klines but no timestamp column - this is a data quality issue
                        missing_data.append("klines_1m (missing timestamp column)")
                else:
                    missing_data.append("klines_1m (empty)")
                
                # Check trades availability (if needed)
                if self.requirements.needs_trades:
                    # Check different trade windows
                    for window_name in ["1s", "3s", "15s", "1m"]:
                        trades_window = rolling_windows.get_window_data(window_name)
                        if trades_window is not None and not trades_window.empty:
                            has_trades = True
                            trades_count += len(trades_window)
                            break
                    if not has_trades:
                        missing_data.append("trades (all windows empty)")
            else:
                missing_data.append("rolling_window (None)")
            
            # Log details for each row (every 100th row or first row)
            if idx == 0 or idx % 100 == 0:
                # Determine candle pattern requirements for logging
                if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                    candle_pattern_minutes = 15
                    candle_pattern_count = 5
                elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                    candle_pattern_minutes = 5
                    candle_pattern_count = 5
                else:
                    candle_pattern_minutes = 3
                    candle_pattern_count = 3
                
                logger.info(
                    "feature_computation_row_details",
                    row_index=idx,
                    current_timestamp=ts_dt.isoformat(),
                    window_start=window_start.isoformat(),
                    window_end=window_end.isoformat(),
                    target_timestamp=target_timestamp.isoformat(),
                    max_lookback_minutes=max_lookback,
                    target_horizon_minutes=self.target_horizon_minutes,
                    has_klines=has_klines,
                    has_trades=has_trades,
                    klines_count=klines_count,
                    trades_count=trades_count,
                    klines_in_last_15m=klines_in_last_15m_count,
                    candle_pattern_required_minutes=candle_pattern_minutes,
                    candle_pattern_required_count=candle_pattern_count,
                    missing_data=missing_data if missing_data else None,
                    required_data_types=list(self.requirements.required_data_types),
                )
            
            # Fail immediately if required data is missing
            if missing_data:
                required_data_types = []
                if self.requirements.needs_klines:
                    required_data_types.append("klines")
                if self.requirements.needs_trades:
                    required_data_types.append("trades")
                
                logger.error(
                    "feature_computation_missing_data",
                    row_index=idx,
                    current_timestamp=ts_dt.isoformat(),
                    window_start=window_start.isoformat(),
                    window_end=window_end.isoformat(),
                    target_timestamp=target_timestamp.isoformat(),
                    missing_data=missing_data,
                    required_data_types=required_data_types,
                )
                raise ValueError(
                    f"Missing required data at timestamp {ts_dt.isoformat()}: {', '.join(missing_data)}. "
                    f"Required data types: {', '.join(required_data_types)}"
                )
            
            if rolling_windows:
                patterns = compute_patterns(rolling_windows)
                
                # Check if patterns returned empty dict (all None values)
                # This happens when compute_all_candle_patterns_* doesn't have enough data
                # This should not happen if our pre-check above worked correctly, but we check anyway
                # Also check for partially None values - if any lookback feature has None, fail immediately
                if not patterns or all(v is None for v in patterns.values()):
                    # Determine which version was used for better error message
                    if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                        required_minutes = 15
                        required_count = 5
                    elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                        required_minutes = 5
                        required_count = 5
                    else:
                        required_minutes = 3
                        required_count = 3
                    
                    logger.error(
                        "feature_computation_patterns_empty",
                        row_index=idx,
                        current_timestamp=ts_dt.isoformat(),
                        window_start=window_start.isoformat(),
                        window_end=window_end.isoformat(),
                        target_timestamp=target_timestamp.isoformat(),
                        klines_count=klines_count,
                        klines_in_last_15m=klines_in_last_15m_count,
                        required_minutes=required_minutes,
                        required_count=required_count,
                    )
                    raise ValueError(
                        f"Pattern computation returned empty result at timestamp {ts_dt.isoformat()}. "
                        f"This indicates insufficient data for candle pattern computation. "
                        f"Klines count: {klines_count}, klines in last {required_minutes}m: {klines_in_last_15m_count} "
                        f"(need at least {required_count}). "
                        f"This should have been caught by pre-validation - possible data inconsistency."
                    )
                
                # Store feature names from first successful computation
                # Columns are already initialized above, so we just store the keys for validation
                if first_patterns is None:
                    first_patterns = patterns
                    # Verify all expected columns are present (already initialized above)
                    missing_cols = set(patterns.keys()) - set(result.columns)
                    if missing_cols:
                        # Initialize any missing columns (should not happen, but safety check)
                        for feature_name in missing_cols:
                            result[feature_name] = None
                
                # Check if patterns contains fewer keys than initialized columns
                # This can happen if compute_all_candle_patterns_* returns incomplete dict
                # even though it should return either full dict or _get_empty_features_dict()
                if first_patterns is not None:
                    missing_keys = set(first_patterns.keys()) - set(patterns.keys())
                    if missing_keys:
                        logger.error(
                            "feature_computation_patterns_missing_keys",
                            row_index=idx,
                            current_timestamp=ts_dt.isoformat(),
                            window_start=window_start.isoformat(),
                            window_end=window_end.isoformat(),
                            target_timestamp=target_timestamp.isoformat(),
                            missing_keys=list(missing_keys),
                            expected_keys_count=len(first_patterns.keys()),
                            actual_keys_count=len(patterns.keys()),
                            expected_keys=list(first_patterns.keys()),
                            actual_keys=list(patterns.keys()),
                            klines_count=klines_count,
                            klines_in_last_15m=klines_in_last_15m_count,
                        )
                        raise ValueError(
                            f"Pattern computation returned incomplete dict at timestamp {ts_dt.isoformat()}. "
                            f"Missing keys: {', '.join(sorted(missing_keys))}. "
                            f"Expected {len(first_patterns.keys())} keys, got {len(patterns.keys())}. "
                            f"This should not happen - compute_all_candle_patterns_* should return either "
                            f"full dict or _get_empty_features_dict() (all None)."
                        )
                    
                    # Check for extra keys (should not happen, but log for debugging)
                    extra_keys = set(patterns.keys()) - set(first_patterns.keys())
                    if extra_keys:
                        logger.warning(
                            "feature_computation_patterns_extra_keys",
                            row_index=idx,
                            current_timestamp=ts_dt.isoformat(),
                            extra_keys=list(extra_keys),
                            expected_keys_count=len(first_patterns.keys()),
                            actual_keys_count=len(patterns.keys()),
                        )
                
                # Check for partially None values in patterns BEFORE assigning to DataFrame
                # This catches cases where compute_all_candle_patterns_* returns some None values
                # even though pre-validation passed
                none_features = []
                for feature_name, value in patterns.items():
                    if value is None:
                        feature_lookback = feature_lookbacks.get(feature_name, 0)
                        if feature_lookback > 0:  # Only check lookback features
                            none_features.append(feature_name)
                
                if none_features:
                    # Determine which version was used for better error message
                    if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                        required_minutes = 15
                        required_count = 5
                    elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                        required_minutes = 5
                        required_count = 5
                    else:
                        required_minutes = 3
                        required_count = 3
                    
                    logger.error(
                        "feature_computation_patterns_partially_none",
                        row_index=idx,
                        current_timestamp=ts_dt.isoformat(),
                        window_start=window_start.isoformat(),
                        window_end=window_end.isoformat(),
                        target_timestamp=target_timestamp.isoformat(),
                        none_features=none_features,
                        klines_count=klines_count,
                        klines_in_last_15m=klines_in_last_15m_count,
                        required_minutes=required_minutes,
                        required_count=required_count,
                    )
                    raise ValueError(
                        f"Pattern computation returned None values for lookback features at timestamp {ts_dt.isoformat()}: {', '.join(none_features)}. "
                        f"This indicates insufficient data for candle pattern computation. "
                        f"Klines count: {klines_count}, klines in last {required_minutes}m: {klines_in_last_15m_count} "
                        f"(need at least {required_count}). "
                        f"This should have been caught by pre-validation - possible data inconsistency."
                    )
                
                # Log details for each feature in this row
                for feature_name, value in patterns.items():
                    if feature_name in result.columns:
                        # Determine required data sources for this feature
                        feature_lookback = feature_lookbacks.get(feature_name, 0)
                        feature_lookback_timestamp = ts_dt - timedelta(minutes=feature_lookback)
                        
                        required_sources = []
                        if self.feature_registry:
                            for feature in self.feature_registry.features:
                                if feature.name == feature_name:
                                    required_sources = list(feature.input_sources) if feature.input_sources else []
                                    break
                        
                        # Fail if feature value is None and it's a required feature (check for ALL rows, not just logged ones)
                        # This is a redundant check, but kept for safety
                        if value is None and feature_lookback > 0:
                            logger.error(
                                "feature_computation_feature_missing_value",
                                row_index=idx,
                                feature_name=feature_name,
                                feature_lookback_minutes=feature_lookback,
                                feature_lookback_timestamp=feature_lookback_timestamp.isoformat(),
                                current_timestamp=ts_dt.isoformat(),
                                window_start=window_start.isoformat(),
                                window_end=window_end.isoformat(),
                                target_timestamp=target_timestamp.isoformat(),
                                required_data_sources=required_sources,
                                klines_count=klines_count,
                                klines_in_last_15m=klines_in_last_15m_count,
                                trades_count=trades_count,
                            )
                            raise ValueError(
                                f"Feature '{feature_name}' returned None at timestamp {ts_dt.isoformat()}. "
                                f"Lookback: {feature_lookback} minutes, lookback timestamp: {feature_lookback_timestamp.isoformat()}. "
                                f"Required data sources: {', '.join(required_sources)}. "
                                f"Available: klines={klines_count}, klines_in_last_15m={klines_in_last_15m_count}, trades={trades_count}"
                            )
                        
                        # Set value in result DataFrame using .at[] for direct assignment
                        # Use .loc[] as fallback if .at[] doesn't work (shouldn't happen, but safety)
                        try:
                            result.at[idx, feature_name] = value
                        except Exception as e:
                            # Fallback to .loc[] if .at[] fails
                            logger.warning(
                                "feature_computation_at_failed_using_loc",
                                row_index=idx,
                                feature_name=feature_name,
                                error=str(e),
                                error_type=type(e).__name__,
                            )
                            result.loc[idx, feature_name] = value
                        
                        # Verify value was written correctly (for debugging)
                        written_value = result.at[idx, feature_name]
                        if written_value != value and not (pd.isna(written_value) and pd.isna(value)):
                            logger.warning(
                                "feature_computation_value_mismatch_after_write",
                                row_index=idx,
                                feature_name=feature_name,
                                expected_value=value,
                                actual_value=written_value,
                                expected_type=type(value).__name__,
                                actual_type=type(written_value).__name__,
                            )
                        
                        # Log feature-specific details (every 100th row or if value is None)
                        if idx == 0 or idx % 100 == 0 or value is None:
                            logger.info(
                                "feature_computation_feature_details",
                                row_index=idx,
                                feature_name=feature_name,
                                feature_lookback_minutes=feature_lookback,
                                feature_lookback_timestamp=feature_lookback_timestamp.isoformat(),
                                feature_value=value,
                                current_timestamp=ts_dt.isoformat(),
                                target_timestamp=target_timestamp.isoformat(),
                                required_data_sources=required_sources,
                                has_required_data=value is not None,
                            )
            else:
                # rolling_windows is None - this should not happen after our check above
                # But if it does, log it and leave values as None (which will be NaN)
                logger.warning(
                    "feature_computation_rolling_windows_none_after_check",
                    row_index=idx,
                    current_timestamp=ts_dt.isoformat(),
                    window_start=window_start.isoformat(),
                    window_end=window_end.isoformat(),
                )
        
        return result
    
    def _parse_lookback_window(self, lookback_window: str) -> Optional[int]:
        """
        Parse lookback_window string to minutes.
        
        Args:
            lookback_window: Lookback window string (e.g., "21m", "5m", "0s", "1h")
            
        Returns:
            Lookback period in minutes, or None if parsing fails
        """
        if not lookback_window:
            return None
        
        try:
            unit = lookback_window[-1]
            value = int(lookback_window[:-1])
            
            # Convert to minutes
            if unit == "s":
                return value // 60  # Convert seconds to minutes (round down)
            elif unit == "m":
                return value
            elif unit == "h":
                return value * 60
            elif unit == "d":
                return value * 24 * 60
            else:
                return None
        except (ValueError, IndexError):
            return None

