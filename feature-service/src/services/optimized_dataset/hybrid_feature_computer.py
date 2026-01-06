"""
Hybrid Feature Computer for optimized dataset building.

Combines vectorized computation (where possible) with streaming computation
(where needed, e.g., for orderbook features).
"""
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import structlog
import asyncio

from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from .requirements_analyzer import DataRequirements
from .vectorized_features import (
    compute_technical_indicators_vectorized,
    compute_orderflow_features_vectorized,
    compute_price_features_vectorized,
)
import numpy as np
from src.features.orderbook_features import compute_all_orderbook_features
from src.features.perpetual_features import compute_all_perpetual_features
from src.features.temporal_features import compute_all_temporal_features
from src.features.candle_patterns import (
    compute_all_candle_patterns_3m,
    compute_all_candle_patterns_5m,
    compute_all_candle_patterns_15m,
    compute_all_candle_patterns_45m,
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
        backfilling_service: Optional[Any] = None,
    ):
        """
        Initialize hybrid feature computer.
        
        Args:
            requirements: DataRequirements from Feature Registry analysis
            feature_registry_version: Feature Registry version
            feature_registry: Optional FeatureRegistry instance for lookback info
            target_horizon_minutes: Target horizon in minutes
            backfilling_service: Optional BackfillingService for automatic backfill
        """
        self.requirements = requirements
        self.feature_registry_version = feature_registry_version
        self.feature_registry = feature_registry
        self.target_horizon_minutes = target_horizon_minutes
        self.backfilling_service = backfilling_service
        
        logger.info(
            "hybrid_feature_computer_initialized",
            needs_orderbook=requirements.needs_orderbook,
            needs_trades=requirements.needs_trades,
            needs_klines=requirements.needs_klines,
            feature_groups=list(requirements.feature_groups.keys()),
            target_horizon_minutes=target_horizon_minutes,
            backfilling_enabled=backfilling_service is not None,
        )
    
    async def _auto_backfill_missing_data(
        self,
        symbol: str,
        timestamp: datetime,
        missing_data: List[str],
        max_lookback_minutes: int,
    ) -> bool:
        """
        Automatically backfill missing data for the given timestamp.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Timestamp where data is missing
            missing_data: List of missing data types (e.g., ["klines_1m", "trades"])
            max_lookback_minutes: Maximum lookback window in minutes
            
        Returns:
            True if backfill was successful and data should be available, False otherwise
        """
        if not self.backfilling_service:
            logger.warning(
                "auto_backfill_not_available",
                symbol=symbol,
                timestamp=timestamp.isoformat(),
                missing_data=missing_data,
                message="BackfillingService not available, cannot auto-backfill",
            )
            return False
        
        # Determine which data types need backfilling
        data_types_to_backfill = []
        if any("klines" in item for item in missing_data):
            data_types_to_backfill.append("klines")
        if any("trades" in item for item in missing_data):
            data_types_to_backfill.append("trades")
        
        if not data_types_to_backfill:
            logger.warning(
                "auto_backfill_no_data_types",
                symbol=symbol,
                timestamp=timestamp.isoformat(),
                missing_data=missing_data,
                message="No backfillable data types found in missing_data",
            )
            return False
        
        # Calculate date range for backfill
        # We need to backfill from (timestamp - max_lookback) to timestamp
        # Add some buffer to ensure we have enough data
        buffer_minutes = 60  # 1 hour buffer
        start_time = timestamp - timedelta(minutes=max_lookback_minutes + buffer_minutes)
        end_time = timestamp + timedelta(minutes=buffer_minutes)
        
        start_date = start_time.date()
        end_date = end_time.date()
        
        logger.info(
            "auto_backfill_starting",
            symbol=symbol,
            timestamp=timestamp.isoformat(),
            missing_data=missing_data,
            data_types=data_types_to_backfill,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            max_lookback_minutes=max_lookback_minutes,
        )
        
        try:
            # Start backfill job
            job_id = await self.backfilling_service.backfill_historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_types=data_types_to_backfill,
            )
            
            # Wait for job to complete (with timeout)
            max_wait_seconds = 300  # 5 minutes
            wait_interval_seconds = 2
            waited_seconds = 0
            
            while waited_seconds < max_wait_seconds:
                status = self.backfilling_service.get_job_status(job_id)
                if not status:
                    logger.error(
                        "auto_backfill_job_not_found",
                        job_id=job_id,
                        symbol=symbol,
                    )
                    return False
                
                if status["status"] == "completed":
                    logger.info(
                        "auto_backfill_completed",
                        job_id=job_id,
                        symbol=symbol,
                        completed_dates=status.get("completed_dates", []),
                    )
                    return True
                elif status["status"] == "failed":
                    logger.error(
                        "auto_backfill_failed",
                        job_id=job_id,
                        symbol=symbol,
                        error_message=status.get("error_message"),
                        failed_dates=status.get("failed_dates", []),
                    )
                    return False
                
                # Job still in progress, wait a bit
                await asyncio.sleep(wait_interval_seconds)
                waited_seconds += wait_interval_seconds
            
            # Timeout reached
            logger.warning(
                "auto_backfill_timeout",
                job_id=job_id,
                symbol=symbol,
                waited_seconds=waited_seconds,
                max_wait_seconds=max_wait_seconds,
            )
            return False
            
        except Exception as e:
            logger.error(
                "auto_backfill_error",
                symbol=symbol,
                timestamp=timestamp.isoformat(),
                error=str(e),
                exc_info=True,
            )
            return False
    
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
        # Check if we need technical indicators (either explicitly or for price_ema_ratio)
        needs_technical = "technical" in self.requirements.feature_groups
        needs_ema_for_price_ratio = False
        
        # Check if price_ema_ratio or price_ema21_ratio are in the registry
        if self.feature_registry and self.feature_registry.features:
            for feature in self.feature_registry.features:
                if feature.name in ["price_ema_ratio", "price_ema21_ratio"]:
                    needs_ema_for_price_ratio = True
                    break
        
        if needs_technical or needs_ema_for_price_ratio:
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
        
        # Initialize ema_21 column if needed for price_ema_ratio but not computed above
        if needs_ema_for_price_ratio and "ema_21" not in result.columns:
            result["ema_21"] = None
            logger.debug(
                "ema_21_initialized_for_price_ema_ratio",
                needs_ema_for_price_ratio=needs_ema_for_price_ratio,
                has_technical_group="technical" in self.requirements.feature_groups,
            )
        
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
            
            # Initialize price_ema21_ratio and price_ema_ratio columns
            result["price_ema21_ratio"] = None
            result["price_ema_ratio"] = None
            
            # Check if we need to compute price_ema_ratio
            needs_price_ema_ratio = False
            if self.feature_registry and self.feature_registry.features:
                for feature in self.feature_registry.features:
                    if feature.name in ["price_ema_ratio", "price_ema21_ratio"]:
                        needs_price_ema_ratio = True
                        break
            
            # Compute price_ema21_ratio and price_ema_ratio if ema_21 is available
            # Check if ema_21 was computed (even if all values are None)
            has_ema_21_column = "ema_21" in result.columns
            if has_ema_21_column and current_prices is not None:
                # price_ema21_ratio = current_price / ema_21
                for idx in range(len(timestamps)):
                    if idx >= len(current_prices) or idx >= len(result):
                        continue
                    current_price = current_prices.iloc[idx]
                    ema_21 = result.at[idx, "ema_21"]
                    if pd.notna(current_price) and current_price > 0:
                        if pd.notna(ema_21) and ema_21 > 0:
                            result.at[idx, "price_ema21_ratio"] = float(current_price / ema_21)
                
                # price_ema_ratio with dynamic period from feature_registry
                logger.debug(
                    "computing_price_ema_ratio",
                    has_feature_registry=self.feature_registry is not None,
                    has_ema_21_column=has_ema_21_column,
                    ema_21_notna_count=result["ema_21"].notna().sum() if has_ema_21_column else 0,
                    current_prices_notna_count=current_prices.notna().sum() if current_prices is not None else 0,
                )
                
                if self.feature_registry:
                    try:
                        # Get lookback_window for price_ema_ratio from registry
                        ema_period = 21  # default
                        registry_model = getattr(self.feature_registry, "_registry_model", None)
                        if registry_model and registry_model.features:
                            for feature in registry_model.features:
                                if feature.name == "price_ema_ratio" and feature.lookback_window:
                                    # Parse lookback_window (e.g., "45m" -> 45)
                                    lookback = feature.lookback_window
                                    if lookback.endswith("m"):
                                        ema_period = int(lookback[:-1])
                                    break
                        else:
                            # Try config fallback
                            config = self.feature_registry.get_config() if hasattr(self.feature_registry, "get_config") else None
                            if config and "features" in config:
                                for feat in config["features"]:
                                    if feat.get("name") == "price_ema_ratio":
                                        lookback = feat.get("lookback_window", "21m")
                                        if lookback.endswith("m"):
                                            ema_period = int(lookback[:-1])
                                        break
                        
                        # Compute EMA with dynamic period
                        if ema_period != 21:
                            # Need to compute EMA with different period
                            technical_df_custom = compute_technical_indicators_vectorized(
                                klines_df=klines_df,
                                timestamps=timestamps,
                                period_ema=ema_period,
                                period_rsi=14,  # Not used for price_ema_ratio
                            )
                            # Note: compute_technical_indicators_vectorized returns "ema_21" column
                            # but it actually contains EMA with the specified period
                            ema_custom_col = "ema_21"  # Column name is fixed, but value is for ema_period
                            if ema_custom_col in technical_df_custom.columns:
                                # Store in a temporary column to avoid overwriting ema_21
                                temp_col = f"_ema_{ema_period}_temp"
                                result[temp_col] = technical_df_custom[ema_custom_col]
                                
                                # Compute price_ema_ratio
                                for idx in range(len(timestamps)):
                                    if idx >= len(current_prices) or idx >= len(result) or temp_col not in result.columns:
                                        continue
                                    current_price = current_prices.iloc[idx]
                                    ema_custom = result.at[idx, temp_col]
                                    if pd.notna(current_price) and current_price > 0:
                                        if pd.notna(ema_custom) and ema_custom > 0:
                                            result.at[idx, "price_ema_ratio"] = float(current_price / ema_custom)
                                # Clean up temporary column
                                result.drop(columns=[temp_col], inplace=True, errors='ignore')
                        else:
                            # Use ema_21 for price_ema_ratio if period is 21
                            result["price_ema_ratio"] = result["price_ema21_ratio"]
                    except Exception as e:
                        logger.warning(
                            "failed_to_compute_price_ema_ratio",
                            error=str(e),
                            error_type=type(e).__name__,
                            exc_info=True,
                        )
                        # Fallback: use price_ema21_ratio
                        result["price_ema_ratio"] = result.get("price_ema21_ratio")
                else:
                    # No feature_registry: use default ema_21
                    result["price_ema_ratio"] = result.get("price_ema21_ratio")
            elif needs_price_ema_ratio and current_prices is not None:
                # price_ema_ratio is in registry but ema_21 was not computed
                # Try to compute it directly with the period from registry
                if self.feature_registry:
                    try:
                        # Get lookback_window for price_ema_ratio from registry
                        ema_period = 21  # default
                        registry_model = getattr(self.feature_registry, "_registry_model", None)
                        if registry_model and registry_model.features:
                            for feature in registry_model.features:
                                if feature.name == "price_ema_ratio" and feature.lookback_window:
                                    lookback = feature.lookback_window
                                    if lookback.endswith("m"):
                                        ema_period = int(lookback[:-1])
                                    break
                        else:
                            # Try config fallback
                            config = self.feature_registry.get_config() if hasattr(self.feature_registry, "get_config") else None
                            if config and "features" in config:
                                for feat in config["features"]:
                                    if feat.get("name") == "price_ema_ratio":
                                        lookback = feat.get("lookback_window", "21m")
                                        if lookback.endswith("m"):
                                            ema_period = int(lookback[:-1])
                                        break
                        
                        # Compute EMA with dynamic period
                        technical_df_custom = compute_technical_indicators_vectorized(
                            klines_df=klines_df,
                            timestamps=timestamps,
                            period_ema=ema_period,
                            period_rsi=14,  # Not used for price_ema_ratio
                        )
                        ema_custom_col = "ema_21"  # Column name is fixed, but value is for ema_period
                        if ema_custom_col in technical_df_custom.columns:
                            # Compute price_ema_ratio directly
                            for idx in range(len(timestamps)):
                                if idx >= len(current_prices) or idx >= len(result):
                                    continue
                                current_price = current_prices.iloc[idx]
                                ema_custom = technical_df_custom.iloc[idx][ema_custom_col]
                                if pd.notna(current_price) and current_price > 0:
                                    if pd.notna(ema_custom) and ema_custom > 0:
                                        result.at[idx, "price_ema_ratio"] = float(current_price / ema_custom)
                    except Exception as e:
                        logger.warning(
                            "failed_to_compute_price_ema_ratio_no_ema21",
                            error=str(e),
                            error_type=type(e).__name__,
                            exc_info=True,
                        )
        
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
        result["dow_sin"] = None
        result["dow_cos"] = None
        result["is_asia_session"] = None
        result["is_london_session"] = None
        result["is_ny_session"] = None
        
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
        
        # Determine which function to use based on lookback_window of pattern features
        compute_patterns = self._get_candle_pattern_function()
        
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
        
        # Track daily statistics for summary logging
        current_day = None
        daily_stats = {}  # {date: {feature_name: {computed: 0, none_count: 0}}}
        daily_stats = {}  # {date: {feature_name: {computed: 0, none_count: 0}}}
        
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
            # Note: Automatic backfill should be handled at a higher level (e.g., in dataset builder)
            # This method is synchronous and cannot perform async backfill operations
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
                
                # Backfill completed successfully, but we need to reload data
                # This will be handled by the caller (streaming_builder) which will retry
                # For now, we raise a special exception to signal that backfill was done
                # and data should be reloaded
                logger.info(
                    "feature_computation_backfill_completed_retry_needed",
                    row_index=idx,
                    current_timestamp=ts_dt.isoformat(),
                    message="Backfill completed, data should be reloaded and computation retried",
                )
                raise RuntimeError(
                    f"Data backfilled for timestamp {ts_dt.isoformat()}, please retry computation. "
                    f"This is a signal to reload data, not an actual error."
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
                        
                        # Track daily statistics instead of logging every row
                        ts_date = ts_dt.date() if hasattr(ts_dt, 'date') else pd.Timestamp(ts_dt).date()
                        
                        # Initialize daily stats for new day and log summary for previous day
                        if current_day is None or ts_date != current_day:
                            # Log summary for previous day (if exists)
                            if current_day is not None and daily_stats.get(current_day):
                                for day_feature, stats in daily_stats[current_day].items():
                                    logger.info(
                                        "feature_computation_daily_summary",
                                        date=str(current_day),
                                        feature_name=day_feature,
                                        total_computed=stats.get("computed", 0),
                                        none_count=stats.get("none_count", 0),
                                        success_rate=1.0 - (stats.get("none_count", 0) / max(stats.get("computed", 1), 1)),
                                    )
                            current_day = ts_date
                            if current_day not in daily_stats:
                                daily_stats[current_day] = {}
                        
                        # Update daily statistics
                        if feature_name not in daily_stats[current_day]:
                            daily_stats[current_day][feature_name] = {"computed": 0, "none_count": 0}
                        daily_stats[current_day][feature_name]["computed"] += 1
                        if value is None:
                            daily_stats[current_day][feature_name]["none_count"] += 1
                        
                        # Log only first row (errors are already logged separately above at line 761)
                        if idx == 0:
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
        
        # Log final daily summary
        if current_day is not None and daily_stats.get(current_day):
            for day_feature, stats in daily_stats[current_day].items():
                logger.info(
                    "feature_computation_daily_summary",
                    date=str(current_day),
                    feature_name=day_feature,
                    total_computed=stats.get("computed", 0),
                    none_count=stats.get("none_count", 0),
                    success_rate=1.0 - (stats.get("none_count", 0) / max(stats.get("computed", 1), 1)),
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
    
    def _get_candle_pattern_function(self):
        """
        Определить функцию для вычисления паттернов на основе lookback_window.
        
        Returns:
            Функция для вычисления паттернов
        """
        if not self.feature_registry or not self.feature_registry.features:
            # Fallback: использовать версию на основе версии Feature Registry
            if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                return compute_all_candle_patterns_15m
            elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                return compute_all_candle_patterns_5m
            else:
                return compute_all_candle_patterns_3m
        
        # Найти все паттерны и определить их lookback_window
        pattern_lookbacks = set()
        for feature in self.feature_registry.features:
            if feature.name.startswith(("candle_", "pattern_")):
                if feature.lookback_window:
                    pattern_lookbacks.add(feature.lookback_window)
        
        if not pattern_lookbacks:
            # Fallback: использовать версию на основе версии Feature Registry
            if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                return compute_all_candle_patterns_15m
            elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                return compute_all_candle_patterns_5m
            else:
                return compute_all_candle_patterns_3m
        
        # Используем наиболее часто встречающийся lookback_window
        from collections import Counter
        counter = Counter(pattern_lookbacks)
        most_common = counter.most_common(1)[0][0]
        
        # Выбрать функцию на основе lookback_window
        if most_common == "45m":
            return compute_all_candle_patterns_45m
        elif most_common == "15m":
            return compute_all_candle_patterns_15m
        elif most_common == "5m":
            return compute_all_candle_patterns_5m
        elif most_common == "3m":
            return compute_all_candle_patterns_3m
        else:
            # Fallback: использовать версию на основе версии Feature Registry
            if self.feature_registry_version and self.feature_registry_version >= "1.5.0":
                return compute_all_candle_patterns_15m
            elif self.feature_registry_version and self.feature_registry_version >= "1.4.0":
                return compute_all_candle_patterns_5m
            else:
                return compute_all_candle_patterns_3m

