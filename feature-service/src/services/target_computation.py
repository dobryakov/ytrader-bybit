"""
Target computation presets and logic.
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from src.models.dataset import TargetComputationConfig, TargetComputationOverrides
from src.logging import get_logger

logger = get_logger(__name__)


def _get_series_summary(series: pd.Series, max_sample: int = 5) -> Dict[str, Any]:
    """
    Generate compact summary statistics for a pandas Series to reduce log volume.
    
    Args:
        series: pandas Series to summarize
        max_sample: Maximum number of values to include in head/tail samples
        
    Returns:
        Dictionary with summary statistics and samples
    """
    if series is None or len(series) == 0:
        return {}
    
    summary = {
        "count": len(series),
        "notna_count": int(series.notna().sum()),
        "na_count": int(series.isna().sum()),
    }
    
    notna_series = series.dropna()
    if len(notna_series) > 0:
        summary.update({
            "min": float(notna_series.min()),
            "max": float(notna_series.max()),
            "mean": float(notna_series.mean()),
            "std": float(notna_series.std()),
            "sample_first": notna_series.head(max_sample).tolist(),
            "sample_last": notna_series.tail(max_sample).tolist(),
        })
    
    return summary


class TargetComputationPresets:
    """Preset configurations for target computation."""
    
    PRESETS: Dict[str, Dict[str, Any]] = {
        "returns": {
            "formula": "returns",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "description": "Expected return over fixed time: (future_price - price) / price",
        },
        "log_returns": {
            "formula": "log_returns",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "description": "Logarithmic returns: log(future_price / price)",
        },
        "sharpe_ratio": {
            "formula": "sharpe_ratio",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "volatility_window": 20,
            "description": "Risk-adjusted returns: returns / volatility",
        },
        "price_change": {
            "formula": "price_change",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "description": "Absolute price change: future_price - price",
        },
        "volatility_normalized_std": {
            "formula": "volatility_normalized_std",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "volatility_window": 20,
            "description": "Volatility-normalized returns: raw_return / rolling_std(raw_return, window=X)",
        },
        # Direction of the next candle (binary or multi-class classification)
        # Base computation is still returns; mapping to classes is done in DatasetBuilder/ModelTrainer.
        "next_candle_direction": {
            "formula": "returns",
            "price_source": "close",
            "future_price_source": "close",
            "lookup_method": "nearest_forward",
            "description": "Direction of next candle based on forward return sign; can be used for binary or 3-class targets.",
        },
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get preset configuration.
        
        Args:
            preset_name: Preset name
            
        Returns:
            Preset configuration dict
            
        Raises:
            ValueError: If preset not found
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {list(cls.PRESETS.keys())}"
            )
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def get_computation_config(
        cls,
        computation: Optional[TargetComputationConfig],
    ) -> Dict[str, Any]:
        """
        Get final computation configuration with overrides applied.
        
        Args:
            computation: Target computation configuration (optional)
            
        Returns:
            Final computation configuration dict
        """
        # Default to "returns" preset if not specified
        preset_name = "returns"
        overrides = None
        
        if computation:
            preset_name = computation.preset
            overrides = computation.overrides
        
        # Get preset defaults
        config = cls.get_preset(preset_name)
        
        # Apply overrides if provided
        if overrides:
            if overrides.price_source:
                config["price_source"] = overrides.price_source
            if overrides.future_price_source:
                config["future_price_source"] = overrides.future_price_source
            if overrides.lookup_method:
                config["lookup_method"] = overrides.lookup_method
            if overrides.tolerance_seconds is not None:
                config["tolerance_seconds"] = overrides.tolerance_seconds
        
        # Apply additional options if provided
        if computation and computation.options:
            config.update(computation.options)
        
        return config


class TargetComputationEngine:
    """Engine for computing target values based on configuration."""
    
    @staticmethod
    def compute_target(
        data: pd.DataFrame,
        horizon: int,
        computation_config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute target values based on computation configuration.
        
        Args:
            data: DataFrame with timestamp and price columns
            horizon: Prediction horizon in seconds
            computation_config: Computation configuration from preset
            historical_price_data: Optional historical price data for future price lookup
                                   (if None, uses data itself)
            
        Returns:
            DataFrame with timestamp and target columns
        """
        formula = computation_config.get("formula", "returns")
        
        if formula == "returns":
            result = TargetComputationEngine._compute_returns(
                data, horizon, computation_config, historical_price_data
            )
            return TargetComputationEngine._postprocess_target(result, computation_config)
        elif formula == "log_returns":
            result = TargetComputationEngine._compute_log_returns(
                data, horizon, computation_config, historical_price_data
            )
            return TargetComputationEngine._postprocess_target(result, computation_config)
        elif formula == "sharpe_ratio":
            # Sharpe уже нормализует таргет, дополнительная нормализация по умолчанию не нужна
            return TargetComputationEngine._compute_sharpe_ratio(
                data, horizon, computation_config, historical_price_data
            )
        elif formula == "price_change":
            result = TargetComputationEngine._compute_price_change(
                data, horizon, computation_config, historical_price_data
            )
            return TargetComputationEngine._postprocess_target(result, computation_config)
        elif formula == "volatility_normalized_std":
            # Здесь тоже уже встроена нормализация
            return TargetComputationEngine._compute_volatility_normalized_std(
                data, horizon, computation_config, historical_price_data
            )
        else:
            raise ValueError(f"Unknown formula: {formula}")
    
    @staticmethod
    def _compute_returns(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute returns: (future_price - price) / price"""
        return TargetComputationEngine._compute_base_target(
            data, horizon, config, formula="returns", historical_price_data=historical_price_data
        )
    
    @staticmethod
    def _compute_log_returns(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute log returns: log(future_price / price)"""
        result = TargetComputationEngine._compute_base_target(
            data, horizon, config, formula="returns", historical_price_data=historical_price_data
        )
        if not result.empty and "target" in result.columns:
            # Convert returns to log returns: log(1 + return) ≈ log(future_price / price)
            result["target"] = np.log1p(result["target"])
        return result
    
    @staticmethod
    def _compute_price_change(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute absolute price change: future_price - price"""
        result = TargetComputationEngine._compute_base_target(
            data, horizon, config, formula="price_change", historical_price_data=historical_price_data
        )
        return result
    
    @staticmethod
    def _compute_sharpe_ratio(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute Sharpe ratio: returns / volatility"""
        # First compute returns
        returns_df = TargetComputationEngine._compute_returns(
            data, horizon, config, historical_price_data
        )
        
        if returns_df.empty:
            return pd.DataFrame(columns=["timestamp", "target"])
        
        # Compute rolling volatility
        volatility_window = config.get("volatility_window", 20)
        returns_df["volatility"] = returns_df["target"].rolling(window=volatility_window).std()
        
        # Compute Sharpe ratio: return / volatility
        returns_df["target"] = returns_df["target"] / returns_df["volatility"]
        returns_df = returns_df.dropna(subset=["target"])
        
        return returns_df[["timestamp", "target"]]
    
    @staticmethod
    def _compute_volatility_normalized_std(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute volatility-normalized returns: raw_return / rolling_std(raw_return, window=X).
        
        This normalizes the target by dividing raw returns by their rolling standard deviation,
        making the target more stable across different market volatility regimes.
        
        Formula: y = ((future - current) / current) / rolling_std(returns, window=X)
        
        Args:
            data: DataFrame with timestamp and price columns
            horizon: Prediction horizon in seconds
            config: Computation configuration
            historical_price_data: Optional historical price data for future price lookup
            
        Returns:
            DataFrame with timestamp and target columns
        """
        # First compute raw returns
        returns_df = TargetComputationEngine._compute_returns(
            data, horizon, config, historical_price_data
        )
        
        if returns_df.empty:
            return pd.DataFrame(columns=["timestamp", "target"])
        
        # Get volatility window from config
        volatility_window = config.get("volatility_window", 20)
        
        # Compute rolling standard deviation of returns
        # This measures the volatility of returns over the rolling window
        returns_df["volatility"] = returns_df["target"].rolling(window=volatility_window).std()
        
        # Normalize returns by volatility: raw_return / rolling_std(raw_return)
        # This gives us returns in units of standard deviations
        returns_df["target"] = returns_df["target"] / returns_df["volatility"]
        
        # Remove rows where volatility is zero or NaN (can't normalize)
        returns_df = returns_df.dropna(subset=["target", "volatility"])
        returns_df = returns_df[returns_df["volatility"] > 0]
        
        # Remove infinite values
        returns_df = returns_df[
            ~returns_df["target"].isin([float("inf"), float("-inf")])
        ]
        
        return returns_df[["timestamp", "target"]]

    @staticmethod
    def _postprocess_target(
        result: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Apply optional clipping / normalization to numeric target according to config.
        
        This is intentionally generic and driven entirely from target registry options.
        If no relevant options are provided, the result is returned as-is.
        """
        if result is None or result.empty or "target" not in result.columns:
            return result

        target = result["target"].astype("float64")

        clip_method = config.get("clip_method")
        normalize = config.get("normalize")

        # --- Clipping ---
        if clip_method in {"quantile", "fixed"}:
            # Work on a copy to avoid pandas SettingWithCopy warnings
            s = target.copy()

            if clip_method == "quantile":
                q_low = config.get("clip_q_low")
                q_high = config.get("clip_q_high")
                # Require both quantiles and sane ordering
                # Also need at least 2 values to compute meaningful quantiles
                if q_low is not None and q_high is not None and 0.0 <= q_low < q_high <= 1.0 and len(s) > 1:
                    try:
                        lo = float(s.quantile(q_low))
                        hi = float(s.quantile(q_high))
                        s = s.clip(lower=lo, upper=hi)
                    except Exception:
                        # On any numeric issue just fall back to un-clipped target
                        pass
                elif len(s) <= 1:
                    # Not enough data for quantile clipping, skip it
                    logger.debug(
                        "_postprocess_target_quantile_skipped",
                        data_length=len(s),
                        message="Not enough data for quantile clipping, returning un-clipped target",
                    )
            elif clip_method == "fixed":
                max_abs = config.get("clip_abs_max")
                if max_abs is not None:
                    try:
                        m = float(max_abs)
                        if m > 0:
                            s = s.clip(lower=-m, upper=m)
                    except Exception:
                        pass

            target = s

        # --- Normalization ---
        if normalize in {"sharpe", "log", "zscore"}:
            s = target.copy()

            if normalize == "sharpe":
                # Rolling std of the series itself (like returns/std)
                window = int(config.get("sharpe_window", 20))
                if window > 1 and len(s) >= window:
                    # Only apply sharpe normalization if we have enough data
                    vol = s.rolling(window=window, min_periods=1).std()
                    # Avoid division by zero
                    vol = vol.replace(0.0, np.nan)
                    s = s / vol
                elif len(s) < window:
                    # If we don't have enough data for rolling window, skip normalization
                    logger.debug(
                        "_postprocess_target_sharpe_skipped",
                        data_length=len(s),
                        required_window=window,
                        message="Not enough data for sharpe normalization, returning raw target",
                    )
                    # Keep original target values without normalization
                    s = target.copy()
            elif normalize == "log":
                # Symmetric log transform: sign(x) * log1p(|x|)
                s = np.sign(s) * np.log1p(np.abs(s))
            elif normalize == "zscore":
                if len(s) > 1:
                    mu = float(s.mean())
                    sigma = float(s.std())
                    if sigma > 0:
                        s = (s - mu) / sigma
                else:
                    # Single value: zscore would be 0 or undefined, keep original
                    logger.debug(
                        "_postprocess_target_zscore_skipped",
                        data_length=len(s),
                        message="Not enough data for zscore normalization, returning raw target",
                    )
                    s = target.copy()

            target = s

        # Replace infinities and drop rows where target is not usable
        target = target.replace([np.inf, -np.inf], np.nan)
        result = result.copy()
        result["target"] = target
        
        target_summary_before = {}
        if "target" in result.columns:
            target_summary_before = _get_series_summary(result["target"])
        
        logger.info(
            "_postprocess_target_before_dropna",
            rows_before=len(result),
            target_summary=target_summary_before,
        )
        
        result = result.dropna(subset=["target"])
        
        target_summary_after = {}
        if "target" in result.columns and len(result) > 0:
            target_summary_after = _get_series_summary(result["target"])
        
        logger.info(
            "_postprocess_target_after_dropna",
            rows_after=len(result),
            target_summary=target_summary_after,
        )

        return result
    
    @staticmethod
    def _compute_base_target(
        data: pd.DataFrame,
        horizon: int,
        config: Dict[str, Any],
        formula: str = "returns",
        historical_price_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Base computation for target values.
        
        Args:
            data: DataFrame with timestamp and price columns
            horizon: Prediction horizon in seconds
            config: Computation configuration
            formula: Formula type ("returns" or "price_change")
            
        Returns:
            DataFrame with timestamp and target columns
        """
        if data.empty:
            logger.warning("_compute_base_target: input data is empty")
            return pd.DataFrame()
        
        # Get price source
        price_source = config.get("price_source", "close")
        future_price_source = config.get("future_price_source", "close")
        lookup_method = config.get("lookup_method", "nearest_forward")
        tolerance_seconds = config.get("tolerance_seconds")
        
        # Determine price column name
        # Note: data may already have "price" column (from merge in _compute_targets)
        # or may have original klines columns (close, open, etc.)
        price_col = None
        if "price" in data.columns:
            # Data already has "price" column (from merge)
            price_col = "price"
        elif price_source in data.columns:
            # Use requested source if available
            price_col = price_source
        elif "close" in data.columns:
            # Fallback to close
            price_col = "close"
        else:
            logger.error(
                "_compute_base_target: no price column found",
                available_columns=list(data.columns),
                requested_source=price_source,
            )
            return pd.DataFrame()
        
        if price_col not in data.columns:
            logger.error(
                "_compute_base_target: price column not found",
                requested=price_col,
                available=list(data.columns),
            )
            return pd.DataFrame()
        
        # Sort by timestamp
        data = data.sort_values("timestamp").copy()
        
        # Create future timestamps
        data["future_timestamp"] = data["timestamp"] + pd.Timedelta(seconds=horizon)
        
        # Save original price column
        original_price_df = data[["timestamp", price_col]].copy()
        original_price_df = original_price_df.rename(columns={price_col: "price"})
        
        # Create price lookup DataFrame
        # Use historical_price_data if provided (for finding future prices),
        # otherwise use data itself
        if historical_price_data is not None and not historical_price_data.empty:
            # Use historical data for future price lookup
            future_price_source = config.get("future_price_source", "close")
            if future_price_source in historical_price_data.columns:
                lookup_col = future_price_source
            elif "close" in historical_price_data.columns:
                lookup_col = "close"
            elif "price" in historical_price_data.columns:
                lookup_col = "price"
            else:
                logger.warning(
                    "_compute_base_target: future_price_source not found in historical data, using data itself",
                    available_columns=list(historical_price_data.columns),
                )
                lookup_col = price_col
                historical_price_data = data
            
            price_lookup = historical_price_data[["timestamp", lookup_col]].copy()
            price_lookup = price_lookup[price_lookup[lookup_col].notna()].copy()
            price_lookup = price_lookup.rename(columns={lookup_col: "future_price"})
        else:
            # Use data itself for future price lookup
            price_lookup = data[["timestamp", price_col]].copy()
            price_lookup = price_lookup[price_lookup[price_col].notna()].copy()
            price_lookup = price_lookup.rename(columns={price_col: "future_price"})
        
        price_lookup = price_lookup.sort_values("timestamp").reset_index(drop=True)
        
        if price_lookup.empty:
            logger.warning("_compute_base_target: price_lookup is empty")
            return pd.DataFrame()
        
        # Sort data by future_timestamp for merge_asof
        data_sorted = data.sort_values("future_timestamp").copy().reset_index(drop=True)
        
        # Determine merge direction
        direction_map = {
            "nearest_forward": "forward",
            "nearest_backward": "backward",
            "nearest": "nearest",
            "exact": "forward",  # For exact, we'll filter after merge
        }
        direction = direction_map.get(lookup_method, "forward")
        
        # Log before merge for debugging
        logger.info(
            "_compute_base_target_before_merge",
            data_rows=len(data_sorted),
            price_lookup_rows=len(price_lookup),
            data_future_timestamp_min=data_sorted["future_timestamp"].min().isoformat() if not data_sorted.empty else None,
            data_future_timestamp_max=data_sorted["future_timestamp"].max().isoformat() if not data_sorted.empty else None,
            price_lookup_timestamp_min=price_lookup["timestamp"].min().isoformat() if not price_lookup.empty else None,
            price_lookup_timestamp_max=price_lookup["timestamp"].max().isoformat() if not price_lookup.empty else None,
            lookup_method=lookup_method,
            direction=direction,
            horizon_seconds=horizon,
        )
        
        # Use merge_asof to find future price
        # merge_asof requires both DataFrames to be sorted by the merge key
        try:
            data_merged = pd.merge_asof(
                data_sorted,
                price_lookup,
                left_on="future_timestamp",
                right_on="timestamp",
                direction=direction,
                suffixes=("", "_future"),
            )
        except Exception as e:
            logger.error(
                "_compute_base_target_merge_asof_error",
                error=str(e),
                error_type=type(e).__name__,
                data_future_timestamp_dtype=str(data_sorted["future_timestamp"].dtype),
                price_lookup_timestamp_dtype=str(price_lookup["timestamp"].dtype),
                exc_info=True,
            )
            raise
        
        # Log after merge for debugging
        future_price_summary = {}
        if not data_merged.empty and "future_price" in data_merged.columns:
            future_price_summary = _get_series_summary(data_merged["future_price"])
        
        logger.info(
            "_compute_base_target_after_merge",
            data_merged_rows=len(data_merged),
            data_merged_columns=list(data_merged.columns) if not data_merged.empty else [],
            has_future_price="future_price" in data_merged.columns if not data_merged.empty else False,
            future_price_summary=future_price_summary,
        )
        
        # Apply tolerance if specified
        if tolerance_seconds is not None and not data_merged.empty and "timestamp_future" in data_merged.columns:
            time_diff = (data_merged["timestamp_future"] - data_merged["future_timestamp"]).abs()
            before_tolerance = len(data_merged)
            data_merged = data_merged[time_diff <= pd.Timedelta(seconds=tolerance_seconds)]
            after_tolerance = len(data_merged)
            if before_tolerance != after_tolerance:
                logger.debug(
                    "_compute_base_target_tolerance_applied",
                    before_tolerance=before_tolerance,
                    after_tolerance=after_tolerance,
                    tolerance_seconds=tolerance_seconds,
                )
        
        # Restore original price column
        data_merged = data_merged.merge(
            original_price_df,
            on="timestamp",
            how="left",
            suffixes=("", "_original"),
        )
        
        if "price_original" in data_merged.columns:
            data_merged["price"] = data_merged["price_original"]
            data_merged = data_merged.drop(columns=["price_original"], errors="ignore")
        
        # Drop helper columns
        data_merged = data_merged.drop(
            columns=["future_timestamp", "timestamp_future"],
            errors="ignore"
        )
        
        # Log after price restoration
        future_price_summary = {}
        price_summary = {}
        if "future_price" in data_merged.columns:
            future_price_summary = _get_series_summary(data_merged["future_price"])
        if "price" in data_merged.columns:
            price_summary = _get_series_summary(data_merged["price"])
        
        logger.info(
            "_compute_base_target_after_price_restore",
            data_merged_rows=len(data_merged),
            has_future_price="future_price" in data_merged.columns,
            has_price="price" in data_merged.columns,
            future_price_summary=future_price_summary,
            price_summary=price_summary,
        )
        
        # Check for valid prices
        if "future_price" not in data_merged.columns:
            logger.error(
                "_compute_base_target: future_price column not found after merge",
                merged_columns=list(data_merged.columns),
                data_rows=len(data),
                price_lookup_rows=len(price_lookup),
            )
            return pd.DataFrame()
        
        valid_rows = (
            data_merged["future_price"].notna() & 
            data_merged["price"].notna()
        )
        
        if valid_rows.sum() == 0:
            future_price_summary = {}
            price_summary = {}
            if "future_price" in data_merged.columns:
                future_price_summary = _get_series_summary(data_merged["future_price"])
            if "price" in data_merged.columns:
                price_summary = _get_series_summary(data_merged["price"])
            
            logger.warning(
                "_compute_base_target: no rows with both prices valid",
                data_merged_rows=len(data_merged),
                future_price_notna=data_merged["future_price"].notna().sum() if "future_price" in data_merged.columns else 0,
                price_notna=data_merged["price"].notna().sum() if "price" in data_merged.columns else 0,
                future_price_summary=future_price_summary,
                price_summary=price_summary,
                data_timestamp_min=data["timestamp"].min() if not data.empty else None,
                data_timestamp_max=data["timestamp"].max() if not data.empty else None,
                future_timestamp_min=data["future_timestamp"].min() if "future_timestamp" in data.columns and not data.empty else None,
                future_timestamp_max=data["future_timestamp"].max() if "future_timestamp" in data.columns and not data.empty else None,
                price_lookup_timestamp_min=price_lookup["timestamp"].min() if not price_lookup.empty else None,
                price_lookup_timestamp_max=price_lookup["timestamp"].max() if not price_lookup.empty else None,
                horizon=horizon,
            )
            return pd.DataFrame()
        
        # Compute target based on formula
        if formula == "returns":
            data_merged["target"] = (
                (data_merged["future_price"] - data_merged["price"]) / 
                data_merged["price"]
            )
        elif formula == "price_change":
            data_merged["target"] = (
                data_merged["future_price"] - data_merged["price"]
            )
        else:
            raise ValueError(f"Unknown formula: {formula}")
        
        # Log after target computation
        target_summary_after_computation = {}
        if "target" in data_merged.columns:
            target_summary_after_computation = _get_series_summary(data_merged["target"])
        
        logger.info(
            "_compute_base_target_after_computation",
            target_summary=target_summary_after_computation,
            rows_before_dropna=len(data_merged),
        )
        
        # Remove invalid values
        data_merged = data_merged.dropna(subset=["target"])
        data_merged = data_merged[
            ~data_merged["target"].isin([float("inf"), float("-inf")])
        ]
        
        # Prepare summary statistics instead of full list to reduce log volume
        target_summary = {}
        if "target" in data_merged.columns and len(data_merged) > 0:
            target_summary = _get_series_summary(data_merged["target"])
        
        logger.info(
            "_compute_base_target_final",
            rows_after_dropna=len(data_merged),
            target_summary=target_summary,
        )
        
        return data_merged[["timestamp", "target"]]

