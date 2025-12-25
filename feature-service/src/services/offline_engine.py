"""
Offline feature engine for computing features from historical data.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, Union
import pandas as pd
import structlog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.feature_registry import FeatureRegistryLoader

from src.models.feature_vector import FeatureVector
from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from src.services.orderbook_manager import OrderbookManager
from src.services.feature_registry import FeatureRegistryLoader
from src.features.price_features import compute_all_price_features
from src.features.orderflow_features import compute_all_orderflow_features
from src.features.orderbook_features import compute_all_orderbook_features
from src.features.perpetual_features import compute_all_perpetual_features
from src.features.temporal_features import compute_all_temporal_features
from src.features.candle_patterns import compute_all_candle_patterns_3m, compute_all_candle_patterns_5m, compute_all_candle_patterns_15m

logger = structlog.get_logger(__name__)


class OfflineEngine:
    """Offline feature computation engine for historical data."""
    
    def __init__(
        self,
        feature_registry_version: str = "1.0.0",
        feature_registry_loader: Optional["FeatureRegistryLoader"] = None,
    ):
        """
        Initialize offline engine.
        
        Args:
            feature_registry_version: Feature Registry version to use
            feature_registry_loader: Optional FeatureRegistryLoader for filtering features
        """
        self._feature_registry_version = feature_registry_version
        self._feature_registry_loader = feature_registry_loader
        
        # Cache allowed feature names from Feature Registry
        self._allowed_feature_names: Optional[set] = None
        self._allowed_features_updated = False  # Track if we've tried to update
        # Try to update, but don't fail if registry not loaded yet
        try:
            self._update_allowed_features()
        except Exception as e:
            logger.warning(
                "failed_to_update_allowed_features_on_init",
                error=str(e),
                message="Will retry on first feature computation",
            )
    
    def _update_allowed_features(self) -> None:
        """Update allowed feature names from Feature Registry."""
        if self._feature_registry_loader is None:
            logger.info(
                "feature_registry_filtering_disabled_no_loader",
                message="Feature Registry loader is None, no filtering will be applied",
            )
            self._allowed_feature_names = None
            return
        
        try:
            registry_model = self._feature_registry_loader._registry_model
            if registry_model:
                self._allowed_feature_names = {f.name for f in registry_model.features}
                logger.info(
                    "feature_registry_features_loaded",
                    count=len(self._allowed_feature_names),
                    features=list(self._allowed_feature_names),
                    version=self._feature_registry_version,
                )
            else:
                # Try to load config
                try:
                    config = self._feature_registry_loader.get_config()
                    if config and "features" in config:
                        self._allowed_feature_names = {
                            f.get("name") for f in config["features"] if f.get("name")
                        }
                        logger.info(
                            "feature_registry_features_loaded_from_config",
                            count=len(self._allowed_feature_names),
                            features=list(self._allowed_feature_names),
                            version=self._feature_registry_version,
                        )
                    else:
                        logger.warning(
                            "feature_registry_config_empty",
                            has_config=config is not None,
                            has_features_key="features" in config if config else False,
                        )
                        self._allowed_feature_names = None
                except Exception as e:
                    logger.warning(
                        "failed_to_load_feature_registry_for_filtering",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    self._allowed_feature_names = None
        except Exception as e:
            logger.warning(
                "failed_to_update_allowed_features",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            self._allowed_feature_names = None
    
    def _compute_max_lookback_minutes(self) -> int:
        """
        Compute maximum lookback period (in minutes) required by features in Feature Registry.
        
        Uses the same logic as FeatureRequirementsAnalyzer for computing max lookback.
        
        Returns:
            Maximum lookback period in minutes (default: 30 if registry not available)
        """
        # Feature lookback mapping (same as FeatureRequirementsAnalyzer)
        FEATURE_LOOKBACK_MAPPING = {
            # Technical indicators
            "ema_21": 26,  # 21 minutes + 5 minute buffer
            "rsi_14": 19,  # 14 minutes + 5 minute buffer (requires period+1)
            # Price features
            "price_ema21_ratio": 26,  # Depends on ema_21
            "volume_ratio_20": 20,  # 20 minutes lookback
            "volatility_5m": 6,  # 5 minutes + buffer
            "returns_5m": 6,  # 5 minutes + buffer
            # Orderflow features (no lookback for klines, but for trades windows)
            "signed_volume_1m": 1,  # 1 minute window
            "signed_volume_15s": 1,  # 15 seconds
            "signed_volume_3s": 1,  # 3 seconds
            # Default for features with lookback_window in registry
            "_default_buffer": 5,  # Additional buffer for safety
        }
        
        max_lookback = 0
        
        # Try to get lookback from Feature Registry if available
        if self._feature_registry_loader is not None:
            try:
                registry_model = self._feature_registry_loader._registry_model
                if registry_model and registry_model.features:
                    for feature in registry_model.features:
                        feature_name = feature.name
                        lookback_window = feature.lookback_window
                        
                        # Check if feature has specific mapping (includes implementation buffers)
                        if feature_name in FEATURE_LOOKBACK_MAPPING:
                            feature_lookback = FEATURE_LOOKBACK_MAPPING[feature_name]
                            max_lookback = max(max_lookback, feature_lookback)
                        else:
                            # Parse lookback_window from registry (e.g., "21m", "5m", "0s")
                            parsed_lookback = self._parse_lookback_window(lookback_window)
                            if parsed_lookback is not None:
                                # Add default buffer for safety
                                feature_lookback = parsed_lookback + FEATURE_LOOKBACK_MAPPING.get("_default_buffer", 5)
                                max_lookback = max(max_lookback, feature_lookback)
                    
                    logger.debug(
                        "computed_max_lookback_from_registry",
                        max_lookback_minutes=max_lookback,
                        registry_version=registry_model.version if registry_model else None,
                    )
            except Exception as e:
                logger.warning(
                    "failed_to_compute_lookback_from_registry",
                    error=str(e),
                    fallback="using_default_30_minutes",
                )
        
        # Use default if no registry or computation failed, or if computed value is too small
        # Minimum required lookback is 26 minutes (for ema_21)
        # If computed value is less than minimum, use default to ensure all features can compute
        if max_lookback == 0 or max_lookback < 26:
            default_lookback = 30  # Default fallback (covers ema_21: 26 min + buffer)
            if max_lookback > 0 and max_lookback < 26:
                logger.warning(
                    "computed_lookback_too_small",
                    computed_lookback=max_lookback,
                    minimum_required=26,
                    using_default=default_lookback,
                )
            else:
                logger.debug(
                    "using_default_lookback",
                    max_lookback_minutes=default_lookback,
                )
            max_lookback = default_lookback
        
        return max_lookback
    
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
                return value // 60  # Convert seconds to minutes
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
    
    async def compute_features_at_timestamp(
        self,
        symbol: str,
        timestamp: datetime,
        orderbook_snapshots: pd.DataFrame,
        orderbook_deltas: pd.DataFrame,
        trades: pd.DataFrame,
        klines: pd.DataFrame,
        ticker: Optional[pd.DataFrame] = None,
        funding: Optional[pd.DataFrame] = None,
        previous_orderbook_state: Optional[OrderbookState] = None,
        previous_rolling_windows: Optional[RollingWindows] = None,
        last_timestamp: Optional[datetime] = None,
        snapshot_refresh_interval: int = 3600,
        return_state: bool = False,
    ) -> Union[Optional[FeatureVector], Optional[Tuple[FeatureVector, Optional[OrderbookState], RollingWindows]]]:
        """
        Compute features at a specific timestamp from historical data.
        
        This method reconstructs orderbook state and rolling windows up to
        the target timestamp, then computes features identically to online mode.
        
        Supports incremental updates to improve performance for batch processing.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp for feature computation
            orderbook_snapshots: Historical orderbook snapshots
            orderbook_deltas: Historical orderbook deltas
            trades: Historical trades
            klines: Historical klines
            ticker: Historical ticker data (optional)
            funding: Historical funding rate data (optional)
            previous_orderbook_state: Optional previous OrderbookState for incremental update
            previous_rolling_windows: Optional previous RollingWindows for incremental update
            last_timestamp: Optional last processed timestamp for incremental update
            snapshot_refresh_interval: Snapshot refresh interval in seconds (default: 3600 = 1 hour)
            return_state: If True, return tuple (FeatureVector, OrderbookState, RollingWindows) for incremental updates
            
        Returns:
            FeatureVector at the target timestamp, or None if insufficient data.
            If return_state=True, returns tuple (FeatureVector, OrderbookState, RollingWindows).
        """
        try:
            # Ensure allowed features are updated (lazy initialization)
            if not self._allowed_features_updated or self._allowed_feature_names is None:
                try:
                    self._update_allowed_features()
                    self._allowed_features_updated = True
                except Exception as e:
                    logger.warning(
                        "failed_to_update_allowed_features_on_compute",
                        error=str(e),
                        message="Feature filtering may not work correctly",
                    )
            
            # Reconstruct orderbook state up to timestamp (if snapshots available)
            orderbook = None
            if not orderbook_snapshots.empty:
                orderbook = await self._reconstruct_orderbook_state(
                    symbol=symbol,
                    timestamp=timestamp,
                    snapshots=orderbook_snapshots,
                    deltas=orderbook_deltas,
                    previous_orderbook_state=previous_orderbook_state,
                    last_timestamp=last_timestamp,
                    snapshot_refresh_interval=snapshot_refresh_interval,
                )
            
            # Reconstruct rolling windows up to timestamp
            rolling_windows = await self._reconstruct_rolling_windows(
                symbol=symbol,
                timestamp=timestamp,
                trades=trades,
                klines=klines,
                previous_rolling_windows=previous_rolling_windows,
                last_timestamp=last_timestamp,
            )
            
            # Get current price from orderbook or klines
            current_price = orderbook.get_mid_price() if orderbook else None
            if current_price is None and not klines.empty:
                # Try to get price from klines
                klines_before = klines[klines["timestamp"] <= timestamp]
                if not klines_before.empty:
                    latest_kline = klines_before.iloc[-1]
                    current_price = latest_kline.get("close") or latest_kline.get("price")
            
            # Get funding rate if available
            funding_rate = None
            time_to_funding = None
            if funding is not None and not funding.empty:
                # Get latest funding rate before timestamp
                funding_before = funding[funding["timestamp"] <= timestamp]
                if not funding_before.empty:
                    latest_funding = funding_before.iloc[-1]
                    funding_rate = latest_funding.get("funding_rate")
                    next_funding_time = latest_funding.get("next_funding_time")
                    if next_funding_time:
                        time_to_funding = (next_funding_time / 1000) - timestamp.timestamp()
            
            # Compute all feature groups (same as online engine)
            all_features = {}
            
            # Price features
            price_features = compute_all_price_features(
                orderbook, rolling_windows, current_price
            )
            all_features.update(price_features)
            
            # Technical indicators (EMA, RSI, etc.)
            from src.features.technical_indicators import compute_all_technical_indicators
            technical_indicators = compute_all_technical_indicators(rolling_windows)
            all_features.update(technical_indicators)
            
            # Orderflow features
            orderflow_features = compute_all_orderflow_features(rolling_windows)
            all_features.update(orderflow_features)
            
            # Orderbook features
            orderbook_features = compute_all_orderbook_features(orderbook)
            all_features.update(orderbook_features)
            
            # Perpetual features
            perpetual_features = compute_all_perpetual_features(
                funding_rate, time_to_funding
            )
            all_features.update(perpetual_features)
            
            # Temporal features
            temporal_features = compute_all_temporal_features(timestamp)
            all_features.update(temporal_features)
            
            # Candle pattern features
            # Use 15m version (5-minute candles) if Feature Registry version >= 1.5.0
            # Use 5m version (1-minute candles) if Feature Registry version >= 1.4.0
            # Otherwise use 3m version
            if self._feature_registry_version and self._feature_registry_version >= "1.5.0":
                candle_pattern_features = compute_all_candle_patterns_15m(rolling_windows)
            elif self._feature_registry_version and self._feature_registry_version >= "1.4.0":
                candle_pattern_features = compute_all_candle_patterns_5m(rolling_windows)
            else:
                candle_pattern_features = compute_all_candle_patterns_3m(rolling_windows)
            all_features.update(candle_pattern_features)
            
            # Filter out None values and NaN/Inf values (not JSON compliant)
            import math
            filtered_features = {
                k: v for k, v in all_features.items()
                if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            }
            
            # Filter features by Feature Registry (if enabled)
            if self._allowed_feature_names is not None:
                registry_filtered_features = {
                    k: v for k, v in filtered_features.items()
                    if k in self._allowed_feature_names
                }
                if len(registry_filtered_features) < len(filtered_features):
                    logger.debug(
                        "features_filtered_by_registry",
                        original_count=len(filtered_features),
                        filtered_count=len(registry_filtered_features),
                        removed_features=set(filtered_features.keys()) - set(registry_filtered_features.keys()),
                        allowed_features=list(self._allowed_feature_names)[:20],  # First 20 for logging
                    )
                filtered_features = registry_filtered_features
            else:
                logger.debug(
                    "feature_registry_filtering_disabled",
                    computed_features_count=len(filtered_features),
                    computed_features=list(filtered_features.keys())[:20],  # First 20 for logging
                )
            
            # Log final feature count for debugging
            if len(filtered_features) == 0:
                logger.error(
                    "no_features_after_filtering",
                    original_features_count=len(all_features),
                    filtered_features_count=len(filtered_features),
                    allowed_feature_names=list(self._allowed_feature_names)[:20] if self._allowed_feature_names else None,
                    allowed_feature_names_count=len(self._allowed_feature_names) if self._allowed_feature_names else 0,
                    computed_feature_names=list(all_features.keys())[:20],
                    feature_registry_loader_present=self._feature_registry_loader is not None,
                    symbol=symbol,
                    timestamp=timestamp.isoformat(),
                )
            
            # Create feature vector
            feature_vector = FeatureVector(
                timestamp=timestamp,
                symbol=symbol,
                features=filtered_features,
                feature_registry_version=self._feature_registry_version,
            )
            
            # Return state if requested (for incremental updates)
            if return_state:
                return (feature_vector, orderbook, rolling_windows)
            
            return feature_vector
        
        except Exception as e:
            logger.error(
                f"Error computing features for {symbol} at {timestamp}",
                error=str(e),
                exc_info=True,
            )
            if return_state:
                return (None, None, None)
            return None
    
    async def _reconstruct_orderbook_state(
        self,
        symbol: str,
        timestamp: datetime,
        snapshots: pd.DataFrame,
        deltas: pd.DataFrame,
        previous_orderbook_state: Optional[OrderbookState] = None,
        last_timestamp: Optional[datetime] = None,
        snapshot_refresh_interval: int = 3600,
    ) -> Optional[OrderbookState]:
        """
        Reconstruct orderbook state at a specific timestamp.
        
        Supports incremental updates to avoid full reconstruction for each timestamp.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            snapshots: Historical orderbook snapshots
            deltas: Historical orderbook deltas
            previous_orderbook_state: Optional previous OrderbookState for incremental update
            last_timestamp: Optional last processed timestamp for incremental update
            snapshot_refresh_interval: Snapshot refresh interval in seconds (default: 3600 = 1 hour)
            
        Returns:
            OrderbookState at timestamp, or None if insufficient data
        """
        # If previous state provided, use incremental update
        if previous_orderbook_state is not None and last_timestamp is not None:
            return await self._reconstruct_orderbook_state_incremental(
                symbol=symbol,
                timestamp=timestamp,
                snapshots=snapshots,
                deltas=deltas,
                previous_orderbook_state=previous_orderbook_state,
                last_timestamp=last_timestamp,
                snapshot_refresh_interval=snapshot_refresh_interval,
            )
        
        # Full reconstruction (backward compatibility)
        if snapshots.empty:
            return None
        
        # Find latest snapshot before or at timestamp
        snapshots_before = snapshots[snapshots["timestamp"] <= timestamp]
        if snapshots_before.empty:
            return None
        
        # Get the latest snapshot
        latest_snapshot = snapshots_before.iloc[-1]
        snapshot_time = latest_snapshot["timestamp"]
        
        # Check if sequence column exists in both snapshots and deltas
        # Use .get() to safely access sequence from Series
        has_sequence = (
            "sequence" in snapshots.columns and 
            "sequence" in deltas.columns
        )
        
        if has_sequence:
            snapshot_sequence = latest_snapshot["sequence"]
            # Get all deltas after snapshot and before/at timestamp
            deltas_after = deltas[
                (deltas["sequence"] > snapshot_sequence) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("sequence")
        else:
            # If no sequence column, use timestamp for filtering and sorting
            deltas_after = deltas[
                (deltas["timestamp"] > snapshot_time) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("timestamp")
        
        # Create orderbook manager with batching disabled for offline processing
        # (data is already ordered, no need for batching optimization)
        orderbook_manager = OrderbookManager(enable_delta_batching=False)
        
        # Convert snapshot to dict format
        snapshot_dict = {
            "event_type": "orderbook_snapshot",
            "symbol": symbol,
            "timestamp": snapshot_time.isoformat() if isinstance(snapshot_time, datetime) else str(snapshot_time),
            "bids": latest_snapshot.get("bids", []),
            "asks": latest_snapshot.get("asks", []),
        }
        # Add sequence if available
        if has_sequence:
            snapshot_dict["sequence"] = int(snapshot_sequence)
        
        # Apply snapshot immediately (not buffered) for offline processing
        orderbook_manager.apply_snapshot(snapshot_dict, buffered=False)
        
        # Apply deltas in sequence order (or timestamp order if no sequence)
        # For historical data, we apply deltas directly to orderbook state
        # to avoid sequence gap checks (gaps are normal in historical data)
        orderbook_state = orderbook_manager.get_orderbook(symbol)
        for _, delta_row in deltas_after.iterrows():
            delta_dict = {
                "event_type": "orderbook_delta",
                "symbol": symbol,
                "timestamp": delta_row["timestamp"].isoformat() if isinstance(delta_row["timestamp"], datetime) else str(delta_row["timestamp"]),
                "delta_type": delta_row.get("delta_type", "update"),
                "side": delta_row.get("side", "bid"),
                "price": float(delta_row.get("price", 0)),
                "quantity": float(delta_row.get("quantity", 0)),
            }
            # Add sequence if available
            if has_sequence and "sequence" in delta_row:
                delta_dict["sequence"] = int(delta_row["sequence"])
            
            # For historical reconstruction, apply delta directly to orderbook state
            # This bypasses sequence gap checks which are not relevant for historical data
            if orderbook_state:
                orderbook_state.apply_delta(delta_dict)
            else:
                # Fallback to manager if state not available
                orderbook_manager.apply_delta(delta_dict)
                orderbook_state = orderbook_manager.get_orderbook(symbol)
        
        # Get final orderbook state
        return orderbook_manager.get_orderbook(symbol)
    
    async def _reconstruct_orderbook_state_incremental(
        self,
        symbol: str,
        timestamp: datetime,
        snapshots: pd.DataFrame,
        deltas: pd.DataFrame,
        previous_orderbook_state: OrderbookState,
        last_timestamp: datetime,
        snapshot_refresh_interval: int = 3600,
    ) -> Optional[OrderbookState]:
        """
        Incremental reconstruction of orderbook state.
        
        Reuses existing OrderbookState and applies only new deltas between
        last_timestamp and current timestamp. Periodically refreshes snapshot
        to prevent delta accumulation errors.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            snapshots: Historical orderbook snapshots
            deltas: Historical orderbook deltas
            previous_orderbook_state: Previous OrderbookState to reuse
            last_timestamp: Last processed timestamp
            snapshot_refresh_interval: Snapshot refresh interval in seconds
            
        Returns:
            OrderbookState with incremental updates, or None if refresh needed
        """
        # Check if snapshot refresh is needed
        time_since_last_snapshot = None
        if previous_orderbook_state.last_snapshot_at:
            time_since_last_snapshot = (timestamp - previous_orderbook_state.last_snapshot_at).total_seconds()
        
        needs_refresh = (
            previous_orderbook_state.last_snapshot_at is None or
            time_since_last_snapshot is None or
            time_since_last_snapshot >= snapshot_refresh_interval or
            previous_orderbook_state.delta_count > 1000  # Too many deltas since snapshot
        )
        
        if needs_refresh:
            logger.debug(
                "orderbook_snapshot_refresh_needed",
                symbol=symbol,
                timestamp=timestamp.isoformat(),
                last_snapshot_at=previous_orderbook_state.last_snapshot_at.isoformat() if previous_orderbook_state.last_snapshot_at else None,
                delta_count=previous_orderbook_state.delta_count,
                time_since_snapshot=time_since_last_snapshot,
            )
            # Fall back to full reconstruction
            return await self._reconstruct_orderbook_state(
                symbol=symbol,
                timestamp=timestamp,
                snapshots=snapshots,
                deltas=deltas,
            )
        
        # Reuse previous orderbook state and apply only new deltas directly
        # Create a copy to avoid mutating the original
        from copy import deepcopy
        orderbook_state = deepcopy(previous_orderbook_state)
        
        # Get only new deltas between last_timestamp and timestamp
        has_sequence = "sequence" in deltas.columns if not deltas.empty else False
        
        if has_sequence:
            last_sequence = previous_orderbook_state.sequence
            new_deltas = deltas[
                (deltas["sequence"] > last_sequence) &
                (deltas["timestamp"] > last_timestamp) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("sequence")
        else:
            new_deltas = deltas[
                (deltas["timestamp"] > last_timestamp) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("timestamp")
        
        logger.debug(
            "orderbook_incremental_update",
            symbol=symbol,
            timestamp=timestamp.isoformat(),
            last_timestamp=last_timestamp.isoformat(),
            new_deltas_count=len(new_deltas),
        )
        
        # Apply new deltas directly to orderbook state (bypassing gap checks for historical data)
        for _, delta_row in new_deltas.iterrows():
            delta_dict = {
                "event_type": "orderbook_delta",
                "symbol": symbol,
                "timestamp": delta_row["timestamp"].isoformat() if isinstance(delta_row["timestamp"], datetime) else str(delta_row["timestamp"]),
                "delta_type": delta_row.get("delta_type", "update"),
                "side": delta_row.get("side", "bid"),
                "price": float(delta_row.get("price", 0)),
                "quantity": float(delta_row.get("quantity", 0)),
            }
            if has_sequence and "sequence" in delta_row:
                delta_dict["sequence"] = int(delta_row["sequence"])
            # Apply directly to orderbook state (bypassing gap checks for historical data)
            orderbook_state.apply_delta(delta_dict)
        
        return orderbook_state
    
    @staticmethod
    def get_orderbook_snapshot_refresh_interval() -> int:
        """
        Get recommended orderbook snapshot refresh interval.
        
        Returns:
            Recommended refresh interval in seconds (default: 3600 = 1 hour)
        """
        return 3600
    
    async def _reconstruct_rolling_windows(
        self,
        symbol: str,
        timestamp: datetime,
        trades: pd.DataFrame,
        klines: pd.DataFrame,
        previous_rolling_windows: Optional[RollingWindows] = None,
        last_timestamp: Optional[datetime] = None,
    ) -> RollingWindows:
        """
        Reconstruct rolling windows at a specific timestamp.
        
        OPTIMIZED: Uses vectorized pandas operations instead of row-by-row loops.
        Supports incremental updates to avoid O(n²) complexity.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            trades: Historical trades
            klines: Historical klines
            previous_rolling_windows: Optional previous RollingWindows object for incremental update
            last_timestamp: Optional last processed timestamp for incremental update
            
        Returns:
            RollingWindows with data up to timestamp
        """
        # If previous state provided, use incremental update
        if previous_rolling_windows is not None and last_timestamp is not None:
            return await self._reconstruct_rolling_windows_incremental(
                symbol=symbol,
                timestamp=timestamp,
                trades=trades,
                klines=klines,
                previous_rolling_windows=previous_rolling_windows,
                last_timestamp=last_timestamp,
            )
        
        # Full reconstruction (backward compatibility)
        # Initialize empty windows
        windows = {
            "1s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "3s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "15s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "1m": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
        }
        
        # Get trades within lookback windows
        if not trades.empty:
            trades_before = trades[trades["timestamp"] <= timestamp].copy()
            
            if not trades_before.empty:
                # OPTIMIZED: Use vectorized operations instead of iterrows()
                # Filter trades within 1 minute lookback using boolean indexing
                trade_time_diff = (timestamp - pd.to_datetime(trades_before["timestamp"])).dt.total_seconds()
                trades_within_window = trades_before[trade_time_diff <= 60].copy()
                
                if not trades_within_window.empty:
                    # Convert to proper format using vectorized operations
                    trade_df = pd.DataFrame({
                        "timestamp": pd.to_datetime(trades_within_window["timestamp"], utc=True),
                        "price": pd.to_numeric(trades_within_window.get("price", 0), errors='coerce').fillna(0.0),
                        "volume": pd.to_numeric(trades_within_window.get("quantity", trades_within_window.get("volume", 0)), errors='coerce').fillna(0.0),
                        "side": trades_within_window.get("side", "Buy"),
                    })
                    
                    # Filter by window sizes using vectorized operations
                    for window_name, window_seconds in [("1s", 1), ("3s", 3), ("15s", 15), ("1m", 60)]:
                        window_start = timestamp - timedelta(seconds=window_seconds)
                        window_trades = trade_df[trade_df["timestamp"] > window_start].copy()
                        windows[window_name] = window_trades
        
        # OPTIMIZED: Process klines using vectorized operations
        # CRITICAL PERFORMANCE FIX: Avoid O(n²) complexity from iterrows() + concat + sort_values
        if not klines.empty:
            # Pre-filter klines once using vectorized boolean indexing
            klines_before = klines[klines["timestamp"] <= timestamp].copy()
            
            if not klines_before.empty:
                # OPTIMIZED: Use vectorized operations instead of iterrows() loop
                # Convert timestamps to datetime in one operation
                kline_timestamps = pd.to_datetime(klines_before["timestamp"], utc=True)
                
                # Ensure timezone-aware
                if kline_timestamps.dt.tz is None:
                    kline_timestamps = kline_timestamps.dt.tz_localize(timezone.utc)
                
                # Create kline DataFrame using vectorized operations
                kline_data = pd.DataFrame({
                    "timestamp": kline_timestamps,
                    "open": pd.to_numeric(klines_before.get("open", 0), errors='coerce').fillna(0.0),
                    "high": pd.to_numeric(klines_before.get("high", 0), errors='coerce').fillna(0.0),
                    "low": pd.to_numeric(klines_before.get("low", 0), errors='coerce').fillna(0.0),
                    "close": pd.to_numeric(klines_before.get("close", 0), errors='coerce').fillna(0.0),
                    "volume": pd.to_numeric(klines_before.get("volume", 0), errors='coerce').fillna(0.0),
                })
                
                # OPTIMIZED: Sort once instead of per-timestamp
                kline_data = kline_data.sort_values("timestamp").reset_index(drop=True)
                
                # Assign directly (no concat needed for full reconstruction)
                windows["1m"] = kline_data
        
        return RollingWindows(
            symbol=symbol,
            windows=windows,
            last_update=timestamp,
        )
    
    async def _reconstruct_rolling_windows_incremental(
        self,
        symbol: str,
        timestamp: datetime,
        trades: pd.DataFrame,
        klines: pd.DataFrame,
        previous_rolling_windows: RollingWindows,
        last_timestamp: datetime,
    ) -> RollingWindows:
        """
        Incremental reconstruction of rolling windows.
        
        Reuses existing RollingWindows object and adds only new trades/klines
        between last_timestamp and current timestamp.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            trades: Historical trades
            klines: Historical klines
            previous_rolling_windows: Previous RollingWindows object to reuse
            last_timestamp: Last processed timestamp
            
        Returns:
            RollingWindows with incremental updates
        """
        # Reuse existing RollingWindows object
        rolling_windows = previous_rolling_windows
        
        # Add new trades between last_timestamp and timestamp
        if not trades.empty:
            new_trades = trades[
                (trades["timestamp"] > last_timestamp) & 
                (trades["timestamp"] <= timestamp)
            ].copy()
            
            if not new_trades.empty:
                # Add trades using RollingWindows.add_trade() method
                for _, trade in new_trades.iterrows():
                    trade_dict = {
                        "timestamp": trade["timestamp"],
                        "price": trade.get("price", 0),
                        "quantity": trade.get("quantity", trade.get("volume", 0)),
                        "side": trade.get("side", "Buy"),
                    }
                    rolling_windows.add_trade(trade_dict)
        
        # Add new klines between last_timestamp and timestamp
        if not klines.empty:
            new_klines = klines[
                (klines["timestamp"] > last_timestamp) & 
                (klines["timestamp"] <= timestamp)
            ].copy()
            
            if not new_klines.empty:
                # Add klines using RollingWindows.add_kline() method
                for _, kline in new_klines.iterrows():
                    kline_dict = {
                        "timestamp": kline["timestamp"],
                        "open": kline.get("open", 0),
                        "high": kline.get("high", 0),
                        "low": kline.get("low", 0),
                        "close": kline.get("close", 0),
                        "volume": kline.get("volume", 0),
                    }
                    rolling_windows.add_kline(kline_dict)
        
        # Update last_update timestamp
        rolling_windows.last_update = timestamp
        
        # Trim old data (automatic via add_trade/add_kline, but ensure it's done)
        # Use computed max_lookback from Feature Registry to ensure all features have sufficient data
        max_lookback_minutes = self._compute_max_lookback_minutes()
        rolling_windows.trim_old_data(max_lookback_minutes_1m=max_lookback_minutes)
        
        return rolling_windows
    
    async def validate_feature_identity(
        self,
        online_features: FeatureVector,
        offline_features: FeatureVector,
        tolerance: float = 1e-6,
    ) -> bool:
        """
        Validate that online and offline features are identical.
        
        Args:
            online_features: Features computed in online mode
            offline_features: Features computed in offline mode
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if features are identical within tolerance
        """
        if online_features.symbol != offline_features.symbol:
            return False
        
        if abs((online_features.timestamp - offline_features.timestamp).total_seconds()) > 1:
            return False
        
        # Compare all feature values
        for feature_name in online_features.features:
            if feature_name not in offline_features.features:
                logger.warning(f"Feature {feature_name} missing in offline features")
                return False
            
            online_value = online_features.features[feature_name]
            offline_value = offline_features.features[feature_name]
            
            # Handle NaN values
            import math
            if math.isnan(online_value) and math.isnan(offline_value):
                continue
            
            if abs(online_value - offline_value) > tolerance:
                logger.warning(
                    f"Feature {feature_name} mismatch: "
                    f"online={online_value}, offline={offline_value}"
                )
                return False
        
        return True
