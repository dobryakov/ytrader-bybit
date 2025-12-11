"""
Offline feature engine for computing features from historical data.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
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
        self._update_allowed_features()
    
    def _update_allowed_features(self) -> None:
        """Update allowed feature names from Feature Registry."""
        if self._feature_registry_loader is None:
            self._allowed_feature_names = None
            return
        
        try:
            registry_model = self._feature_registry_loader._registry_model
            if registry_model:
                self._allowed_feature_names = {f.name for f in registry_model.features}
                logger.debug(
                    "feature_registry_features_loaded",
                    count=len(self._allowed_feature_names),
                    features=list(self._allowed_feature_names),
                )
            else:
                # Try to load config
                try:
                    config = self._feature_registry_loader.get_config()
                    if config and "features" in config:
                        self._allowed_feature_names = {
                            f.get("name") for f in config["features"] if f.get("name")
                        }
                except Exception as e:
                    logger.warning(
                        "failed_to_load_feature_registry_for_filtering",
                        error=str(e),
                    )
                    self._allowed_feature_names = None
        except Exception as e:
            logger.warning(
                "failed_to_update_allowed_features",
                error=str(e),
            )
            self._allowed_feature_names = None
    
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
    ) -> Optional[FeatureVector]:
        """
        Compute features at a specific timestamp from historical data.
        
        This method reconstructs orderbook state and rolling windows up to
        the target timestamp, then computes features identically to online mode.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp for feature computation
            orderbook_snapshots: Historical orderbook snapshots
            orderbook_deltas: Historical orderbook deltas
            trades: Historical trades
            klines: Historical klines
            ticker: Historical ticker data (optional)
            funding: Historical funding rate data (optional)
            
        Returns:
            FeatureVector at the target timestamp, or None if insufficient data
        """
        try:
            # Reconstruct orderbook state up to timestamp (if snapshots available)
            orderbook = None
            if not orderbook_snapshots.empty:
                orderbook = await self._reconstruct_orderbook_state(
                    symbol=symbol,
                    timestamp=timestamp,
                    snapshots=orderbook_snapshots,
                    deltas=orderbook_deltas,
                )
            
            # Reconstruct rolling windows up to timestamp
            rolling_windows = await self._reconstruct_rolling_windows(
                symbol=symbol,
                timestamp=timestamp,
                trades=trades,
                klines=klines,
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
                    )
                filtered_features = registry_filtered_features
            
            # Create feature vector
            return FeatureVector(
                timestamp=timestamp,
                symbol=symbol,
                features=filtered_features,
                feature_registry_version=self._feature_registry_version,
            )
        
        except Exception as e:
            logger.error(
                f"Error computing features for {symbol} at {timestamp}",
                error=str(e),
                exc_info=True,
            )
            return None
    
    async def _reconstruct_orderbook_state(
        self,
        symbol: str,
        timestamp: datetime,
        snapshots: pd.DataFrame,
        deltas: pd.DataFrame,
    ) -> Optional[OrderbookState]:
        """
        Reconstruct orderbook state at a specific timestamp.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            snapshots: Historical orderbook snapshots
            deltas: Historical orderbook deltas
            
        Returns:
            OrderbookState at timestamp, or None if insufficient data
        """
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
        
        # Create orderbook manager and apply snapshot
        orderbook_manager = OrderbookManager()
        
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
        
        orderbook_manager.apply_snapshot(snapshot_dict)
        
        # Apply deltas in sequence order (or timestamp order if no sequence)
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
            
            orderbook_manager.apply_delta(delta_dict)
        
        # Get final orderbook state
        return orderbook_manager.get_orderbook(symbol)
    
    async def _reconstruct_rolling_windows(
        self,
        symbol: str,
        timestamp: datetime,
        trades: pd.DataFrame,
        klines: pd.DataFrame,
    ) -> RollingWindows:
        """
        Reconstruct rolling windows at a specific timestamp.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            trades: Historical trades
            klines: Historical klines
            
        Returns:
            RollingWindows with data up to timestamp
        """
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
                # Convert trades to window format
                trade_data = []
                for _, trade in trades_before.iterrows():
                    trade_time = trade["timestamp"]
                    if isinstance(trade_time, str):
                        trade_time = pd.to_datetime(trade_time)
                    
                    # Only include trades within 1 minute lookback
                    if (timestamp - trade_time).total_seconds() <= 60:
                        trade_data.append({
                            "timestamp": trade_time,
                            "price": float(trade.get("price", 0)),
                            "volume": float(trade.get("quantity", 0)),
                            "side": trade.get("side", "Buy"),
                        })
                
                if trade_data:
                    trade_df = pd.DataFrame(trade_data)
                    
                    # Filter by window sizes
                    for window_name, window_seconds in [("1s", 1), ("3s", 3), ("15s", 15), ("1m", 60)]:
                        window_start = timestamp - timedelta(seconds=window_seconds)
                        window_trades = trade_df[trade_df["timestamp"] > window_start]
                        windows[window_name] = window_trades.copy()
        
        # Get klines for 1-minute window
        if not klines.empty:
            klines_before = klines[klines["timestamp"] <= timestamp].copy()
            
            if not klines_before.empty:
                # Use latest kline for 1-minute features
                latest_kline = klines_before.iloc[-1]
                kline_time = latest_kline["timestamp"]
                if isinstance(kline_time, str):
                    kline_time = pd.to_datetime(kline_time)
                
                # Add kline data to 1m window
                kline_data = pd.DataFrame([{
                    "timestamp": kline_time,
                    "price": float(latest_kline.get("close", 0)),
                    "volume": float(latest_kline.get("volume", 0)),
                    "side": "Buy",  # Kline doesn't have side
                }])
                
                # Fix FutureWarning: handle empty DataFrame before concat
                # If windows["1m"] is empty, just assign the new data
                # Otherwise, concat with explicit dtype preservation
                if windows["1m"].empty:
                    windows["1m"] = kline_data
                else:
                    # Use concat with sort=False to preserve column order and avoid FutureWarning
                    windows["1m"] = pd.concat(
                        [windows["1m"], kline_data], 
                        ignore_index=True,
                        sort=False
                    )
        
        return RollingWindows(
            symbol=symbol,
            windows=windows,
            last_update=timestamp,
        )
    
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
