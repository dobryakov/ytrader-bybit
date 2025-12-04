"""
Feature Computer service for orchestrating feature computations.
"""
from datetime import datetime, timezone
from typing import Dict, Optional
import structlog
import time

from src.models.feature_vector import FeatureVector
from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from src.services.orderbook_manager import OrderbookManager
from src.features.price_features import compute_all_price_features
from src.features.orderflow_features import compute_all_orderflow_features
from src.features.orderbook_features import compute_all_orderbook_features
from src.features.perpetual_features import compute_all_perpetual_features
from src.features.temporal_features import compute_all_temporal_features

logger = structlog.get_logger(__name__)


class FeatureComputer:
    """Orchestrates feature computation from market data."""
    
    def __init__(
        self,
        orderbook_manager: OrderbookManager,
        feature_registry_version: str = "1.0.0",
    ):
        """Initialize feature computer."""
        self._orderbook_manager = orderbook_manager
        self._rolling_windows: Dict[str, RollingWindows] = {}
        self._feature_registry_version = feature_registry_version
        self._latest_funding_rate: Dict[str, Optional[float]] = {}
        self._latest_next_funding_time: Dict[str, Optional[int]] = {}
        self._latency_threshold_ms = 70.0
        # Store last computed features per symbol for resilience (T078)
        self._last_features: Dict[str, FeatureVector] = {}
    
    def get_rolling_windows(self, symbol: str) -> RollingWindows:
        """Get or create rolling windows for symbol."""
        if symbol not in self._rolling_windows:
            from datetime import datetime, timezone
            import pandas as pd
            
            self._rolling_windows[symbol] = RollingWindows(
                symbol=symbol,
                windows={
                    "1s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
                    "3s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
                    "15s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
                    "1m": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
                },
                last_update=datetime.now(timezone.utc),
            )
        
        return self._rolling_windows[symbol]
    
    def update_funding_rate(self, symbol: str, funding_rate: float, next_funding_time: int) -> None:
        """Update funding rate for symbol."""
        self._latest_funding_rate[symbol] = funding_rate
        self._latest_next_funding_time[symbol] = next_funding_time
    
    def compute_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[FeatureVector]:
        """Compute all features for symbol at timestamp."""
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        try:
            # Get orderbook state
            orderbook = self._orderbook_manager.get_orderbook(symbol)
            
            # Get rolling windows
            rolling_windows = self.get_rolling_windows(symbol)
            
            # Get current price
            current_price = orderbook.get_mid_price() if orderbook else None
            
            # Compute all feature groups
            all_features = {}
            
            # Price features
            price_features = compute_all_price_features(orderbook, rolling_windows, current_price)
            all_features.update(price_features)
            
            # Orderflow features
            orderflow_features = compute_all_orderflow_features(rolling_windows)
            all_features.update(orderflow_features)
            
            # Orderbook features
            orderbook_features = compute_all_orderbook_features(orderbook)
            all_features.update(orderbook_features)
            
            # Perpetual features
            funding_rate = self._latest_funding_rate.get(symbol)
            next_funding_time = self._latest_next_funding_time.get(symbol)
            perpetual_features = compute_all_perpetual_features(
                rolling_windows,
                funding_rate=funding_rate,
                next_funding_time=next_funding_time,
            )
            all_features.update(perpetual_features)
            
            # Temporal features
            temporal_features = compute_all_temporal_features(timestamp)
            all_features.update(temporal_features)
            
            # Filter out None values
            filtered_features = {k: v for k, v in all_features.items() if v is not None}
            
            # Create feature vector
            feature_vector = FeatureVector(
                timestamp=timestamp,
                symbol=symbol,
                features=filtered_features,
                feature_registry_version=self._feature_registry_version,
                trace_id=trace_id,
            )
            
            # Compute latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Add latency monitoring and metrics (T077)
            # Log latency warning if exceeds threshold (T079)
            if latency_ms > self._latency_threshold_ms:
                logger.warning(
                    "feature_computation_latency_exceeded",
                    symbol=symbol,
                    latency_ms=latency_ms,
                    threshold_ms=self._latency_threshold_ms,
                    trace_id=trace_id,
                )
            
            # Add logging for feature computation operations (T076)
            logger.info(
                "features_computed",
                symbol=symbol,
                features_count=len(filtered_features),
                latency_ms=latency_ms,
                feature_registry_version=self._feature_registry_version,
                trace_id=trace_id,
            )
            
            # Store last computed features for resilience (T078)
            self._last_features[symbol] = feature_vector
            
            return feature_vector
        
        except Exception as e:
            logger.error(
                "feature_computation_error",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
                exc_info=True,
            )
            
            # Handle ws-gateway unavailability (T078): return last available features
            if symbol in self._last_features:
                logger.warning(
                    "using_last_available_features",
                    symbol=symbol,
                    message="ws-gateway unavailable, using last computed features",
                )
                return self._last_features[symbol]
            
            return None
    
    def get_last_features(self, symbol: str) -> Optional[FeatureVector]:
        """Get last computed features for symbol (for resilience)."""
        return self._last_features.get(symbol)
    
    def update_market_data(self, event: Dict) -> None:
        """Update internal state with market data event."""
        symbol = event.get("symbol")
        if not symbol:
            return
        
        event_type = event.get("event_type")
        
        # Update orderbook
        if event_type == "orderbook_snapshot":
            self._orderbook_manager.apply_snapshot(event)
        elif event_type == "orderbook_delta":
            success = self._orderbook_manager.apply_delta(event)
            if not success:
                # Request snapshot if desynchronized
                if self._orderbook_manager.is_desynchronized(symbol):
                    self._orderbook_manager.request_snapshot(symbol)
        
        # Update rolling windows
        rolling_windows = self.get_rolling_windows(symbol)
        
        if event_type == "trade":
            rolling_windows.add_trade(event)
        elif event_type == "kline":
            rolling_windows.add_kline(event)
        elif event_type == "funding_rate":
            funding_rate = event.get("funding_rate")
            next_funding_time = event.get("next_funding_time")
            if funding_rate is not None:
                self.update_funding_rate(symbol, funding_rate, next_funding_time or 0)

