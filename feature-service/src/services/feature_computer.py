"""
Feature Computer service for orchestrating feature computations.
"""
from datetime import datetime, timezone
from typing import Dict, Optional, TYPE_CHECKING
import structlog
import time

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
from src.features.candle_patterns import compute_all_candle_patterns_3m

logger = structlog.get_logger(__name__)


class FeatureComputer:
    """Orchestrates feature computation from market data."""
    
    def __init__(
        self,
        orderbook_manager: OrderbookManager,
        feature_registry_version: str = "1.0.0",
        feature_registry_loader: Optional["FeatureRegistryLoader"] = None,
    ):
        """
        Initialize feature computer.
        
        Args:
            orderbook_manager: OrderbookManager instance
            feature_registry_version: Feature Registry version string
            feature_registry_loader: Optional FeatureRegistryLoader for filtering features
        """
        self._orderbook_manager = orderbook_manager
        self._rolling_windows: Dict[str, RollingWindows] = {}
        self._feature_registry_version = feature_registry_version
        self._feature_registry_loader = feature_registry_loader
        self._latest_funding_rate: Dict[str, Optional[float]] = {}
        self._latest_next_funding_time: Dict[str, Optional[int]] = {}
        self._latency_threshold_ms = 70.0
        # Store last computed features per symbol for resilience (T078)
        self._last_features: Dict[str, FeatureVector] = {}
        
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
            
            # Candlestick pattern features (3-minute window)
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
                    )
                filtered_features = registry_filtered_features
            
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
        payload = event.get("payload", {})
        
        # Update orderbook
        # ws-gateway publishes events with event_type="orderbook"
        # Payload contains data directly (s, b, a, seq, u) or nested in data field
        if event_type == "orderbook":
            # Extract data from payload (ws-gateway format)
            # Data can be in payload.data or directly in payload
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            if not isinstance(data, dict):
                data = payload if isinstance(payload, dict) else {}
            
            # Determine type: check payload.data.type, payload.type, or infer from structure
            orderbook_type = data.get("type") or payload.get("type")
            
            # If no type field, check if it's a snapshot by structure (has bids/asks)
            if orderbook_type is None:
                if "b" in data or "a" in data or "bids" in data or "asks" in data:
                    orderbook_type = "snapshot"
                else:
                    orderbook_type = "delta"
            
            logger.debug(
                "processing_orderbook_event",
                symbol=symbol,
                orderbook_type=orderbook_type,
                has_data=bool(data),
            )
            
            if orderbook_type == "snapshot":
                # Convert ws-gateway format to feature-service format
                # Handle timestamp conversion
                timestamp = event.get("timestamp") or event.get("exchange_timestamp")
                if isinstance(timestamp, str):
                    from dateutil.parser import parse
                    timestamp = parse(timestamp)
                elif timestamp is None:
                    from datetime import datetime, timezone
                    timestamp = datetime.now(timezone.utc)
                elif not isinstance(timestamp, datetime):
                    # Convert numeric timestamp to datetime
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.fromtimestamp(
                            timestamp / 1000 if timestamp > 1e10 else timestamp,
                            tz=timezone.utc
                        )
                    else:
                        timestamp = datetime.now(timezone.utc)
                
                snapshot_data = {
                    "symbol": data.get("s") or symbol,
                    "bids": data.get("b", []),
                    "asks": data.get("a", []),
                    "sequence": data.get("seq", data.get("u", 0)),
                    "timestamp": timestamp,
                }
                logger.debug(
                    "applying_orderbook_snapshot",
                    symbol=snapshot_data["symbol"],
                    bids_count=len(snapshot_data["bids"]),
                    asks_count=len(snapshot_data["asks"]),
                    sequence=snapshot_data["sequence"],
                )
                try:
                    self._orderbook_manager.apply_snapshot(snapshot_data)
                except Exception as e:
                    logger.error(
                        "orderbook_snapshot_apply_failed",
                        symbol=symbol,
                        error=str(e),
                        exc_info=True,
                    )
            elif orderbook_type == "delta" or orderbook_type == "update":
                # Convert ws-gateway format to feature-service format
                delta_data = {
                    "symbol": data.get("s") or symbol,
                    "sequence": data.get("seq", data.get("u", 0)),
                    "delta_type": "update",  # Bybit uses update for all changes
                    "side": "bid",  # Will be determined from data
                    "price": None,  # Will be extracted from data
                    "quantity": None,  # Will be extracted from data
                }
                # Bybit delta format: data.b and data.a contain updates
                # For now, we'll mark as needing snapshot if delta format is complex
                # TODO: Implement proper delta parsing from Bybit format
                if self._orderbook_manager.is_desynchronized(symbol):
                    self._orderbook_manager.request_snapshot(symbol)
        elif event_type == "orderbook_snapshot":
            self._orderbook_manager.apply_snapshot(event)
        elif event_type == "orderbook_delta":
            success = self._orderbook_manager.apply_delta(event)
            if not success:
                # Request snapshot if desynchronized
                if self._orderbook_manager.is_desynchronized(symbol):
                    self._orderbook_manager.request_snapshot(symbol)
        
        # Update rolling windows
        rolling_windows = self.get_rolling_windows(symbol)
        
        # Handle trades - ws-gateway may publish as "trade" or "trades"
        if event_type == "trade" or event_type == "trades":
            # ws-gateway creates one event per trade item, payload contains the trade directly
            # Payload format: { "p": price, "v": volume, "S": side, "T": timestamp, "s": symbol, ... }
            if isinstance(payload, dict):
                # Payload is the trade object itself (not wrapped in "data")
                trade_data = {
                    "price": payload.get("p") or payload.get("price"),
                    "quantity": payload.get("v") or payload.get("quantity") or payload.get("volume"),
                    "side": payload.get("S") or payload.get("side", "Buy"),
                    "timestamp": payload.get("T") or payload.get("timestamp") or event.get("timestamp") or event.get("exchange_timestamp"),
                    "symbol": payload.get("s") or symbol,
                }
                rolling_windows.add_trade(trade_data)
            elif isinstance(payload, list) and len(payload) > 0:
                # If payload is a list (shouldn't happen with current ws-gateway, but handle it)
                for trade_item in payload:
                    if isinstance(trade_item, dict):
                        trade_data = {
                            "price": trade_item.get("p") or trade_item.get("price"),
                            "quantity": trade_item.get("v") or trade_item.get("quantity") or trade_item.get("volume"),
                            "side": trade_item.get("S") or trade_item.get("side", "Buy"),
                            "timestamp": trade_item.get("T") or trade_item.get("timestamp") or event.get("timestamp") or event.get("exchange_timestamp"),
                            "symbol": trade_item.get("s") or symbol,
                        }
                        rolling_windows.add_trade(trade_data)
        elif event_type == "kline":
            try:
                rolling_windows.add_kline(event)
            except (KeyError, TypeError) as e:
                logger.warning(
                    "kline_processing_error",
                    symbol=symbol,
                    error=str(e),
                    event_keys=list(event.keys()),
                    payload_keys=list(event.get("payload", {}).keys()) if isinstance(event.get("payload"), dict) else None,
                )
        elif event_type == "funding":
            # ws-gateway publishes funding events with payload containing fundingRate and nextFundingTime
            # Payload can have camelCase (fundingRate, nextFundingTime) or snake_case (funding_rate, next_funding_time)
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                funding_rate = payload.get("fundingRate") or payload.get("funding_rate")
                next_funding_time = payload.get("nextFundingTime") or payload.get("next_funding_time")
                
                # Convert funding_rate to float if it's a string
                if funding_rate is not None:
                    try:
                        funding_rate = float(funding_rate) if isinstance(funding_rate, str) else funding_rate
                    except (ValueError, TypeError):
                        logger.warning(
                            "invalid_funding_rate",
                            symbol=symbol,
                            funding_rate=funding_rate,
                            funding_rate_type=type(funding_rate).__name__,
                        )
                        funding_rate = None
                
                # Convert next_funding_time to int if it's a string
                if next_funding_time is not None:
                    try:
                        next_funding_time = int(next_funding_time) if isinstance(next_funding_time, str) else next_funding_time
                    except (ValueError, TypeError):
                        logger.warning(
                            "invalid_next_funding_time",
                            symbol=symbol,
                            next_funding_time=next_funding_time,
                            next_funding_time_type=type(next_funding_time).__name__,
                        )
                        next_funding_time = 0
                
                if funding_rate is not None:
                    self.update_funding_rate(symbol, funding_rate, next_funding_time or 0)
                    logger.debug(
                        "funding_rate_updated",
                        symbol=symbol,
                        funding_rate=funding_rate,
                        next_funding_time=next_funding_time,
                    )

