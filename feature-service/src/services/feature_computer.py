"""
Feature Computer service for orchestrating feature computations.
"""
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, TYPE_CHECKING
import structlog
import time

if TYPE_CHECKING:
    from src.services.feature_registry import FeatureRegistryLoader

from src.models.feature_vector import FeatureVector
from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from src.services.orderbook_manager import OrderbookManager
from src.services.feature_registry import FeatureRegistryLoader
from src.services.feature_requirements import (
    FeatureRequirementsAnalyzer,
    WindowRequirements,
)
from src.features.price_features import compute_all_price_features
from src.features.orderflow_features import compute_all_orderflow_features
from src.features.orderbook_features import compute_all_orderbook_features
from src.features.perpetual_features import compute_all_perpetual_features
from src.features.temporal_features import compute_all_temporal_features
from src.features.candle_patterns import compute_all_candle_patterns_3m, compute_all_candle_patterns_5m, compute_all_candle_patterns_15m

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
        # Требования к окнам (trade intervals + max lookback по 1m-клайнам)
        self._window_requirements: Optional[WindowRequirements] = None
        self._update_window_requirements()
    
    async def warmup_rolling_windows(
        self,
        symbols: list[str],
        bybit_client: Optional[Any] = None,
    ) -> None:
        """
        Preload recent klines into rolling windows on service startup.
        
        This ensures that rolling windows have sufficient historical data
        for feature computation immediately after startup, without waiting
        for klines to accumulate from the WebSocket stream.
        
        Args:
            symbols: List of symbols to warmup
            bybit_client: Optional BybitClient instance for fetching klines
        """
        if not bybit_client:
            logger.warning(
                "warmup_rolling_windows_skipped",
                reason="bybit_client_not_provided",
            )
            return
        
        if not self._window_requirements:
            self._update_window_requirements()
        
        max_lookback_minutes = (
            self._window_requirements.max_lookback_minutes_1m
            if self._window_requirements
            else 30
        )
        # Request slightly more than needed to account for API limits and rounding
        minutes_to_fetch = max_lookback_minutes + 10  # +10 minutes buffer
        
        logger.info(
            "warmup_rolling_windows_start",
            symbols=symbols,
            max_lookback_minutes=max_lookback_minutes,
            minutes_to_fetch=minutes_to_fetch,
        )
        
        for symbol in symbols:
            try:
                # Calculate start timestamp (minutes_to_fetch minutes ago)
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                start_time = now - timedelta(minutes=minutes_to_fetch)
                start_timestamp_ms = int(start_time.timestamp() * 1000)
                end_timestamp_ms = int(now.timestamp() * 1000)
                
                # Fetch klines from Bybit REST API
                from src.config import config
                response = await bybit_client.get(
                    endpoint="/v5/market/kline",
                    params={
                        "category": config.bybit_market_category,
                        "symbol": symbol,
                        "interval": "1",  # 1-minute klines
                        "start": start_timestamp_ms,
                        "end": end_timestamp_ms,
                        "limit": 200,  # Bybit API limit
                    },
                    authenticated=False,
                )
                
                if "result" not in response or "list" not in response["result"]:
                    logger.warning(
                        "warmup_klines_api_error",
                        symbol=symbol,
                        response_keys=list(response.keys()),
                    )
                    continue
                
                klines_data = response["result"]["list"]
                if not klines_data:
                    logger.warning(
                        "warmup_klines_empty",
                        symbol=symbol,
                    )
                    continue
                
                # Get or create rolling windows for symbol
                rolling_windows = self.get_rolling_windows(symbol)
                
                # Add klines to rolling windows
                klines_added = 0
                for kline_item in klines_data:
                    # Bybit API returns: [timestamp, open, high, low, close, volume, ...]
                    timestamp_ms = int(kline_item[0])
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    
                    kline_dict = {
                        "timestamp": timestamp,
                        "payload": {
                            "open": float(kline_item[1]),
                            "high": float(kline_item[2]),
                            "low": float(kline_item[3]),
                            "close": float(kline_item[4]),
                            "volume": float(kline_item[5]),
                        },
                    }
                    
                    rolling_windows.add_kline(kline_dict)
                    klines_added += 1
                
                logger.info(
                    "warmup_rolling_windows_complete",
                    symbol=symbol,
                    klines_added=klines_added,
                    time_span_minutes=round((now - datetime.fromtimestamp(int(klines_data[0][0]) / 1000, tz=timezone.utc)).total_seconds() / 60.0, 2),
                )
                
            except Exception as e:
                logger.error(
                    "warmup_rolling_windows_error",
                    symbol=symbol,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
    
    def _update_allowed_features(self) -> None:
        """Update allowed feature names from Feature Registry."""
        if self._feature_registry_loader is None:
            self._allowed_feature_names = None
            logger.info(
                "feature_registry_loader_not_available",
                message="Feature Registry loader is None, all features will be computed",
            )
            return
        
        try:
            registry_model = self._feature_registry_loader._registry_model
            if registry_model:
                self._allowed_feature_names = {f.name for f in registry_model.features}
                logger.info(
                    "feature_registry_features_loaded",
                    count=len(self._allowed_feature_names),
                    version=self._feature_registry_version,
                    has_returns_3m="returns_3m" in self._allowed_feature_names,
                    has_returns_5m="returns_5m" in self._allowed_feature_names,
                    has_volatility_10m="volatility_10m" in self._allowed_feature_names,
                    has_volatility_15m="volatility_15m" in self._allowed_feature_names,
                    sample_features=sorted(list(self._allowed_feature_names))[:10],
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
                            version=config.get("version"),
                            has_returns_3m="returns_3m" in self._allowed_feature_names,
                            has_returns_5m="returns_5m" in self._allowed_feature_names,
                            has_volatility_10m="volatility_10m" in self._allowed_feature_names,
                            has_volatility_15m="volatility_15m" in self._allowed_feature_names,
                        )
                    else:
                        logger.warning(
                            "feature_registry_config_empty",
                            message="Config is None or has no features section",
                        )
                        self._allowed_feature_names = None
                except Exception as e:
                    logger.warning(
                        "failed_to_load_feature_registry_for_filtering",
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
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

    def _update_window_requirements(self) -> None:
        """
        Обновить требования к rolling windows на основе активного Feature Registry.

        Используется только онлайн-движком, оффлайн-движок имеет свою реализацию
        в OfflineEngine._compute_max_lookback_minutes().
        """
        try:
            analyzer = FeatureRequirementsAnalyzer(self._feature_registry_loader)
            self._window_requirements = analyzer.compute_requirements()
        except Exception as e:
            logger.warning(
                "failed_to_update_window_requirements",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Безопасный fallback — полный набор трейдовых окон + 30 минут lookback
            self._window_requirements = WindowRequirements(
                trade_intervals={"1s", "3s", "15s", "1m"},
                max_lookback_minutes_1m=30,
            )
    
    def get_rolling_windows(self, symbol: str) -> RollingWindows:
        """Get or create rolling windows for symbol."""
        if symbol not in self._rolling_windows:
            from datetime import datetime, timezone
            import pandas as pd

            # Если по какой-то причине требования ещё не посчитаны — посчитаем здесь
            if self._window_requirements is None:
                self._update_window_requirements()

            trade_intervals = (
                self._window_requirements.trade_intervals
                if self._window_requirements is not None
                else {"1m"}
            )

            windows: Dict[str, "pd.DataFrame"] = {}
            for interval in trade_intervals:
                # трейдовые окна содержат price/volume/side
                windows[interval] = pd.DataFrame(
                    columns=["timestamp", "price", "volume", "side"]
                )

            # 1m окно для клайнов всегда нужно (add_kline пишет именно туда)
            if "1m" not in windows:
                windows["1m"] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            max_lookback = (
                self._window_requirements.max_lookback_minutes_1m
                if self._window_requirements is not None
                else None
            )
            
            logger.info(
                "rolling_windows_created",
                symbol=symbol,
                trade_intervals=sorted(trade_intervals),
                max_lookback_minutes_1m=max_lookback,
                feature_registry_version=self._feature_registry_version,
            )
            
            self._rolling_windows[symbol] = RollingWindows(
                symbol=symbol,
                windows=windows,
                last_update=datetime.now(timezone.utc),
                window_intervals=trade_intervals,
                max_lookback_minutes_1m=max_lookback,
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
            
            # Get current price - try orderbook first, then fallback to latest kline
            current_price = orderbook.get_mid_price() if orderbook else None
            if current_price is None:
                # Fallback: get price from latest kline
                now = datetime.now(timezone.utc)
                klines = rolling_windows.get_klines_for_window("1m", now - timedelta(minutes=1), now)
                if len(klines) > 0 and "close" in klines.columns:
                    klines_sorted = klines.sort_values("timestamp")
                    latest_close = klines_sorted.iloc[-1]["close"]
                    try:
                        current_price = float(latest_close)
                    except (ValueError, TypeError):
                        current_price = None
            
            # Compute all feature groups
            all_features = {}
            
            # Price features
            price_features = compute_all_price_features(
                orderbook,
                rolling_windows,
                current_price,
                allowed_feature_names=self._allowed_feature_names,
            )
            all_features.update(price_features)
            
            # Always add price feature explicitly (model-service expects it)
            # Use mid_price if available, otherwise use current_price from kline
            if current_price is not None:
                all_features["price"] = current_price
            else:
                logger.warning(
                    "current_price_unavailable",
                    symbol=symbol,
                    has_orderbook=orderbook is not None,
                    has_klines=len(rolling_windows.get_klines_for_window("1m", datetime.now(timezone.utc) - timedelta(minutes=1), datetime.now(timezone.utc))) > 0,
                )
            
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
            
            # Log all features before filtering for debugging
            import math
            logger.debug(
                "all_features_before_filtering",
                all_features_count=len(all_features),
                all_feature_names=sorted(list(all_features.keys())),
                none_features=[k for k, v in all_features.items() if v is None],
                nan_inf_features=[k for k, v in all_features.items() if isinstance(v, float) and (math.isnan(v) or math.isinf(v))],
            )
            
            # Log target features before filtering
            target_features_before = {
                k: v for k, v in all_features.items()
                if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
            }
            if target_features_before:
                logger.info(
                    "target_features_before_filtering",
                    features=target_features_before,
                    all_features_count=len(all_features),
                )
            
            # Filter out None values and NaN/Inf values (not JSON compliant)
            filtered_features = {}
            removed_features = []
            none_count = 0
            nan_inf_count = 0
            for k, v in all_features.items():
                if v is None:
                    removed_features.append(f"{k}=None")
                    none_count += 1
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    removed_features.append(f"{k}={v} (NaN/Inf)")
                    nan_inf_count += 1
                else:
                    filtered_features[k] = v
            
            if removed_features:
                logger.info(
                    "features_filtered_none_nan_inf",
                    filtered_count=len(filtered_features),
                    removed_count=len(removed_features),
                    none_count=none_count,
                    nan_inf_count=nan_inf_count,
                    removed_features_sample=removed_features[:20],  # Первые 20 для краткости
                )
            
            # Log target features after None/NaN/Inf filtering
            target_features_after = {
                k: v for k, v in filtered_features.items()
                if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
            }
            if target_features_before and len(target_features_after) < len(target_features_before):
                logger.warning(
                    "target_features_filtered_out",
                    before=target_features_before,
                    after=target_features_after,
                    removed=[k for k in target_features_before.keys() if k not in target_features_after],
                    all_removed_count=len(removed_features),
                )
            
            # Filter features by Feature Registry (if enabled)
            # Note: "price" is always included even if not in registry (required by model-service)
            if self._allowed_feature_names is not None:
                # Log target features before registry filtering
                target_before_registry = {
                    k: v for k, v in filtered_features.items()
                    if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
                }
                
                registry_filtered_features = {
                    k: v for k, v in filtered_features.items()
                    if k in self._allowed_feature_names or k == "price"
                }
                
                # Log target features after registry filtering
                target_after_registry = {
                    k: v for k, v in registry_filtered_features.items()
                    if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
                }
                
                if len(registry_filtered_features) < len(filtered_features):
                    removed_features = set(filtered_features.keys()) - set(registry_filtered_features.keys())
                    logger.warning(
                        "features_filtered_by_registry",
                        original_count=len(filtered_features),
                        filtered_count=len(registry_filtered_features),
                        removed_features=sorted(list(removed_features)),
                        removed_count=len(removed_features),
                        target_features_before_registry=target_before_registry,
                        target_features_after_registry=target_after_registry,
                        target_features_removed={
                            "returns_3m": "returns_3m" in removed_features,
                            "returns_5m": "returns_5m" in removed_features,
                            "volatility_10m": "volatility_10m" in removed_features,
                            "volatility_15m": "volatility_15m" in removed_features,
                        },
                        allowed_features_count=len(self._allowed_feature_names),
                    )
                filtered_features = registry_filtered_features
            
            # Log final features before creating FeatureVector
            target_final = {
                k: v for k, v in filtered_features.items()
                if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
            }
            logger.info(
                "features_before_feature_vector_creation",
                target_features=target_final,
                total_features_count=len(filtered_features),
                all_feature_names=sorted(list(filtered_features.keys())),
            )
            
            # Create feature vector
            feature_vector = FeatureVector(
                timestamp=timestamp,
                symbol=symbol,
                features=filtered_features,
                feature_registry_version=self._feature_registry_version,
                trace_id=trace_id,
            )
            
            # Log features after FeatureVector creation
            target_after_creation = {
                k: v for k, v in feature_vector.features.items()
                if k in ["returns_3m", "returns_5m", "volatility_10m", "volatility_15m"]
            }
            if len(target_after_creation) < len(target_final):
                logger.warning(
                    "target_features_lost_in_feature_vector",
                    before=target_final,
                    after=target_after_creation,
                    lost=[k for k in target_final.keys() if k not in target_after_creation],
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
                klines_before = len(rolling_windows.get_window_data("1m"))
                rolling_windows.add_kline(event)
                klines_after = len(rolling_windows.get_window_data("1m"))
                logger.debug(
                    "kline_added",
                    symbol=symbol,
                    klines_before=klines_before,
                    klines_after=klines_after,
                    timestamp=event.get("timestamp") or (event.get("payload", {}) or {}).get("start"),
                )
            except (KeyError, TypeError) as e:
                logger.warning(
                    "kline_processing_error",
                    symbol=symbol,
                    error=str(e),
                    event_keys=list(event.keys()),
                    payload_keys=list(event.get("payload", {}).keys()) if isinstance(event.get("payload"), dict) else None,
                )
        elif event_type in ("funding", "funding_rate"):
            # ws-gateway publishes funding events with payload containing fundingRate and nextFundingTime,
            # но в тестах и некоторых утилитах значения могут приходить на верхнем уровне события.
            # Поддерживаем оба варианта.
            payload = event.get("payload", {})
            src = payload if isinstance(payload, dict) and payload else event

            funding_rate = src.get("fundingRate") or src.get("funding_rate")
            next_funding_time = src.get("nextFundingTime") or src.get("next_funding_time")
                
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

