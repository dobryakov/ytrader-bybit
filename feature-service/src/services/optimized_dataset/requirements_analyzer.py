"""
Feature Requirements Analyzer for optimized dataset building.

Analyzes Feature Registry to determine data requirements, timestamp frequency,
and optimization strategies.
"""
from datetime import timedelta
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import structlog

from src.models.feature_registry import FeatureRegistry, FeatureDefinition

logger = structlog.get_logger(__name__)


class TimestampStrategy(str, Enum):
    """Strategy for generating timestamps."""
    KLINES_ONLY = "klines_only"  # Only klines timestamps (1 minute intervals)
    TRADES_ONLY = "trades_only"  # All trades timestamps (high frequency)
    KLINES_WITH_TRADES = "klines_with_trades"  # Klines + trades for orderflow
    CUSTOM_INTERVAL = "custom_interval"  # Fixed interval (if needed)


@dataclass
class DataRequirements:
    """Data requirements determined from Feature Registry."""
    # Required data types
    required_data_types: Set[str]  # {"orderbook", "kline", "trades", "ticker", "funding"}
    
    # Maximum lookback period in minutes
    max_lookback_minutes: int
    
    # Timestamp generation strategy
    timestamp_strategy: TimestampStrategy
    
    # Required kline columns
    required_kline_columns: Set[str]  # {"open", "close", "high", "low", "volume"}
    
    # Data source flags
    needs_orderbook: bool
    needs_trades: bool
    needs_klines: bool
    needs_ticker: bool
    needs_funding: bool
    
    # Feature grouping for optimization
    feature_groups: Dict[str, List[str]]  # {"price": [...], "technical": [...], "orderflow": [...]}
    
    # Storage type mapping
    storage_types: Dict[str, List[str]]  # {"orderbook": ["snapshots", "deltas"], ...}


class FeatureRequirementsAnalyzer:
    """Analyzes Feature Registry to determine data requirements."""
    
    # Mapping of feature names to their groups (for optimization)
    FEATURE_GROUP_MAPPING = {
        # Price features
        "price": [
            "mid_price", "spread_abs", "spread_rel",
            "returns_1s", "returns_3s", "returns_1m", "returns_3m", "returns_5m",
            "vwap_3s", "vwap_15s", "vwap_1m", "vwap_3m", "vwap_5m",
            "volume_3s", "volume_15s", "volume_1m", "volume_3m", "volume_5m",
            "volatility_1m", "volatility_5m", "volatility_10m", "volatility_15m",
            "price_ema21_ratio",
        ],
        # Technical indicators
        "technical": [
            "ema_21", "rsi_14",
        ],
        # Orderflow features
        "orderflow": [
            "signed_volume_1s", "signed_volume_3s", "signed_volume_15s", "signed_volume_1m",
            "buy_sell_volume_ratio", "trade_count_3s", "net_aggressor_pressure",
        ],
        # Orderbook features
        "orderbook": [
            "depth_bid_top5", "depth_bid_top10", "depth_ask_top5", "depth_ask_top10",
            "depth_imbalance_top5", "depth_imbalance_top10",
        ],
        # Perpetual features
        "perpetual": [
            "funding_rate", "time_to_funding",
        ],
        # Temporal features
        "temporal": [
            "time_of_day_sin", "time_of_day_cos",
        ],
        # Candle pattern features
        "candle_patterns": [
            # Pattern names vary by version
        ],
    }
    
    def analyze(self, registry: FeatureRegistry) -> DataRequirements:
        """
        Analyze Feature Registry and determine data requirements.
        
        Args:
            registry: FeatureRegistry instance
            
        Returns:
            DataRequirements with all determined requirements
        """
        # Get required data types
        required_data_types = registry.get_required_data_types()
        
        # Determine data source flags
        needs_orderbook = "orderbook" in required_data_types
        needs_trades = "trades" in required_data_types
        needs_klines = "kline" in required_data_types
        needs_ticker = "ticker" in required_data_types
        needs_funding = "funding" in required_data_types
        
        # Determine timestamp strategy
        timestamp_strategy = self._determine_timestamp_strategy(
            needs_klines, needs_trades
        )
        
        # Determine maximum lookback
        max_lookback_minutes = self._compute_max_lookback_minutes(registry)
        
        # Determine required kline columns
        required_kline_columns = self._determine_kline_columns(registry, needs_klines)
        
        # Group features for optimization
        feature_groups = self._group_features(registry)
        
        # Get storage type mapping
        storage_types = self._get_storage_types(required_data_types)
        
        requirements = DataRequirements(
            required_data_types=required_data_types,
            max_lookback_minutes=max_lookback_minutes,
            timestamp_strategy=timestamp_strategy,
            required_kline_columns=required_kline_columns,
            needs_orderbook=needs_orderbook,
            needs_trades=needs_trades,
            needs_klines=needs_klines,
            needs_ticker=needs_ticker,
            needs_funding=needs_funding,
            feature_groups=feature_groups,
            storage_types=storage_types,
        )
        
        logger.info(
            "feature_requirements_analyzed",
            required_data_types=sorted(required_data_types),
            max_lookback_minutes=max_lookback_minutes,
            timestamp_strategy=timestamp_strategy.value,
            needs_orderbook=needs_orderbook,
            needs_trades=needs_trades,
            needs_klines=needs_klines,
            feature_groups_count={k: len(v) for k, v in feature_groups.items()},
        )
        
        return requirements
    
    def _determine_timestamp_strategy(
        self, needs_klines: bool, needs_trades: bool
    ) -> TimestampStrategy:
        """
        Determine timestamp generation strategy based on data requirements.
        
        Args:
            needs_klines: Whether klines are needed
            needs_trades: Whether trades are needed
            
        Returns:
            TimestampStrategy
        """
        if needs_trades and needs_klines:
            return TimestampStrategy.KLINES_WITH_TRADES
        elif needs_trades and not needs_klines:
            return TimestampStrategy.TRADES_ONLY
        elif needs_klines and not needs_trades:
            return TimestampStrategy.KLINES_ONLY
        else:
            # Fallback: use klines if available
            return TimestampStrategy.KLINES_ONLY
    
    def _compute_max_lookback_minutes(self, registry: FeatureRegistry) -> int:
        """
        Compute maximum lookback period in minutes from all features.
        
        Args:
            registry: FeatureRegistry instance
            
        Returns:
            Maximum lookback in minutes (with 5 minute buffer)
        """
        max_lookback = 0
        
        for feature in registry.features:
            lookback_minutes = self._parse_lookback_window(feature.lookback_window)
            if lookback_minutes is not None:
                max_lookback = max(max_lookback, lookback_minutes)
        
        # Add 5 minute buffer for safety
        max_lookback = max_lookback + 5
        
        # Minimum lookback is 30 minutes (covers most common features)
        if max_lookback < 30:
            max_lookback = 30
        
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
            logger.warning(
                "failed_to_parse_lookback_window",
                lookback_window=lookback_window,
            )
            return None
    
    def _determine_kline_columns(
        self, registry: FeatureRegistry, needs_klines: bool
    ) -> Set[str]:
        """
        Determine which kline columns are required based on features.
        
        Args:
            registry: FeatureRegistry instance
            needs_klines: Whether klines are needed
            
        Returns:
            Set of required column names
        """
        if not needs_klines:
            return set()
        
        # Default: all columns are needed for most features
        # But we can optimize by analyzing feature names
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        
        # Check if we can reduce columns based on feature names
        feature_names = {f.name for f in registry.features}
        
        # If no features use high/low, we might not need them
        # But for safety, keep all columns for now
        # TODO: Add more sophisticated analysis if needed
        
        return required_columns
    
    def _group_features(self, registry: FeatureRegistry) -> Dict[str, List[str]]:
        """
        Group features by type for optimization purposes.
        
        Args:
            registry: FeatureRegistry instance
            
        Returns:
            Dictionary mapping group names to feature name lists
        """
        groups: Dict[str, List[str]] = {
            "price": [],
            "technical": [],
            "orderflow": [],
            "orderbook": [],
            "perpetual": [],
            "temporal": [],
            "candle_patterns": [],
            "other": [],
        }
        
        for feature in registry.features:
            feature_name = feature.name
            assigned = False
            
            # Check each group mapping
            for group_name, feature_list in self.FEATURE_GROUP_MAPPING.items():
                if feature_name in feature_list:
                    groups[group_name].append(feature_name)
                    assigned = True
                    break
            
            # Check by input sources for fallback grouping
            if not assigned:
                if "orderbook" in feature.input_sources:
                    groups["orderbook"].append(feature_name)
                elif "trades" in feature.input_sources:
                    groups["orderflow"].append(feature_name)
                elif "kline" in feature.input_sources:
                    groups["price"].append(feature_name)
                elif "funding" in feature.input_sources:
                    groups["perpetual"].append(feature_name)
                else:
                    groups["other"].append(feature_name)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _get_storage_types(self, required_data_types: Set[str]) -> Dict[str, List[str]]:
        """
        Map Feature Registry input_sources to actual data storage types.
        
        Args:
            required_data_types: Set of required input sources
            
        Returns:
            Dictionary mapping input_source to list of storage types
        """
        mapping = {
            "orderbook": ["orderbook_snapshots", "orderbook_deltas"],
            "kline": ["klines"],
            "trades": ["trades"],
            "ticker": ["ticker"],
            "funding": ["funding"],
        }
        
        return {source: mapping[source] for source in required_data_types if source in mapping}

