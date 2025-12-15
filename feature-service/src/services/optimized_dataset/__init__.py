"""
Optimized Dataset Building Components.

This package contains optimized implementations for building datasets
with improved performance through caching, vectorization, and streaming.
"""
from .requirements_analyzer import (
    FeatureRequirementsAnalyzer,
    DataRequirements,
    TimestampStrategy,
)
from .rolling_window import OptimizedRollingWindow
from .cache_strategy import (
    AdaptiveCacheStrategy,
    CacheStrategy,
    CacheUnit,
)
from .daily_cache import OptimizedDailyDataCache
from .prefetcher import AdaptivePrefetcher
from .vectorized_features import (
    compute_technical_indicators_vectorized,
    compute_orderflow_features_vectorized,
    compute_price_features_vectorized,
)
from .hybrid_feature_computer import HybridFeatureComputer
from .incremental_orderbook import IncrementalOrderbookManager
from .streaming_builder import StreamingDatasetBuilder
from .optimized_builder import OptimizedDatasetBuilder

__all__ = [
    "FeatureRequirementsAnalyzer",
    "DataRequirements",
    "TimestampStrategy",
    "OptimizedRollingWindow",
    "AdaptiveCacheStrategy",
    "CacheStrategy",
    "CacheUnit",
    "OptimizedDailyDataCache",
    "AdaptivePrefetcher",
    "compute_technical_indicators_vectorized",
    "compute_orderflow_features_vectorized",
    "compute_price_features_vectorized",
    "HybridFeatureComputer",
    "IncrementalOrderbookManager",
    "StreamingDatasetBuilder",
    "OptimizedDatasetBuilder",
]

