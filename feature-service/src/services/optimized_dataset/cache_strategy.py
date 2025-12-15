"""
Adaptive Cache Strategy for optimized dataset building.

Determines optimal caching strategy based on dataset period length and available resources.
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class CacheUnit(str, Enum):
    """Cache unit type."""
    DAYS = "days"  # Cache by days
    FULL_PERIOD = "full_period"  # Cache entire period
    HOURS = "hours"  # Cache by hours (for very long periods)


@dataclass
class CacheStrategy:
    """Cache strategy configuration."""
    cache_unit: CacheUnit
    cache_size_days: int  # Number of days to cache (for DAYS unit)
    cache_size_hours: Optional[int] = None  # Number of hours to cache (for HOURS unit)
    prefetch_enabled: bool = False
    prefetch_ahead_hours: Optional[int] = None  # Hours ahead to prefetch
    prefetch_adaptive: bool = False  # Use adaptive prefetching
    ttl_seconds: int = 3600  # TTL for cache entries (default: 1 hour)
    local_buffer_minutes: int = 60  # Local in-memory buffer size in minutes


class AdaptiveCacheStrategy:
    """
    Adaptively determines optimal caching strategy based on period length.
    
    Strategies:
    - Short periods (1-3 days): Cache entire period
    - Medium periods (4-7 days): Cache 2 days + prefetch next day
    - Long periods (8+ days): Cache 1 day + adaptive prefetch (2 hours ahead)
    """
    
    # Configuration thresholds
    SHORT_PERIOD_DAYS = 3  # Cache entire period
    MEDIUM_PERIOD_DAYS = 7  # Cache 2 days
    LONG_PERIOD_DAYS = 8  # Cache 1 day + prefetch
    
    def __init__(
        self,
        short_period_days: int = None,
        medium_period_days: int = None,
        long_period_days: int = None,
    ):
        """
        Initialize adaptive cache strategy.
        
        Args:
            short_period_days: Threshold for short period (default: 3)
            medium_period_days: Threshold for medium period (default: 7)
            long_period_days: Threshold for long period (default: 8)
        """
        self.short_period_days = short_period_days or self.SHORT_PERIOD_DAYS
        self.medium_period_days = medium_period_days or self.MEDIUM_PERIOD_DAYS
        self.long_period_days = long_period_days or self.LONG_PERIOD_DAYS
    
    def determine_strategy(
        self,
        period_days: int,
        symbol: str,
        data_types: list,
        available_redis_memory: Optional[int] = None,
        max_lookback_minutes: int = 30,
    ) -> CacheStrategy:
        """
        Determine optimal cache strategy based on period length.
        
        Args:
            period_days: Total period length in days
            symbol: Trading pair symbol
            data_types: List of required data types
            available_redis_memory: Available Redis memory in bytes (optional)
            max_lookback_minutes: Maximum lookback period in minutes
            
        Returns:
            CacheStrategy configuration
        """
        # Estimate data size per day (rough estimate)
        estimated_size_per_day_mb = self._estimate_data_size_per_day(
            symbol, data_types
        )
        
        # Adjust strategy based on available memory if provided
        if available_redis_memory:
            max_cache_days = int(available_redis_memory / (estimated_size_per_day_mb * 1024 * 1024))
            if max_cache_days < period_days:
                # Memory constraint - use more aggressive caching
                if max_cache_days >= 2:
                    cache_size_days = 2
                else:
                    cache_size_days = 1
            else:
                cache_size_days = None  # No constraint
        else:
            cache_size_days = None
        
        # Determine strategy based on period length
        if period_days <= self.short_period_days:
            # Short period: cache entire period
            strategy = CacheStrategy(
                cache_unit=CacheUnit.FULL_PERIOD,
                cache_size_days=period_days,
                prefetch_enabled=False,
                ttl_seconds=3600,
                local_buffer_minutes=max_lookback_minutes + 60,
            )
            
            logger.info(
                "cache_strategy_determined",
                strategy="full_period",
                period_days=period_days,
                cache_size_days=period_days,
            )
            
        elif period_days <= self.medium_period_days:
            # Medium period: cache 2 days + prefetch next day
            cache_days = cache_size_days if cache_size_days else 2
            
            strategy = CacheStrategy(
                cache_unit=CacheUnit.DAYS,
                cache_size_days=cache_days,
                prefetch_enabled=True,
                prefetch_ahead_hours=24,  # Prefetch next day
                prefetch_adaptive=False,
                ttl_seconds=3600,
                local_buffer_minutes=max_lookback_minutes + 60,
            )
            
            logger.info(
                "cache_strategy_determined",
                strategy="medium_period",
                period_days=period_days,
                cache_size_days=cache_days,
                prefetch_ahead_hours=24,
            )
            
        else:
            # Long period: cache 1 day + adaptive prefetch
            cache_days = cache_size_days if cache_size_days else 1
            
            strategy = CacheStrategy(
                cache_unit=CacheUnit.DAYS,
                cache_size_days=cache_days,
                prefetch_enabled=True,
                prefetch_ahead_hours=2,  # Prefetch 2 hours ahead
                prefetch_adaptive=True,  # Adaptive based on processing speed
                ttl_seconds=3600,
                local_buffer_minutes=max_lookback_minutes + 60,
            )
            
            logger.info(
                "cache_strategy_determined",
                strategy="long_period",
                period_days=period_days,
                cache_size_days=cache_days,
                prefetch_ahead_hours=2,
                prefetch_adaptive=True,
            )
        
        return strategy
    
    def _estimate_data_size_per_day(self, symbol: str, data_types: list) -> float:
        """
        Estimate data size per day in MB (rough estimate).
        
        Args:
            symbol: Trading pair symbol
            data_types: List of required data types
            
        Returns:
            Estimated size in MB per day
        """
        # Rough estimates per day (in MB)
        size_estimates = {
            "klines": 0.1,  # ~1440 records/day * 50 bytes = ~72 KB
            "trades": 50.0,  # ~100K-1M records/day * 100 bytes = ~10-100 MB
            "orderbook_snapshots": 1.2,  # ~24 snapshots/day * 50 KB = ~1.2 MB
            "orderbook_deltas": 50.0,  # ~100K-500K deltas/day * 200 bytes = ~20-100 MB
            "ticker": 0.01,  # ~1440 records/day * 100 bytes = ~144 KB
            "funding": 0.001,  # ~8 records/day * 100 bytes = ~1 KB
        }
        
        total_size = 0.0
        for data_type in data_types:
            if data_type == "orderbook":
                total_size += size_estimates.get("orderbook_snapshots", 0)
                total_size += size_estimates.get("orderbook_deltas", 0)
            elif data_type == "kline":
                total_size += size_estimates.get("klines", 0)
            elif data_type == "trades":
                total_size += size_estimates.get("trades", 0)
            elif data_type == "ticker":
                total_size += size_estimates.get("ticker", 0)
            elif data_type == "funding":
                total_size += size_estimates.get("funding", 0)
        
        return total_size

