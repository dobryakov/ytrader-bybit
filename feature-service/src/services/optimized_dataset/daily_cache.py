"""
Optimized Daily Data Cache with multi-level caching.

Implements Redis cache + local in-memory buffer for efficient data access.
"""
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
import hashlib
import pandas as pd
import structlog

from src.services.cache_service import CacheService
from src.storage.parquet_storage import ParquetStorage
from .cache_strategy import CacheStrategy, CacheUnit

logger = structlog.get_logger(__name__)


class OptimizedDailyDataCache:
    """
    Multi-level cache for daily market data.
    
    Levels:
    1. Local in-memory cache (fastest) - current day + lookback buffer
    2. Redis cache (fast) - multiple days based on strategy
    3. Parquet storage (slowest) - source of truth
    """
    
    def __init__(
        self,
        redis_cache: CacheService,
        parquet_storage: ParquetStorage,
        strategy: CacheStrategy,
        symbol: str,
        data_types: List[str],
    ):
        """
        Initialize optimized daily data cache.
        
        Args:
            redis_cache: Redis cache service
            parquet_storage: Parquet storage service
            strategy: Cache strategy configuration
            symbol: Trading pair symbol
            data_types: List of required data types
        """
        self.redis_cache = redis_cache
        self.parquet_storage = parquet_storage
        self.strategy = strategy
        self.symbol = symbol
        self.data_types = data_types
        
        # Local in-memory cache (current day + buffer)
        self.local_cache: Dict[date, Dict[str, pd.DataFrame]] = {}
        self.local_cache_dates: List[date] = []  # Keep track of dates for eviction
        
        # Statistics
        self.stats = {
            "local_hits": 0,
            "redis_hits": 0,
            "parquet_reads": 0,
            "cache_writes": 0,
        }
        
        logger.info(
            "optimized_daily_cache_initialized",
            symbol=symbol,
            cache_unit=strategy.cache_unit.value,
            cache_size_days=strategy.cache_size_days,
            prefetch_enabled=strategy.prefetch_enabled,
            local_buffer_minutes=strategy.local_buffer_minutes,
        )
    
    def _generate_cache_key(self, date_obj: date, data_type: str) -> str:
        """
        Generate cache key for date and data type.
        
        Args:
            date_obj: Date object
            data_type: Data type (klines, trades, etc.)
            
        Returns:
            Cache key string
        """
        date_str = date_obj.isoformat()
        key = f"dataset_cache:{self.symbol}:{data_type}:{date_str}"
        return key
    
    async def get_day_data(
        self,
        date_obj: date,
        data_types: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for a specific day with multi-level cache lookup.
        
        Args:
            date_obj: Date to get data for
            data_types: List of data types to retrieve (default: all required)
            
        Returns:
            Dictionary mapping data type to DataFrame
        """
        if data_types is None:
            data_types = self.data_types
        
        result: Dict[str, pd.DataFrame] = {}
        
        for data_type in data_types:
            # Level 1: Check local cache
            if date_obj in self.local_cache and data_type in self.local_cache[date_obj]:
                result[data_type] = self.local_cache[date_obj][data_type].copy()
                self.stats["local_hits"] += 1
                logger.debug(
                    "cache_local_hit",
                    symbol=self.symbol,
                    date=date_obj.isoformat(),
                    data_type=data_type,
                )
                continue
            
            # Level 2: Check Redis cache
            cache_key = self._generate_cache_key(date_obj, data_type)
            cached_data = await self.redis_cache.get(cache_key)
            
            if cached_data is not None:
                result[data_type] = cached_data
                self.stats["redis_hits"] += 1
                
                # Store in local cache for faster access
                if date_obj not in self.local_cache:
                    self.local_cache[date_obj] = {}
                self.local_cache[date_obj][data_type] = cached_data.copy()
                self._update_local_cache_dates(date_obj)
                
                logger.debug(
                    "cache_redis_hit",
                    symbol=self.symbol,
                    date=date_obj.isoformat(),
                    data_type=data_type,
                )
                continue
            
            # Level 3: Read from Parquet
            data = await self._read_from_parquet(date_obj, data_type)
            result[data_type] = data
            self.stats["parquet_reads"] += 1
            
            # Store in both caches
            await self._store_in_cache(date_obj, data_type, data)
            
            logger.debug(
                "cache_parquet_read",
                symbol=self.symbol,
                date=date_obj.isoformat(),
                data_type=data_type,
                rows=len(data) if not data.empty else 0,
            )
        
        return result
    
    async def _read_from_parquet(
        self, date_obj: date, data_type: str
    ) -> pd.DataFrame:
        """
        Read data from Parquet storage.
        
        Args:
            date_obj: Date to read
            data_type: Data type to read
            
        Returns:
            DataFrame with data
        """
        date_str = date_obj.isoformat()
        
        try:
            if data_type == "klines":
                return await self.parquet_storage.read_klines(self.symbol, date_str)
            elif data_type == "trades":
                return await self.parquet_storage.read_trades(self.symbol, date_str)
            elif data_type == "orderbook_snapshots":
                return await self.parquet_storage.read_orderbook_snapshots(
                    self.symbol, date_str
                )
            elif data_type == "orderbook_deltas":
                return await self.parquet_storage.read_orderbook_deltas(
                    self.symbol, date_str
                )
            elif data_type == "ticker":
                return await self.parquet_storage.read_ticker(self.symbol, date_str)
            elif data_type == "funding":
                return await self.parquet_storage.read_funding(self.symbol, date_str)
            else:
                logger.warning(
                    "unknown_data_type",
                    data_type=data_type,
                    symbol=self.symbol,
                )
                return pd.DataFrame()
        except FileNotFoundError:
            logger.debug(
                "parquet_file_not_found",
                symbol=self.symbol,
                date=date_str,
                data_type=data_type,
            )
            return pd.DataFrame()
        except Exception as e:
            logger.error(
                "parquet_read_error",
                symbol=self.symbol,
                date=date_str,
                data_type=data_type,
                error=str(e),
                exc_info=True,
            )
            return pd.DataFrame()
    
    async def _store_in_cache(
        self, date_obj: date, data_type: str, data: pd.DataFrame
    ) -> None:
        """
        Store data in both Redis and local cache.
        
        Args:
            date_obj: Date
            data_type: Data type
            data: DataFrame to cache
        """
        # Store in Redis cache
        cache_key = self._generate_cache_key(date_obj, data_type)
        await self.redis_cache.set(
            cache_key, data, ttl=self.strategy.ttl_seconds
        )
        self.stats["cache_writes"] += 1
        
        # Store in local cache
        if date_obj not in self.local_cache:
            self.local_cache[date_obj] = {}
        self.local_cache[date_obj][data_type] = data.copy()
        self._update_local_cache_dates(date_obj)
        
        # Evict old local cache entries if needed
        self._evict_local_cache_if_needed()
    
    def _update_local_cache_dates(self, date_obj: date) -> None:
        """Update local cache dates list."""
        if date_obj not in self.local_cache_dates:
            self.local_cache_dates.append(date_obj)
            self.local_cache_dates.sort()
    
    def _evict_local_cache_if_needed(self) -> None:
        """Evict old entries from local cache based on strategy."""
        # Keep only dates within local buffer window
        if not self.local_cache_dates:
            return
        
        # Calculate how many days to keep based on local_buffer_minutes
        # For simplicity, keep at least 2 days in local cache
        max_local_days = max(2, self.strategy.local_buffer_minutes // (24 * 60) + 1)
        
        if len(self.local_cache_dates) > max_local_days:
            # Remove oldest dates
            dates_to_remove = self.local_cache_dates[:-max_local_days]
            for date_obj in dates_to_remove:
                if date_obj in self.local_cache:
                    del self.local_cache[date_obj]
                self.local_cache_dates.remove(date_obj)
            
            logger.debug(
                "local_cache_evicted",
                symbol=self.symbol,
                evicted_dates=[d.isoformat() for d in dates_to_remove],
                remaining_dates=[d.isoformat() for d in self.local_cache_dates],
            )
    
    async def prefetch_day(self, date_obj: date) -> None:
        """
        Prefetch data for a day in background.
        
        Args:
            date_obj: Date to prefetch
        """
        # Check if already cached
        cache_key = self._generate_cache_key(date_obj, self.data_types[0])
        if await self.redis_cache.exists(cache_key):
            logger.debug(
                "prefetch_skipped_already_cached",
                symbol=self.symbol,
                date=date_obj.isoformat(),
            )
            return
        
        # Prefetch all data types
        logger.info(
            "prefetch_started",
            symbol=self.symbol,
            date=date_obj.isoformat(),
        )
        
        for data_type in self.data_types:
            try:
                data = await self._read_from_parquet(date_obj, data_type)
                await self._store_in_cache(date_obj, data_type, data)
            except Exception as e:
                logger.warning(
                    "prefetch_error",
                    symbol=self.symbol,
                    date=date_obj.isoformat(),
                    data_type=data_type,
                    error=str(e),
                )
        
        logger.info(
            "prefetch_completed",
            symbol=self.symbol,
            date=date_obj.isoformat(),
        )
    
    async def evict_old_data(self, current_date: date) -> None:
        """
        Evict old data from Redis cache based on strategy.
        
        Args:
            current_date: Current date to calculate cutoff from
        """
        if self.strategy.cache_unit == CacheUnit.FULL_PERIOD:
            # Don't evict for full period strategy
            return
        
        # Calculate cutoff date based on cache size
        cutoff_date = current_date - timedelta(days=self.strategy.cache_size_days + 1)
        
        # Evict dates older than cutoff
        for data_type in self.data_types:
            date_to_evict = cutoff_date
            while date_to_evict < current_date - timedelta(days=self.strategy.cache_size_days):
                cache_key = self._generate_cache_key(date_to_evict, data_type)
                await self.redis_cache.delete(cache_key)
                date_to_evict += timedelta(days=1)
        
        logger.debug(
            "cache_evicted_old_data",
            symbol=self.symbol,
            cutoff_date=cutoff_date.isoformat(),
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_requests = (
            self.stats["local_hits"]
            + self.stats["redis_hits"]
            + self.stats["parquet_reads"]
        )
        
        hit_rate = (
            (self.stats["local_hits"] + self.stats["redis_hits"]) / total_requests
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
            "local_cache_dates": [d.isoformat() for d in self.local_cache_dates],
        }
    
    def clear_local_cache(self) -> None:
        """Clear local in-memory cache."""
        self.local_cache.clear()
        self.local_cache_dates.clear()
        logger.debug("local_cache_cleared", symbol=self.symbol)

