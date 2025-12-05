"""
In-memory cache for feature vectors from Feature Service.

Provides caching with TTL, size limits, and thread-safe operations for concurrent access.
"""

from typing import Optional, Dict, Any
import time
import asyncio
from collections import OrderedDict

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.feature_vector import FeatureVector

logger = get_logger(__name__)


class FeatureCache:
    """
    Thread-safe in-memory cache for feature vectors.

    Features:
    - TTL-based expiration
    - LRU eviction when max size is reached
    - Thread-safe operations for concurrent access
    - Cache invalidation by symbol
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 30,
        max_size: int = 1000,
    ):
        """
        Initialize feature cache.

        Args:
            enabled: Whether caching is enabled
            ttl_seconds: Time-to-live for cached entries in seconds
            max_size: Maximum number of cached entries
        """
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

        if not self.enabled:
            logger.info("Feature cache is disabled")

    async def get(self, symbol: str, max_age_seconds: Optional[int] = None) -> Optional[FeatureVector]:
        """
        Get cached feature vector for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            max_age_seconds: Optional maximum age threshold (overrides TTL)

        Returns:
            Cached FeatureVector or None if not found/expired
        """
        if not self.enabled:
            return None

        async with self._lock:
            if symbol not in self._cache:
                logger.debug("Feature cache miss", symbol=symbol)
                return None

            entry = self._cache[symbol]
            cached_at = entry.get("cached_at", 0)
            age = time.time() - cached_at

            # Use max_age_seconds if provided, otherwise use TTL
            expiration_threshold = max_age_seconds if max_age_seconds is not None else self.ttl_seconds

            if age > expiration_threshold:
                # Entry expired
                del self._cache[symbol]
                logger.debug("Feature cache entry expired", symbol=symbol, age_seconds=age)
                return None

            # Move to end (LRU)
            self._cache.move_to_end(symbol)
            logger.debug("Feature cache hit", symbol=symbol, age_seconds=age)
            return entry.get("data")

    async def set(self, symbol: str, feature_vector: FeatureVector) -> None:
        """
        Cache feature vector for a symbol.

        Args:
            symbol: Trading pair symbol
            feature_vector: FeatureVector to cache
        """
        if not self.enabled:
            return

        async with self._lock:
            # Remove if exists to update position in LRU
            if symbol in self._cache:
                del self._cache[symbol]

            # Check if we need to evict oldest entry
            if len(self._cache) >= self.max_size:
                oldest_symbol = next(iter(self._cache))
                del self._cache[oldest_symbol]
                logger.debug("Feature cache evicted oldest entry", evicted_symbol=oldest_symbol, max_size=self.max_size)

            # Add new entry
            self._cache[symbol] = {
                "data": feature_vector,
                "cached_at": time.time(),
            }
            logger.debug("Feature vector cached", symbol=symbol, cache_size=len(self._cache))

    async def invalidate(self, symbol: str) -> None:
        """
        Invalidate cached feature vector for a symbol.

        Args:
            symbol: Trading pair symbol to invalidate
        """
        if not self.enabled:
            return

        async with self._lock:
            if symbol in self._cache:
                del self._cache[symbol]
                logger.debug("Feature cache invalidated", symbol=symbol)
            else:
                logger.debug("Feature cache invalidate called but entry not found", symbol=symbol)

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            size = len(self._cache)
            self._cache.clear()
            logger.info("Feature cache cleared", previous_size=size)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            current_time = time.time()
            entries = []
            for symbol, entry in self._cache.items():
                age = current_time - entry.get("cached_at", 0)
                entries.append({
                    "symbol": symbol,
                    "age_seconds": round(age, 2),
                })

            return {
                "enabled": self.enabled,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "entries": entries,
            }


# Global feature cache instance
feature_cache = FeatureCache(
    enabled=True,
    ttl_seconds=settings.feature_service_feature_cache_ttl_seconds,
    max_size=1000,  # Default max size for feature cache
)

