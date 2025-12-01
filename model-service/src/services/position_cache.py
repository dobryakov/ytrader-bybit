"""
In-memory cache for position data from Position Manager.

Provides caching with TTL, size limits, and thread-safe operations for concurrent access.
"""

from typing import Optional, Dict, Any
import time
import asyncio
from collections import OrderedDict

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class PositionCache:
    """
    Thread-safe in-memory cache for position data.

    Features:
    - TTL-based expiration
    - LRU eviction when max size is reached
    - Thread-safe operations for concurrent access
    - Cache invalidation by asset
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 30,
        max_size: int = 1000,
    ):
        """
        Initialize position cache.

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
            logger.info("Position cache is disabled")

    async def get(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Get cached position data for an asset.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Cached position data or None if not found/expired
        """
        if not self.enabled:
            return None

        async with self._lock:
            if asset not in self._cache:
                logger.debug("Cache miss", asset=asset)
                return None

            entry = self._cache[asset]
            cached_at = entry.get("cached_at", 0)
            age = time.time() - cached_at

            if age > self.ttl_seconds:
                # Entry expired
                del self._cache[asset]
                logger.debug("Cache entry expired", asset=asset, age_seconds=age)
                return None

            # Move to end (LRU)
            self._cache.move_to_end(asset)
            logger.debug("Cache hit", asset=asset, age_seconds=age)
            return entry.get("data")

    async def set(self, asset: str, data: Dict[str, Any]) -> None:
        """
        Cache position data for an asset.

        Args:
            asset: Trading pair symbol
            data: Position data to cache
        """
        if not self.enabled:
            return

        async with self._lock:
            # Remove if exists to update position in LRU
            if asset in self._cache:
                del self._cache[asset]

            # Check if we need to evict oldest entry
            if len(self._cache) >= self.max_size:
                oldest_asset = next(iter(self._cache))
                del self._cache[oldest_asset]
                logger.debug("Cache evicted oldest entry", evicted_asset=oldest_asset, max_size=self.max_size)

            # Add new entry
            self._cache[asset] = {
                "data": data,
                "cached_at": time.time(),
            }
            logger.debug("Cached position data", asset=asset, cache_size=len(self._cache))

    async def invalidate(self, asset: str) -> None:
        """
        Invalidate cached position data for an asset.

        Args:
            asset: Trading pair symbol to invalidate
        """
        if not self.enabled:
            return

        async with self._lock:
            if asset in self._cache:
                del self._cache[asset]
                logger.debug("Cache invalidated", asset=asset)
            else:
                logger.debug("Cache invalidate called but entry not found", asset=asset)

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            size = len(self._cache)
            self._cache.clear()
            logger.info("Cache cleared", previous_size=size)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            {
                'enabled': bool,
                'size': int,
                'max_size': int,
                'ttl_seconds': int,
                'entries': [{'asset': str, 'age_seconds': float}, ...]
            }
        """
        async with self._lock:
            current_time = time.time()
            entries = []
            for asset, entry in self._cache.items():
                age = current_time - entry.get("cached_at", 0)
                entries.append({
                    "asset": asset,
                    "age_seconds": round(age, 2),
                })

            return {
                "enabled": self.enabled,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "entries": entries,
            }


# Global position cache instance
position_cache = PositionCache(
    enabled=settings.position_cache_enabled,
    ttl_seconds=settings.position_cache_ttl_seconds,
    max_size=settings.position_cache_max_size,
)

