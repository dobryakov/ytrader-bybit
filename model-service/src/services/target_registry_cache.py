"""
In-memory cache for target registry configurations.

Provides caching with TTL, size limits, and thread-safe operations for concurrent access.
"""

from typing import Optional, Dict, Any
import time
import asyncio
from collections import OrderedDict

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class TargetRegistryCache:
    """
    Thread-safe in-memory cache for target registry configurations.

    Features:
    - TTL-based expiration (configurations change rarely, so longer TTL)
    - LRU eviction when max size is reached
    - Thread-safe operations for concurrent access
    - Cache invalidation by version
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 3600,  # 1 hour - configs change rarely
        max_size: int = 50,  # Small cache - only a few versions
    ):
        """
        Initialize target registry cache.

        Args:
            enabled: Whether caching is enabled
            ttl_seconds: Time-to-live for cached entries in seconds (default: 3600 = 1 hour)
            max_size: Maximum number of cached entries (default: 50)
        """
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

        if not self.enabled:
            logger.info("Target registry cache is disabled")

    async def get(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get cached target registry config for a version.

        Args:
            version: Target registry version identifier

        Returns:
            Cached target registry config or None if not found/expired
        """
        if not self.enabled:
            return None

        async with self._lock:
            if version not in self._cache:
                logger.debug("Target registry cache miss", version=version)
                return None

            entry = self._cache[version]
            cached_at = entry.get("cached_at", 0)
            age = time.time() - cached_at

            if age > self.ttl_seconds:
                # Entry expired
                del self._cache[version]
                logger.debug("Target registry cache entry expired", version=version, age_seconds=age)
                return None

            # Move to end (LRU)
            self._cache.move_to_end(version)
            logger.debug("Target registry cache hit", version=version, age_seconds=age)
            return entry.get("data")

    async def set(self, version: str, config: Dict[str, Any]) -> None:
        """
        Cache target registry config for a version.

        Args:
            version: Target registry version identifier
            config: Target registry configuration to cache
        """
        if not self.enabled:
            return

        async with self._lock:
            # Remove if exists to update position in LRU
            if version in self._cache:
                del self._cache[version]

            # Check if we need to evict oldest entry
            if len(self._cache) >= self.max_size:
                oldest_version = next(iter(self._cache))
                del self._cache[oldest_version]
                logger.debug("Target registry cache evicted oldest entry", evicted_version=oldest_version, max_size=self.max_size)

            # Add new entry
            self._cache[version] = {
                "data": config,
                "cached_at": time.time(),
            }
            logger.debug("Cached target registry config", version=version, cache_size=len(self._cache))

    async def invalidate(self, version: str) -> None:
        """
        Invalidate cached target registry config for a version.

        Args:
            version: Target registry version identifier to invalidate
        """
        if not self.enabled:
            return

        async with self._lock:
            if version in self._cache:
                del self._cache[version]
                logger.debug("Target registry cache invalidated", version=version)
            else:
                logger.debug("Target registry cache invalidate called but entry not found", version=version)

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            size = len(self._cache)
            self._cache.clear()
            logger.info("Target registry cache cleared", previous_size=size)


# Global target registry cache instance
target_registry_cache = TargetRegistryCache(enabled=True, ttl_seconds=3600, max_size=50)

