"""
Cache service abstraction with Redis (primary) and in-memory (fallback) implementations.
"""
import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import structlog

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class CacheService(ABC):
    """Abstract base class for cache services."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL (seconds)."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern (or all if pattern is None)."""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics (hit_count, miss_count, hit_rate, etc.)."""
        pass


class RedisCacheService(CacheService):
    """Redis-based cache service (PRIMARY/PRIORITY)."""
    
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
    ):
        """Initialize Redis cache service."""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not available. Install with: pip install redis[hiredis]")
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._connected = False
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=False,  # We handle binary data
            )
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            self._connected = True
            
            logger.info(
                "redis_cache_connected",
                host=self.host,
                port=self.port,
                db=self.db,
            )
            return True
        except Exception as e:
            logger.warning(
                "redis_cache_connection_failed",
                host=self.host,
                port=self.port,
                error=str(e),
            )
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.info("redis_cache_disconnected")
    
    def _is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None
    
    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is active."""
        if not self._is_connected():
            return await self.connect()
        try:
            await self._client.ping()
            return True
        except Exception:
            self._connected = False
            return await self.connect()
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value to bytes (using pickle for complex objects)."""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not await self._ensure_connected():
            return None
        
        try:
            data = await self._client.get(key)
            if data is None:
                self._miss_count += 1
                return None
            
            value = self._deserialize(data)
            self._hit_count += 1
            return value
        except Exception as e:
            logger.warning(
                "redis_cache_get_error",
                key=key,
                error=str(e),
            )
            self._miss_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache with optional TTL."""
        if not await self._ensure_connected():
            return False
        
        try:
            data = self._serialize(value)
            if ttl:
                await self._client.setex(key, ttl, data)
            else:
                await self._client.set(key, data)
            return True
        except Exception as e:
            logger.warning(
                "redis_cache_set_error",
                key=key,
                error=str(e),
            )
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not await self._ensure_connected():
            return False
        
        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(
                "redis_cache_delete_error",
                key=key,
                error=str(e),
            )
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not await self._ensure_connected():
            return False
        
        try:
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.warning(
                "redis_cache_exists_error",
                key=key,
                error=str(e),
            )
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern (or all if pattern is None)."""
        if not await self._ensure_connected():
            return 0
        
        try:
            if pattern:
                # Use SCAN to find matching keys
                deleted_count = 0
                async for key in self._client.scan_iter(match=pattern):
                    await self._client.delete(key)
                    deleted_count += 1
                return deleted_count
            else:
                # Clear all keys in current database
                await self._client.flushdb()
                return -1  # Indicates all keys cleared
        except Exception as e:
            logger.warning(
                "redis_cache_clear_error",
                pattern=pattern,
                error=str(e),
            )
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0.0
        
        # Get Redis info
        redis_info = {}
        if await self._ensure_connected():
            try:
                info = await self._client.info("memory")
                redis_info = {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "maxmemory": info.get("maxmemory", 0),
                    "maxmemory_policy": info.get("maxmemory_policy", "none"),
                }
            except Exception:
                pass
        
        return {
            "type": "redis",
            "connected": self._connected,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self._eviction_count,
            "total_requests": total_requests,
            "redis_info": redis_info,
        }


class InMemoryCacheService(CacheService):
    """In-memory cache service with LRU eviction (FALLBACK)."""
    
    def __init__(
        self,
        max_size_mb: int = 1024,
        max_entries: int = 10000,
    ):
        """Initialize in-memory cache service."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        
        # LRU cache: OrderedDict with (key, (value, size_bytes, expires_at))
        self._cache: OrderedDict[str, tuple[Any, int, Optional[float]]] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Current size in bytes
        self._current_size_bytes = 0
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Background task for TTL cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False
    
    async def start_cleanup_task(self) -> None:
        """Start background task for TTL cleanup."""
        if self._cleanup_running:
            return
        
        self._cleanup_running = True
        
        async def cleanup_loop():
            while self._cleanup_running:
                try:
                    await asyncio.sleep(60)  # Run every minute
                    await self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(
                        "memory_cache_cleanup_error",
                        error=str(e),
                    )
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        self._cleanup_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            now = time.time()
            expired_keys = [
                key for key, (_, _, expires_at) in self._cache.items()
                if expires_at is not None and expires_at < now
            ]
            for key in expired_keys:
                await self._remove_entry(key)
    
    def _serialize(self, value: Any) -> tuple[Any, int]:
        """Estimate size of value in bytes."""
        try:
            # Use pickle to estimate size
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return value, len(data)
        except Exception:
            # Fallback: rough estimate
            return value, 1024  # Default 1KB estimate
    
    def _deserialize(self, value: Any) -> Any:
        """Deserialize is not needed for in-memory cache (value is stored as-is)."""
        return value
    
    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and update size."""
        if key in self._cache:
            _, size_bytes, _ = self._cache.pop(key)
            self._current_size_bytes -= size_bytes
            self._eviction_count += 1
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entries until under limits."""
        while (
            (self._current_size_bytes > self.max_size_bytes or len(self._cache) > self.max_entries)
            and self._cache
        ):
            # Remove oldest entry (LRU)
            oldest_key = next(iter(self._cache))
            await self._remove_entry(oldest_key)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        async with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            value, size_bytes, expires_at = self._cache.pop(key)
            
            # Check expiration
            if expires_at is not None and expires_at < time.time():
                self._miss_count += 1
                self._current_size_bytes -= size_bytes
                return None
            
            # Move to end (most recently used)
            self._cache[key] = (value, size_bytes, expires_at)
            self._hit_count += 1
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache with optional TTL."""
        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                await self._remove_entry(key)
            
            # Serialize and get size
            serialized_value, size_bytes = self._serialize(value)
            
            # Check if entry is too large
            if size_bytes > self.max_size_bytes:
                logger.warning(
                    "memory_cache_entry_too_large",
                    key=key,
                    size_bytes=size_bytes,
                    max_size_bytes=self.max_size_bytes,
                )
                return False
            
            # Compute expiration time
            expires_at = None
            if ttl:
                expires_at = time.time() + ttl
            
            # Evict if needed
            await self._evict_lru()
            
            # Add entry
            self._cache[key] = (serialized_value, size_bytes, expires_at)
            self._current_size_bytes += size_bytes
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from in-memory cache."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache."""
        async with self._lock:
            if key not in self._cache:
                return False
            
            # Check expiration
            _, _, expires_at = self._cache[key]
            if expires_at is not None and expires_at < time.time():
                await self._remove_entry(key)
                return False
            
            return True
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern (or all if pattern is None)."""
        async with self._lock:
            if pattern is None:
                deleted_count = len(self._cache)
                self._cache.clear()
                self._current_size_bytes = 0
                return deleted_count
            
            # Pattern matching (simple prefix/suffix matching)
            deleted_count = 0
            keys_to_delete = []
            for key in self._cache.keys():
                if pattern.endswith("*") and key.startswith(pattern[:-1]):
                    keys_to_delete.append(key)
                elif pattern.startswith("*") and key.endswith(pattern[1:]):
                    keys_to_delete.append(key)
                elif "*" in pattern:
                    # Simple glob matching
                    import fnmatch
                    if fnmatch.fnmatch(key, pattern):
                        keys_to_delete.append(key)
                elif key == pattern:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                await self._remove_entry(key)
                deleted_count += 1
            
            return deleted_count
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "type": "memory",
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count,
                "total_requests": total_requests,
                "current_size_bytes": self._current_size_bytes,
                "current_size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_bytes": self.max_size_bytes,
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "current_entries": len(self._cache),
                "max_entries": self.max_entries,
            }


class CacheServiceFactory:
    """Factory for creating cache service instances with Redis-first strategy."""
    
    @staticmethod
    async def create(
        redis_host: str = "redis",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_max_connections: int = 10,
        redis_socket_timeout: int = 5,
        redis_socket_connect_timeout: int = 5,
        cache_redis_enabled: bool = True,
        cache_max_size_mb: int = 1024,
        cache_max_entries: int = 10000,
    ) -> CacheService:
        """
        Create cache service with Redis-first strategy.
        
        Attempts to connect to Redis first. If Redis is unavailable or disabled,
        falls back to in-memory cache.
        """
        # If Redis is disabled, use memory cache
        if not cache_redis_enabled:
            logger.info("redis_cache_disabled_using_memory_cache")
            memory_cache = InMemoryCacheService(
                max_size_mb=cache_max_size_mb,
                max_entries=cache_max_entries,
            )
            await memory_cache.start_cleanup_task()
            return memory_cache
        
        # Try Redis first
        if not REDIS_AVAILABLE:
            logger.warning(
                "redis_package_not_available_fallback_to_memory",
                note="Install redis package: pip install redis[hiredis]",
            )
            memory_cache = InMemoryCacheService(
                max_size_mb=cache_max_size_mb,
                max_entries=cache_max_entries,
            )
            await memory_cache.start_cleanup_task()
            return memory_cache
        
        redis_cache = RedisCacheService(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            max_connections=redis_max_connections,
            socket_timeout=redis_socket_timeout,
            socket_connect_timeout=redis_socket_connect_timeout,
        )
        
        # Attempt to connect
        connected = await redis_cache.connect()
        if connected:
            logger.info("redis_cache_connected_using_redis_as_primary")
            return redis_cache
        
        # Fallback to memory cache
        logger.warning(
            "redis_cache_unavailable_fallback_to_memory",
            host=redis_host,
            port=redis_port,
        )
        memory_cache = InMemoryCacheService(
            max_size_mb=cache_max_size_mb,
            max_entries=cache_max_entries,
        )
        await memory_cache.start_cleanup_task()
        return memory_cache

