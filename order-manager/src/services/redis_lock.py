"""
Redis-based distributed locking for signal processing.

Provides distributed locks using Redis to prevent concurrent processing
of the same signal across multiple workers.
"""

from typing import Optional
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio import Redis

from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class RedisLock:
    """Redis-based distributed lock manager."""

    _client: Optional[Redis] = None
    _lock_timeout: int = 300  # 5 minutes default timeout

    @classmethod
    async def get_client(cls) -> Redis:
        """Get or create Redis client."""
        if cls._client is None or await cls._client.ping() is False:
            try:
                redis_url = f"redis://{settings.redis_host}:{settings.redis_port}"
                if settings.redis_password:
                    redis_url = f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}"
                
                cls._client = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # Keep bytes for lock values
                )
                # Test connection
                await cls._client.ping()
                logger.info(
                    "redis_connection_created",
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
            except Exception as e:
                logger.error(
                    "redis_connection_failed",
                    error=str(e),
                    host=settings.redis_host,
                    port=settings.redis_port,
                )
                raise
        return cls._client

    @classmethod
    async def close_client(cls) -> None:
        """Close Redis client connection."""
        if cls._client:
            await cls._client.aclose()
            cls._client = None
            logger.info("redis_connection_closed")

    @classmethod
    @asynccontextmanager
    async def acquire(
        cls,
        lock_key: str,
        timeout: Optional[int] = None,
        blocking_timeout: float = 0.1,
    ):
        """
        Acquire a distributed lock using Redis.
        
        Args:
            lock_key: Unique key for the lock
            timeout: Lock expiration time in seconds (default: 300)
            blocking_timeout: Maximum time to wait for lock acquisition (default: 0.1s)
        
        Yields:
            bool: True if lock acquired, False otherwise
        
        Example:
            async with RedisLock.acquire("signal:123") as acquired:
                if acquired:
                    # Process signal
                    pass
        """
        if timeout is None:
            timeout = cls._lock_timeout
        
        client = await cls.get_client()
        lock = client.lock(
            lock_key,
            timeout=timeout,
            blocking_timeout=blocking_timeout,
        )
        
        acquired = False
        try:
            acquired = await lock.acquire(blocking=False)
            if acquired:
                logger.debug(
                    "redis_lock_acquired",
                    lock_key=lock_key,
                    timeout=timeout,
                )
            else:
                logger.debug(
                    "redis_lock_busy",
                    lock_key=lock_key,
                )
            yield acquired
        finally:
            if acquired:
                try:
                    await lock.release()
                    logger.debug(
                        "redis_lock_released",
                        lock_key=lock_key,
                    )
                except Exception as e:
                    logger.warning(
                        "redis_lock_release_failed",
                        lock_key=lock_key,
                        error=str(e),
                    )

    @classmethod
    async def try_acquire(cls, lock_key: str, timeout: Optional[int] = None) -> bool:
        """
        Try to acquire a lock without blocking.
        
        Args:
            lock_key: Unique key for the lock
            timeout: Lock expiration time in seconds (default: 300)
        
        Returns:
            True if lock acquired, False otherwise
        """
        if timeout is None:
            timeout = cls._lock_timeout
        
        client = await cls.get_client()
        lock = client.lock(lock_key, timeout=timeout)
        
        try:
            acquired = await lock.acquire(blocking=False)
            if acquired:
                logger.debug(
                    "redis_lock_acquired",
                    lock_key=lock_key,
                    timeout=timeout,
                )
            return acquired
        except Exception as e:
            logger.error(
                "redis_lock_acquire_error",
                lock_key=lock_key,
                error=str(e),
            )
            return False

    @classmethod
    async def release(cls, lock_key: str) -> None:
        """
        Release a lock.
        
        Args:
            lock_key: Unique key for the lock
        """
        client = await cls.get_client()
        lock = client.lock(lock_key)
        
        try:
            await lock.release()
            logger.debug(
                "redis_lock_released",
                lock_key=lock_key,
            )
        except Exception as e:
            logger.warning(
                "redis_lock_release_failed",
                lock_key=lock_key,
                error=str(e),
            )


# Global Redis lock instance
redis_lock = RedisLock()

