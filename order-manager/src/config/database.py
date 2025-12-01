"""Base database connection pool using asyncpg."""

import asyncpg
from typing import Optional

from .settings import settings
from ..exceptions import DatabaseError
from .logging import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """Manages asyncpg connection pool for PostgreSQL."""

    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def create_pool(cls) -> asyncpg.Pool:
        """Create and return a connection pool."""
        # Check if pool exists and is not closed
        if cls._pool is not None:
            try:
                # Try to acquire a connection to verify pool is still open
                async with cls._pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return cls._pool
            except Exception:
                # Pool is closed or invalid, reset it
                cls._pool = None
                logger.warning("database_pool_closed_or_invalid", message="Pool was closed, recreating")
        
        # Create new pool
        try:
            cls._pool = await asyncpg.create_pool(
                settings.database_url_async,
                min_size=5,
                max_size=20,
                command_timeout=60,
            )
            logger.info(
                "database_pool_created",
                host=settings.postgres_host,
                port=settings.postgres_port,
                database=settings.postgres_db,
            )
        except Exception as e:
            logger.error(
                "database_pool_creation_failed",
                error=str(e),
                host=settings.postgres_host,
                port=settings.postgres_port,
            )
            raise DatabaseError(f"Failed to create database pool: {e}") from e
        return cls._pool

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get the existing connection pool, creating it if necessary."""
        if cls._pool is None:
            await cls.create_pool()
        return cls._pool

    @classmethod
    async def close_pool(cls) -> None:
        """Close the connection pool."""
        if cls._pool is not None:
            await cls._pool.close()
            cls._pool = None
            logger.info("database_pool_closed")

    @classmethod
    async def execute(cls, query: str, *args) -> str:
        """Execute a query and return the result."""
        pool = await cls.get_pool()
        try:
            return await pool.execute(query, *args)
        except Exception as e:
            logger.error("database_execute_failed", query=query, error=str(e))
            raise DatabaseError(f"Database query failed: {e}") from e

    @classmethod
    async def fetch(cls, query: str, *args) -> list:
        """Fetch rows from a query."""
        pool = await cls.get_pool()
        try:
            return await pool.fetch(query, *args)
        except Exception as e:
            logger.error("database_fetch_failed", query=query, error=str(e))
            raise DatabaseError(f"Database fetch failed: {e}") from e

    @classmethod
    async def fetchrow(cls, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row from a query."""
        pool = await cls.get_pool()
        try:
            return await pool.fetchrow(query, *args)
        except Exception as e:
            logger.error("database_fetchrow_failed", query=query, error=str(e))
            raise DatabaseError(f"Database fetchrow failed: {e}") from e

    @classmethod
    def is_connected(cls) -> bool:
        """Check if database connection pool is available and not closed."""
        if cls._pool is None:
            return False
        # Check if pool is closed (asyncpg pools don't have is_closed, so we check if it's None)
        return cls._pool is not None

