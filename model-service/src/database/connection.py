"""
Database connection pool using asyncpg.

Provides async PostgreSQL connection pooling for efficient database operations.
"""

import asyncpg
from typing import Optional
import asyncio

from ..config.settings import settings
from ..config.exceptions import DatabaseConnectionError
from ..config.logging import get_logger

logger = get_logger(__name__)


class DatabaseConnectionPool:
    """Manages PostgreSQL connection pool."""

    def __init__(self):
        """Initialize the connection pool."""
        self._pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()

    async def create_pool(
        self,
        min_size: int = 10,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
    ) -> asyncpg.Pool:
        """
        Create and configure connection pool.

        Args:
            min_size: Minimum number of connections in the pool
            max_size: Maximum number of connections in the pool
            max_queries: Maximum number of queries per connection before recycling
            max_inactive_connection_lifetime: Maximum seconds a connection can be idle

        Returns:
            Configured connection pool

        Raises:
            DatabaseConnectionError: If pool creation fails
        """
        async with self._lock:
            if self._pool is not None:
                logger.info("Connection pool already exists")
                return self._pool

            try:
                logger.info(
                    "Creating database connection pool",
                    host=settings.postgres_host,
                    port=settings.postgres_port,
                    database=settings.postgres_db,
                    min_size=min_size,
                    max_size=max_size,
                )
                self._pool = await asyncpg.create_pool(
                    host=settings.postgres_host,
                    port=settings.postgres_port,
                    user=settings.postgres_user,
                    password=settings.postgres_password,
                    database=settings.postgres_db,
                    min_size=min_size,
                    max_size=max_size,
                    max_queries=max_queries,
                    max_inactive_connection_lifetime=max_inactive_connection_lifetime,
                    command_timeout=60,  # 60 second timeout for queries
                )
                logger.info("Database connection pool created successfully")
                return self._pool
            except Exception as e:
                logger.error("Failed to create database connection pool", error=str(e), exc_info=True)
                raise DatabaseConnectionError(f"Failed to create connection pool: {e}") from e

    async def close_pool(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool is not None:
                await self._pool.close()
                self._pool = None
                logger.info("Database connection pool closed")

    async def get_pool(self) -> asyncpg.Pool:
        """
        Get or create the connection pool.

        Returns:
            Connection pool

        Raises:
            DatabaseConnectionError: If pool is not available
        """
        if self._pool is None:
            await self.create_pool()
        return self._pool

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query and return the result.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Query result

        Raises:
            DatabaseConnectionError: If execution fails
        """
        pool = await self.get_pool()
        try:
            return await pool.execute(query, *args)
        except Exception as e:
            logger.error("Database query execution failed", query=query, error=str(e), exc_info=True)
            raise DatabaseConnectionError(f"Query execution failed: {e}") from e

    async def fetch(self, query: str, *args) -> list:
        """
        Execute a query and fetch all results.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of records

        Raises:
            DatabaseConnectionError: If execution fails
        """
        pool = await self.get_pool()
        try:
            return await pool.fetch(query, *args)
        except Exception as e:
            logger.error("Database query fetch failed", query=query, error=str(e), exc_info=True)
            raise DatabaseConnectionError(f"Query fetch failed: {e}") from e

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Execute a query and fetch a single row.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single record or None

        Raises:
            DatabaseConnectionError: If execution fails
        """
        pool = await self.get_pool()
        try:
            return await pool.fetchrow(query, *args)
        except Exception as e:
            logger.error("Database query fetchrow failed", query=query, error=str(e), exc_info=True)
            raise DatabaseConnectionError(f"Query fetchrow failed: {e}") from e

    async def fetchval(self, query: str, *args) -> Optional[any]:
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single value or None

        Raises:
            DatabaseConnectionError: If execution fails
        """
        pool = await self.get_pool()
        try:
            return await pool.fetchval(query, *args)
        except Exception as e:
            logger.error("Database query fetchval failed", query=query, error=str(e), exc_info=True)
            raise DatabaseConnectionError(f"Query fetchval failed: {e}") from e

    @property
    def is_connected(self) -> bool:
        """Check if connection pool is available."""
        return self._pool is not None and not self._pool.is_closing()


# Global connection pool instance
db_pool = DatabaseConnectionPool()

