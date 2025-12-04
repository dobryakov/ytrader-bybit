"""
Metadata storage using asyncpg connection pool.
"""
import asyncpg
from typing import Optional, AsyncContextManager
from contextlib import asynccontextmanager
from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class MetadataStorage:
    """PostgreSQL metadata storage using asyncpg connection pool."""
    
    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """
        Initialize metadata storage.
        
        Args:
            pool: Optional asyncpg pool (for testing). If None, creates new pool.
        """
        self._pool = pool
        self._own_pool = pool is None
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_db,
                user=config.postgres_user,
                password=config.postgres_password,
                min_size=2,
                max_size=10,
            )
            logger.info("Database connection pool initialized")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool and self._own_pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[asyncpg.Connection]:
        """
        Get a database connection from the pool.
        
        Yields:
            asyncpg.Connection: Database connection
        """
        if self._pool is None:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[asyncpg.Connection]:
        """
        Get a database connection with transaction.
        
        Yields:
            asyncpg.Connection: Database connection in transaction
        """
        async with self.get_connection() as connection:
            async with connection.transaction():
                yield connection
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the connection pool (for testing)."""
        return self._pool

