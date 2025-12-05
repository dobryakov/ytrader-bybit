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
    
    async def create_dataset(self, dataset_data: dict) -> str:
        """
        Create a new dataset record.
        
        Args:
            dataset_data: Dataset data dictionary
            
        Returns:
            Dataset ID (UUID as string)
        """
        async with self.transaction() as conn:
            dataset_id = await conn.fetchval(
                """
                INSERT INTO datasets (
                    symbol, status, split_strategy,
                    train_period_start, train_period_end,
                    validation_period_start, validation_period_end,
                    test_period_start, test_period_end,
                    walk_forward_config, target_config,
                    feature_registry_version, output_format
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
                """,
                dataset_data["symbol"],
                dataset_data["status"],
                dataset_data["split_strategy"],
                dataset_data.get("train_period_start"),
                dataset_data.get("train_period_end"),
                dataset_data.get("validation_period_start"),
                dataset_data.get("validation_period_end"),
                dataset_data.get("test_period_start"),
                dataset_data.get("test_period_end"),
                dataset_data.get("walk_forward_config"),
                dataset_data["target_config"],
                dataset_data["feature_registry_version"],
                dataset_data.get("output_format", "parquet"),
            )
            return str(dataset_id)
    
    async def get_dataset(self, dataset_id: str) -> Optional[dict]:
        """
        Get dataset by ID.
        
        Args:
            dataset_id: Dataset ID (UUID string)
            
        Returns:
            Dataset data dictionary or None if not found
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM datasets WHERE id = $1
                """,
                dataset_id,
            )
            if row is None:
                return None
            return dict(row)
    
    async def update_dataset(
        self,
        dataset_id: str,
        updates: dict,
    ) -> None:
        """
        Update dataset record.
        
        Args:
            dataset_id: Dataset ID (UUID string)
            updates: Dictionary of fields to update
        """
        if not updates:
            return
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        param_num = 1
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = ${param_num}")
            values.append(value)
            param_num += 1
        
        values.append(dataset_id)  # WHERE clause parameter
        
        async with self.transaction() as conn:
            await conn.execute(
                f"""
                UPDATE datasets
                SET {', '.join(set_clauses)}
                WHERE id = ${param_num}
                """,
                *values,
            )
    
    async def list_datasets(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """
        List datasets with optional filters.
        
        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)
            limit: Maximum number of results
            
        Returns:
            List of dataset dictionaries
        """
        async with self.get_connection() as conn:
            query = "SELECT * FROM datasets WHERE 1=1"
            params = []
            param_num = 1
            
            if symbol:
                query += f" AND symbol = ${param_num}"
                params.append(symbol)
                param_num += 1
            
            if status:
                query += f" AND status = ${param_num}"
                params.append(status)
                param_num += 1
            
            query += f" ORDER BY created_at DESC LIMIT ${param_num}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

