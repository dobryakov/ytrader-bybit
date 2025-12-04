"""
Integration tests for database connection pool.
"""
import pytest
from unittest.mock import AsyncMock, patch
import asyncpg


@pytest.mark.asyncio
class TestMetadataStorage:
    """Integration tests for metadata storage (database connection pool)."""
    
    async def test_database_pool_connects(self, mock_db_pool):
        """Test that database pool connects successfully."""
        from src.storage.metadata_storage import MetadataStorage
        
        storage = MetadataStorage(pool=mock_db_pool)
        
        # Test connection
        async with storage.get_connection() as conn:
            assert conn is not None
    
    async def test_database_pool_executes_queries(self, mock_db_pool):
        """Test that database pool executes queries."""
        from src.storage.metadata_storage import MetadataStorage
        
        storage = MetadataStorage(pool=mock_db_pool)
        
        # Mock query result
        mock_db_pool.acquire.return_value.__aenter__.return_value.fetch.return_value = [
            {"id": "123", "symbol": "BTCUSDT"}
        ]
        
        async with storage.get_connection() as conn:
            result = await conn.fetch("SELECT * FROM datasets WHERE symbol = $1", "BTCUSDT")
            assert result is not None
    
    async def test_database_pool_handles_connection_errors(self):
        """Test that database pool handles connection errors."""
        from src.storage.metadata_storage import MetadataStorage
        
        # Create pool that raises connection error
        mock_pool = AsyncMock()
        mock_pool.acquire.side_effect = asyncpg.PostgresConnectionError("Connection failed")
        
        storage = MetadataStorage(pool=mock_pool)
        
        with pytest.raises(asyncpg.PostgresConnectionError):
            async with storage.get_connection() as conn:
                pass
    
    async def test_database_pool_releases_connections(self, mock_db_pool):
        """Test that database pool releases connections properly."""
        from src.storage.metadata_storage import MetadataStorage
        
        storage = MetadataStorage(pool=mock_db_pool)
        
        async with storage.get_connection() as conn:
            pass
        
        # Verify release was called
        mock_db_pool.release.assert_called()
    
    async def test_database_pool_transactions(self, mock_db_pool):
        """Test that database pool supports transactions."""
        from src.storage.metadata_storage import MetadataStorage
        
        storage = MetadataStorage(pool=mock_db_pool)
        
        async with storage.transaction() as tx:
            # Execute operations within transaction
            await tx.execute("INSERT INTO datasets (symbol) VALUES ($1)", "BTCUSDT")
        
        # Verify transaction was committed
        # (adjust based on actual implementation)

