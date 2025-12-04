"""
Test fixtures for database connection mocking.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncpg


@pytest.fixture
def mock_db_pool():
    """Mock asyncpg connection pool."""
    pool = AsyncMock(spec=asyncpg.Pool)
    
    # Mock connection context manager
    mock_conn = AsyncMock(spec=asyncpg.Connection)
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.executemany = AsyncMock(return_value="INSERT 0 1")
    mock_conn.transaction = AsyncMock()
    
    async def acquire():
        return mock_conn
    
    pool.acquire = AsyncMock(side_effect=acquire)
    pool.release = AsyncMock()
    pool.close = AsyncMock()
    
    return pool


@pytest.fixture
def mock_db_connection():
    """Mock asyncpg connection."""
    conn = AsyncMock(spec=asyncpg.Connection)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.executemany = AsyncMock(return_value="INSERT 0 1")
    conn.transaction = AsyncMock()
    return conn


@pytest.fixture
def mock_db_transaction():
    """Mock database transaction."""
    transaction = AsyncMock()
    transaction.__aenter__ = AsyncMock(return_value=transaction)
    transaction.__aexit__ = AsyncMock(return_value=None)
    transaction.commit = AsyncMock()
    transaction.rollback = AsyncMock()
    return transaction

