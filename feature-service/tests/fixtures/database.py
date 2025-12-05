"""
Test fixtures for database connection mocking.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager
import asyncpg


@pytest.fixture
def mock_db_pool():
    """Mock asyncpg connection pool."""
    # Mock connection
    mock_conn = AsyncMock(spec=asyncpg.Connection)
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_conn.executemany = AsyncMock(return_value="INSERT 0 1")
    
    # Mock transaction context manager
    @asynccontextmanager
    async def mock_transaction():
        yield mock_conn
    
    mock_conn.transaction = MagicMock(return_value=mock_transaction())
    
    # Mock acquire as async context manager
    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn
    
    # Create pool with proper acquire mock
    pool = MagicMock(spec=asyncpg.Pool)
    pool.acquire = MagicMock(return_value=mock_acquire())
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

