"""
Shared test fixtures for Feature Service tests.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_logger():
    """Mock structured logger."""
    return MagicMock()


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    return pool


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock RabbitMQ connection."""
    connection = AsyncMock()
    channel = AsyncMock()
    connection.channel = AsyncMock(return_value=channel)
    return connection, channel


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for ws-gateway API."""
    client = AsyncMock()
    return client

