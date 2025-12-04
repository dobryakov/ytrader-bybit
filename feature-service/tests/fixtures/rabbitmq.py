"""
Test fixtures for RabbitMQ connection mocking.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from aio_pika import Connection, Channel, Exchange, Queue


@pytest.fixture
def mock_rabbitmq_connection():
    """Mock aio-pika connection."""
    connection = AsyncMock(spec=Connection)
    connection.is_closed = False
    connection.close = AsyncMock()
    
    return connection


@pytest.fixture
def mock_rabbitmq_channel():
    """Mock aio-pika channel."""
    channel = AsyncMock(spec=Channel)
    channel.is_closed = False
    channel.close = AsyncMock()
    
    return channel


@pytest.fixture
def mock_rabbitmq_exchange():
    """Mock aio-pika exchange."""
    exchange = AsyncMock(spec=Exchange)
    exchange.publish = AsyncMock()
    return exchange


@pytest.fixture
def mock_rabbitmq_queue():
    """Mock aio-pika queue."""
    queue = AsyncMock(spec=Queue)
    queue.consume = AsyncMock()
    queue.cancel = AsyncMock()
    return queue


@pytest.fixture
def mock_rabbitmq_connection_and_channel():
    """Mock RabbitMQ connection and channel together."""
    connection = AsyncMock(spec=Connection)
    connection.is_closed = False
    connection.close = AsyncMock()
    
    channel = AsyncMock(spec=Channel)
    channel.is_closed = False
    channel.close = AsyncMock()
    
    async def get_channel():
        return channel
    
    connection.channel = AsyncMock(side_effect=get_channel)
    
    return connection, channel

