"""
Pytest configuration and fixtures.
"""

import pytest
import asyncio
from typing import AsyncGenerator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_pool():
    """Fixture for database connection pool."""
    from src.database.connection import db_pool
    await db_pool.create_pool()
    yield db_pool
    await db_pool.close_pool()


@pytest.fixture
async def rabbitmq_manager():
    """Fixture for RabbitMQ connection manager."""
    from src.config.rabbitmq import rabbitmq_manager
    await rabbitmq_manager.connect()
    yield rabbitmq_manager
    await rabbitmq_manager.disconnect()

