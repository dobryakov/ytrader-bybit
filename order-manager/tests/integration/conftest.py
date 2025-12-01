"""Pytest fixtures for integration tests."""

import pytest
import asyncio
from typing import AsyncGenerator

from src.config.database import DatabaseConnection
from src.config.rabbitmq import RabbitMQConnection
from src.config.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_pool() -> AsyncGenerator:
    """Create database connection pool for each test function."""
    try:
        # Close any existing pool first to ensure clean state
        await DatabaseConnection.close_pool()
        # Create fresh pool for this test
        pool = await DatabaseConnection.create_pool()
        yield pool
    finally:
        # Close pool after test to avoid connection conflicts between tests
        await DatabaseConnection.close_pool()


@pytest.fixture(scope="session")
async def rabbitmq_connection() -> AsyncGenerator:
    """Create RabbitMQ connection for integration tests."""
    try:
        connection = await RabbitMQConnection.create_connection()
        yield connection
    finally:
        await RabbitMQConnection.close_connection()



