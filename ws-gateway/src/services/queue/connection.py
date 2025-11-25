"""Base RabbitMQ connection using aio-pika."""

from typing import Optional
import aio_pika
from aio_pika import Connection, Channel

from ...config.settings import settings
from ...exceptions import QueueError
from ...config.logging import get_logger

logger = get_logger(__name__)


class QueueConnection:
    """Manages aio-pika connection and channel for RabbitMQ."""

    _connection: Optional[Connection] = None
    _channel: Optional[Channel] = None

    @classmethod
    async def create_connection(cls) -> Connection:
        """Create and return a RabbitMQ connection."""
        if cls._connection is None or cls._connection.is_closed:
            try:
                cls._connection = await aio_pika.connect_robust(
                    settings.rabbitmq_url,
                    client_properties={"connection_name": settings.ws_gateway_service_name},
                )
                logger.info(
                    "rabbitmq_connection_created",
                    host=settings.rabbitmq_host,
                    port=settings.rabbitmq_port,
                )
            except Exception as e:
                logger.error(
                    "rabbitmq_connection_failed",
                    error=str(e),
                    host=settings.rabbitmq_host,
                    port=settings.rabbitmq_port,
                )
                raise QueueError(f"Failed to create RabbitMQ connection: {e}") from e
        return cls._connection

    @classmethod
    async def get_connection(cls) -> Connection:
        """Get the existing connection, creating it if necessary."""
        if cls._connection is None or cls._connection.is_closed:
            await cls.create_connection()
        return cls._connection

    @classmethod
    async def get_channel(cls) -> Channel:
        """Get or create a channel."""
        if cls._channel is None or cls._channel.is_closed:
            connection = await cls.get_connection()
            try:
                cls._channel = await connection.channel()
                logger.info("rabbitmq_channel_created")
            except Exception as e:
                logger.error("rabbitmq_channel_failed", error=str(e))
                raise QueueError(f"Failed to create RabbitMQ channel: {e}") from e
        return cls._channel

    @classmethod
    async def close_connection(cls) -> None:
        """Close the RabbitMQ connection and channel."""
        if cls._channel is not None and not cls._channel.is_closed:
            await cls._channel.close()
            cls._channel = None
            logger.info("rabbitmq_channel_closed")

        if cls._connection is not None and not cls._connection.is_closed:
            await cls._connection.close()
            cls._connection = None
            logger.info("rabbitmq_connection_closed")

    @classmethod
    def is_connected(cls) -> bool:
        """Check if RabbitMQ connection is available and not closed."""
        if cls._connection is None:
            return False
        return not cls._connection.is_closed

