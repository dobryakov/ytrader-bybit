"""
RabbitMQ connection manager using aio-pika.

Provides async connection management and channel pooling for message queue operations.
"""

import aio_pika
from aio_pika import Connection, Channel, Exchange, Queue
from typing import Optional, Dict
import asyncio

from .settings import settings
from .exceptions import MessageQueueConnectionError, MessageQueueError
from .logging import get_logger

logger = get_logger(__name__)


class RabbitMQConnectionManager:
    """Manages RabbitMQ connections and channels."""

    def __init__(self):
        """Initialize the connection manager."""
        self._connection: Optional[Connection] = None
        self._channel: Optional[Channel] = None
        self._exchanges: Dict[str, Exchange] = {}
        self._queues: Dict[str, Queue] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """
        Establish connection to RabbitMQ.

        Raises:
            MessageQueueConnectionError: If connection fails
        """
        if self._connection and not self._connection.is_closed:
            logger.info("RabbitMQ connection already established")
            return

        try:
            logger.info(
                "Connecting to RabbitMQ",
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
            )
            self._connection = await aio_pika.connect_robust(
                settings.rabbitmq_url,
                client_properties={"connection_name": settings.model_service_service_name},
            )
            self._channel = await self._connection.channel()
            logger.info("RabbitMQ connection established")
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", error=str(e), exc_info=True)
            raise MessageQueueConnectionError(f"Failed to connect to RabbitMQ: {e}") from e

    async def disconnect(self) -> None:
        """Close RabbitMQ connection and channel."""
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
            self._channel = None

        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            self._connection = None

        self._exchanges.clear()
        self._queues.clear()
        logger.info("RabbitMQ connection closed")

    async def get_channel(self) -> Channel:
        """
        Get or create a channel.

        Returns:
            RabbitMQ channel

        Raises:
            MessageQueueConnectionError: If connection is not established
        """
        if not self._connection or self._connection.is_closed:
            await self.connect()

        if not self._channel or self._channel.is_closed:
            self._channel = await self._connection.channel()

        return self._channel

    async def get_exchange(
        self, name: str, exchange_type: aio_pika.ExchangeType = aio_pika.ExchangeType.TOPIC
    ) -> Exchange:
        """
        Get or create an exchange.

        Args:
            name: Exchange name
            exchange_type: Exchange type (default: TOPIC)

        Returns:
            RabbitMQ exchange
        """
        if name not in self._exchanges:
            channel = await self.get_channel()
            self._exchanges[name] = await channel.declare_exchange(
                name, exchange_type, durable=True
            )
        return self._exchanges[name]

    async def get_queue(self, name: str, durable: bool = True) -> Queue:
        """
        Get or create a queue.

        Args:
            name: Queue name
            durable: Whether the queue should survive broker restarts

        Returns:
            RabbitMQ queue
        """
        if name not in self._queues:
            channel = await self.get_channel()
            self._queues[name] = await channel.declare_queue(name, durable=durable)
        return self._queues[name]

    @property
    def is_connected(self) -> bool:
        """Check if connection is established and open."""
        return (
            self._connection is not None
            and not self._connection.is_closed
            and self._channel is not None
            and not self._channel.is_closed
        )


# Global connection manager instance
rabbitmq_manager = RabbitMQConnectionManager()

