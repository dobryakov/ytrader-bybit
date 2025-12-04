"""
RabbitMQ connection manager using aio-pika.
"""
import aio_pika
from typing import Optional
from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class MQConnectionManager:
    """RabbitMQ connection manager."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize MQ connection manager.
        
        Args:
            host: RabbitMQ host (defaults to config)
            port: RabbitMQ port (defaults to config)
            user: RabbitMQ user (defaults to config)
            password: RabbitMQ password (defaults to config)
        """
        self._host = host or config.rabbitmq_host
        self._port = port or config.rabbitmq_port
        self._user = user or config.rabbitmq_user
        self._password = password or config.rabbitmq_password
        self._connection: Optional[aio_pika.Connection] = None
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        if self._connection is None or self._connection.is_closed:
            self._connection = await aio_pika.connect_robust(
                host=self._host,
                port=self._port,
                login=self._user,
                password=self._password,
            )
            logger.info("Connected to RabbitMQ", host=self._host, port=self._port)
    
    async def close(self) -> None:
        """Close RabbitMQ connection."""
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("Closed RabbitMQ connection")
    
    async def get_channel(self) -> aio_pika.Channel:
        """
        Get a RabbitMQ channel.
        
        Returns:
            aio_pika.Channel: RabbitMQ channel
        """
        if self._connection is None or self._connection.is_closed:
            await self.connect()
        
        return await self._connection.channel()
    
    def is_connected(self) -> bool:
        """
        Check if connected to RabbitMQ.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._connection is not None and not self._connection.is_closed
    
    @property
    def connection(self) -> Optional[aio_pika.Connection]:
        """Get the connection (for testing)."""
        return self._connection

