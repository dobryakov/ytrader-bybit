"""
RabbitMQ connection manager using aio-pika.
"""
import asyncio
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
        self._connecting_lock: Optional[asyncio.Lock] = None  # Prevent concurrent connection attempts
    
    async def connect(self) -> None:
        """Connect to RabbitMQ."""
        # Initialize lock if not already created
        if self._connecting_lock is None:
            self._connecting_lock = asyncio.Lock()
        
        # Use lock to prevent concurrent connection attempts
        async with self._connecting_lock:
            # Check if connection exists and is valid
            if self._connection is not None:
                try:
                    # For RobustConnection, if it's not closed, it will auto-reconnect
                    # So we should wait for it to reconnect instead of creating a new one
                    if not self._connection.is_closed:
                        # Connection exists and is not closed, wait a bit for it to be ready
                        # This handles the case when it's reconnecting
                        try:
                            # Try to check if connection is actually usable
                            _ = self._connection.is_closed
                            # If we got here, connection seems valid
                            # Wait a short time for robust connection to stabilize if reconnecting
                            await asyncio.sleep(0.1)
                            return
                        except (RuntimeError, AttributeError, asyncio.CancelledError):
                            # Connection is in invalid state or was cancelled
                            pass
                except (RuntimeError, AttributeError, asyncio.CancelledError):
                    # Connection object is invalid, reset it
                    pass
                
                # Connection is closed or invalid, close it properly
                if self._connection is not None:
                    try:
                        if not self._connection.is_closed:
                            await self._connection.close()
                    except (Exception, asyncio.CancelledError):
                        pass  # Ignore errors when closing invalid connection
                    self._connection = None
            
            # Create new connection with timeout and error handling
            try:
                self._connection = await asyncio.wait_for(
                    aio_pika.connect_robust(
                        host=self._host,
                        port=self._port,
                        login=self._user,
                        password=self._password,
                    ),
                    timeout=10.0,  # 10 second timeout for connection
                )
                logger.info("Connected to RabbitMQ", host=self._host, port=self._port)
            except (asyncio.TimeoutError, asyncio.CancelledError, StopAsyncIteration) as e:
                # Connection attempt was cancelled or timed out
                logger.warning(
                    "Connection attempt failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._connection = None
                raise RuntimeError(f"Failed to connect to RabbitMQ: {e}") from e
            except Exception as e:
                logger.error(
                    "Failed to connect to RabbitMQ",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                self._connection = None
                raise
    
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
        
        Raises:
            RuntimeError: If connection cannot be established
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure connection is valid
                if self._connection is None or self._connection.is_closed:
                    await self.connect()
                
                # Check connection is still valid after connect()
                if self._connection is None:
                    raise RuntimeError("Connection is None after connect()")
                
                # Try to get channel with timeout
                try:
                    return await asyncio.wait_for(
                        self._connection.channel(),
                        timeout=5.0,  # 5 second timeout for channel creation
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError, StopAsyncIteration) as e:
                    # Channel creation was cancelled or timed out
                    logger.warning(
                        "Channel creation failed, reconnecting",
                        error=str(e),
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    # Reset connection and retry
                    if self._connection is not None:
                        try:
                            if not self._connection.is_closed:
                                await self._connection.close()
                        except Exception:
                            pass
                        self._connection = None
                    
                    if attempt < max_retries - 1:
                        # Wait before retry
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(f"Failed to get channel after {max_retries} attempts: {e}") from e
                except AttributeError as e:
                    # Connection object is invalid (None or missing attributes)
                    logger.warning(
                        "Connection object invalid when getting channel, reconnecting",
                        error=str(e),
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    # Reset connection and retry
                    self._connection = None
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(f"Failed to get channel after {max_retries} attempts: {e}") from e
                
            except RuntimeError as e:
                # Connection was not opened or is invalid
                if "Connection was not opened" in str(e) or "not opened" in str(e).lower() or "Connection is None" in str(e):
                    logger.warning(
                        "Connection invalid when getting channel, reconnecting",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    # Reset connection and reconnect
                    if self._connection is not None:
                        try:
                            if not self._connection.is_closed:
                                await self._connection.close()
                        except Exception:
                            pass
                        self._connection = None
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        raise RuntimeError(f"Failed to get channel after {max_retries} attempts: {e}") from e
                else:
                    # Re-raise other RuntimeErrors
                    raise
            except (asyncio.CancelledError, StopAsyncIteration) as e:
                # Task was cancelled or connection closed during operation
                logger.warning(
                    "Operation cancelled during channel creation",
                    error=str(e),
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                # Re-raise CancelledError immediately - don't retry on cancellation
                raise
            except AttributeError as e:
                # Connection object is invalid (None or missing attributes)
                logger.warning(
                    "Connection object invalid when getting channel, reconnecting",
                    error=str(e),
                    error_type=type(e).__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                # Reset connection and retry
                self._connection = None
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Failed to get channel after {max_retries} attempts: {e}") from e
        
        # Should not reach here, but just in case
        raise RuntimeError("Failed to get channel: max retries exceeded")
    
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

