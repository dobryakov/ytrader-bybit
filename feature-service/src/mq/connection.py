"""
RabbitMQ connection manager using aio-pika.
"""
import asyncio
import aio_pika
import aiormq.exceptions
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
        self._heartbeat = config.rabbitmq_heartbeat
        self._reconnect_interval = config.rabbitmq_reconnect_interval
        self._connection: Optional[aio_pika.Connection] = None
        self._cached_channel: Optional[aio_pika.Channel] = None  # Cache for channel reuse
        self._connecting_lock: Optional[asyncio.Lock] = None  # Prevent concurrent connection attempts
        self._channel_lock: Optional[asyncio.Lock] = None  # Prevent concurrent channel creation attempts
    
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
                        heartbeat=self._heartbeat,  # Match RABBITMQ_HEARTBEAT from docker-compose.yml
                        reconnect_interval=self._reconnect_interval,  # Control reconnection frequency
                    ),
                    timeout=10.0,  # 10 second timeout for connection
                )
                logger.info(
                    "Connected to RabbitMQ",
                    host=self._host,
                    port=self._port,
                    heartbeat=self._heartbeat,
                    reconnect_interval=self._reconnect_interval,
                )
                # Reset cached channel when connection is recreated
                self._cached_channel = None
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
        # Close cached channel first to avoid callback errors
        if self._cached_channel is not None:
            try:
                if not self._cached_channel.is_closed:
                    await self._cached_channel.close()
                    logger.debug("Closed cached channel")
            except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError, Exception) as e:
                # Channel is in invalid state or already closed, just ignore
                logger.debug(
                    "Error closing cached channel (expected if channel is invalid)",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                self._cached_channel = None
        
        if self._connection is not None:
            try:
                # Safely check if connection is closed
                if not self._connection.is_closed:
                    await self._connection.close()
                    logger.info("Closed RabbitMQ connection")
            except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError):
                # Connection is in invalid state, just reset it
                logger.debug("Connection was in invalid state during close, resetting")
            finally:
                self._connection = None
    
    async def reset_connection(self) -> None:
        """
        Reset connection and channels, forcing reconnection on next use.
        
        This method should be used when connection errors occur to ensure
        clean state for reconnection. It closes channels first to avoid
        callback errors, then closes the connection.
        """
        # Close cached channel first to avoid callback errors
        if self._cached_channel is not None:
            try:
                if not self._cached_channel.is_closed:
                    await self._cached_channel.close()
                    logger.debug("Closed cached channel during reset")
            except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError, Exception) as e:
                # Channel is in invalid state or already closed, just ignore
                logger.debug(
                    "Error closing cached channel during reset (expected if channel is invalid)",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                self._cached_channel = None
        
        if self._connection is not None:
            try:
                # Safely check if connection is closed
                if not self._connection.is_closed:
                    await self._connection.close()
                    logger.debug("Closed connection during reset")
            except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError, Exception) as e:
                # Connection is in invalid state, just reset it
                logger.debug(
                    "Error closing connection during reset (expected if connection is invalid)",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                self._connection = None
    
    async def get_channel(self) -> aio_pika.Channel:
        """
        Get a RabbitMQ channel.
        
        Returns:
            aio_pika.Channel: RabbitMQ channel
        
        Raises:
            RuntimeError: If connection cannot be established
        """
        # Initialize lock if not already created
        if self._channel_lock is None:
            self._channel_lock = asyncio.Lock()
        
        # Use lock to prevent concurrent channel creation attempts
        async with self._channel_lock:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Safely check if connection is valid
                    connection_valid = False
                    if self._connection is not None:
                        try:
                            # Check if connection is closed - this may raise exception if connection is invalid
                            connection_valid = not self._connection.is_closed
                        except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError) as e:
                            # Connection is in invalid state
                            logger.debug(
                                "Connection check failed, will reconnect",
                                error=str(e),
                                error_type=type(e).__name__,
                            )
                            connection_valid = False
                    
                    # Ensure connection is valid
                    if not connection_valid:
                        await self.connect()
                    
                    # Check connection is still valid after connect()
                    if self._connection is None:
                        raise RuntimeError("Connection is None after connect()")
                    
                    # Try to reuse cached channel if it's still valid
                    if self._cached_channel is not None:
                        try:
                            # Check if cached channel is still valid
                            if not self._cached_channel.is_closed:
                                logger.debug("Reusing cached channel")
                                return self._cached_channel
                        except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError):
                            # Cached channel is invalid, reset it
                            logger.debug("Cached channel is invalid, creating new one")
                            self._cached_channel = None
                    
                    # For RobustConnection, channel() will handle reconnection automatically
                    # Just try to create channel - if connection is reconnecting, it will wait
                    # Use a reasonable timeout that accounts for potential reconnection delays
                    try:
                        channel = await asyncio.wait_for(
                            self._connection.channel(),
                            timeout=10.0,  # 10 second timeout for channel creation
                        )
                        # Cache the channel for reuse
                        self._cached_channel = channel
                        logger.debug("Created and cached new channel")
                        return channel
                    except asyncio.TimeoutError:
                        # Timeout occurred - connection might be in reconnecting state
                        # Log additional info and re-raise
                        logger.warning(
                            "Channel creation timeout - connection may be reconnecting",
                            connection_is_none=self._connection is None,
                            attempt=attempt + 1,
                        )
                        raise
                
                except (asyncio.TimeoutError, asyncio.CancelledError, StopAsyncIteration) as e:
                    # Channel creation was cancelled or timed out
                    connection_state = "unknown"
                    if self._connection is not None:
                        try:
                            connection_state = "closed" if self._connection.is_closed else "open"
                        except Exception:
                            connection_state = "invalid"
                    logger.warning(
                        "Channel creation failed, reconnecting",
                        error=str(e),
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        connection_state=connection_state,
                        connection_is_none=self._connection is None,
                    )
                    # Reset connection and retry - use reset_connection() to properly close channels first
                    await self.reset_connection()
                    
                    if attempt < max_retries - 1:
                        # Wait before retry - ensure we have event loop
                        try:
                            loop = asyncio.get_running_loop()
                            await asyncio.sleep(0.5 * (attempt + 1))
                        except RuntimeError:
                            # No event loop - this shouldn't happen in async context, but handle it gracefully
                            logger.warning(
                                "No event loop available for sleep, skipping delay",
                                attempt=attempt + 1,
                            )
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
                        # Reset connection and reconnect - use reset_connection() to properly close channels first
                        await self.reset_connection()
                        
                        if attempt < max_retries - 1:
                            # Wait before retry - ensure we have event loop
                            try:
                                loop = asyncio.get_running_loop()
                                await asyncio.sleep(0.5 * (attempt + 1))
                            except RuntimeError:
                                # No event loop - this shouldn't happen in async context, but handle it gracefully
                                logger.warning(
                                    "No event loop available for sleep, skipping delay",
                                    attempt=attempt + 1,
                                )
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
                
                except (AttributeError, aiormq.exceptions.ChannelInvalidStateError) as e:
                    # Connection object is invalid (None or missing attributes) or channel is in invalid state
                    logger.warning(
                        "Connection/channel invalid when getting channel, reconnecting",
                        error=str(e),
                        error_type=type(e).__name__,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                    )
                    # Reset connection and retry - use reset_connection() to properly close channels first
                    await self.reset_connection()
                    if attempt < max_retries - 1:
                        # Wait before retry - ensure we have event loop
                        try:
                            loop = asyncio.get_running_loop()
                            await asyncio.sleep(0.5 * (attempt + 1))
                        except RuntimeError:
                            # No event loop - this shouldn't happen in async context, but handle it gracefully
                            logger.warning(
                                "No event loop available for sleep, skipping delay",
                                attempt=attempt + 1,
                            )
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
        if self._connection is None:
            return False
        try:
            return not self._connection.is_closed
        except (RuntimeError, AttributeError, aiormq.exceptions.ChannelInvalidStateError):
            # Connection is in invalid state
            return False
    
    @property
    def connection(self) -> Optional[aio_pika.Connection]:
        """Get the connection (for testing)."""
        return self._connection

