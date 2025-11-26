"""Connection manager for managing dual WebSocket connections (public and private)."""

from typing import Optional

from ...config.logging import get_logger
from ...models.subscription import Subscription
from .channel_types import get_endpoint_type_for_channel
from .connection import WebSocketConnection

logger = get_logger(__name__)


class ConnectionManager:
    """Manages dual WebSocket connections (public and private endpoints)."""

    def __init__(self):
        """Initialize connection manager."""
        self._public_connection: Optional[WebSocketConnection] = None
        self._private_connection: Optional[WebSocketConnection] = None

    async def get_connection_for_subscription(
        self, subscription: Subscription
    ) -> WebSocketConnection:
        """
        Get the appropriate connection for a subscription based on channel type.
        
        Args:
            subscription: Subscription object with channel_type
            
        Returns:
            WebSocketConnection instance (public or private)
        """
        endpoint_type = get_endpoint_type_for_channel(subscription.channel_type)
        
        if endpoint_type == "public":
            return await self.get_public_connection()
        else:
            return await self.get_private_connection()

    async def get_public_connection(self) -> WebSocketConnection:
        """
        Get or create public WebSocket connection with lazy initialization.
        
        Returns:
            Connected WebSocketConnection instance for public endpoint
        """
        # Check if connection exists and is connected
        if self._public_connection and self._public_connection.is_connected:
            return self._public_connection

        # Create new connection if needed
        if not self._public_connection:
            logger.info("connection_manager_creating_public_connection")
            self._public_connection = WebSocketConnection(endpoint_type="public")

        # Connect if not already connected
        if not self._public_connection.is_connected:
            logger.info("connection_manager_connecting_public")
            await self._public_connection.connect()

        return self._public_connection

    async def get_private_connection(self) -> WebSocketConnection:
        """
        Get or create private WebSocket connection with lazy initialization.
        
        Returns:
            Connected WebSocketConnection instance for private endpoint
        """
        # Check if connection exists and is connected
        if self._private_connection and self._private_connection.is_connected:
            return self._private_connection

        # Create new connection if needed
        if not self._private_connection:
            logger.info("connection_manager_creating_private_connection")
            self._private_connection = WebSocketConnection(endpoint_type="private")

        # Connect if not already connected
        if not self._private_connection.is_connected:
            logger.info("connection_manager_connecting_private")
            await self._private_connection.connect()

        return self._private_connection

    async def disconnect_all(self) -> None:
        """Disconnect all connections."""
        if self._public_connection:
            try:
                await self._public_connection.disconnect()
            except Exception as e:
                logger.warning(
                    "connection_manager_disconnect_public_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            self._public_connection = None

        if self._private_connection:
            try:
                await self._private_connection.disconnect()
            except Exception as e:
                logger.warning(
                    "connection_manager_disconnect_private_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            self._private_connection = None

        logger.info("connection_manager_all_disconnected")

    def get_public_connection_sync(self) -> Optional[WebSocketConnection]:
        """
        Get public connection synchronously (without connecting).
        
        Returns:
            WebSocketConnection instance or None if not created yet
        """
        return self._public_connection

    def get_private_connection_sync(self) -> Optional[WebSocketConnection]:
        """
        Get private connection synchronously (without connecting).
        
        Returns:
            WebSocketConnection instance or None if not created yet
        """
        return self._private_connection


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager

