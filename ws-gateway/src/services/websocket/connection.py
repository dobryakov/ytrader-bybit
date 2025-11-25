"""WebSocket connection manager with websockets library."""

import asyncio
import json
from datetime import datetime
from typing import Optional
from collections import deque

import websockets
from websockets.client import WebSocketClientProtocol

from ...config.logging import get_logger
from ...config.settings import settings
from ...models.websocket_state import ConnectionStatus, WebSocketState
from .auth import generate_auth_message, validate_auth_response

logger = get_logger(__name__)

# Global storage for recent messages (for testing/viewing)
_recent_messages: deque = deque(maxlen=100)


def get_recent_messages():
    """Get recent WebSocket messages (for testing)."""
    return list(_recent_messages)


class WebSocketConnection:
    """Manages WebSocket connection to Bybit exchange."""

    def __init__(self):
        """Initialize WebSocket connection manager."""
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._state = WebSocketState(
            environment=settings.bybit_environment,
            status=ConnectionStatus.DISCONNECTED,
        )
        self._receive_task: Optional[asyncio.Task] = None
        self._connected_event = asyncio.Event()
        self._disconnection_callback = None

    @property
    def state(self) -> WebSocketState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        # Primary check: status must be CONNECTED and websocket must exist
        if self._websocket is None or self._state.status != ConnectionStatus.CONNECTED:
            return False
        # Secondary check: verify websocket is not closed (if attribute exists)
        # Some websockets library versions may not have 'closed' attribute
        try:
            closed = getattr(self._websocket, 'closed', None)
            if closed is not None:
                return not closed
        except (AttributeError, TypeError):
            pass
        # If we can't check closed status, trust the CONNECTED status
        return True

    async def connect(self) -> None:
        """
        Establish WebSocket connection to Bybit and authenticate.

        Raises:
            ConnectionError: If connection or authentication fails
        """
        if self.is_connected:
            logger.warning("websocket_already_connected")
            return

        self._state.status = ConnectionStatus.CONNECTING
        logger.info(
            "websocket_connecting",
            url=settings.bybit_ws_url,
            environment=settings.bybit_environment,
        )

        try:
            # Connect to Bybit WebSocket
            self._websocket = await websockets.connect(
                settings.bybit_ws_url,
                ping_interval=None,  # We'll handle ping/pong manually
                ping_timeout=None,
            )

            logger.info("websocket_connected", url=settings.bybit_ws_url)

            # Authenticate
            await self._authenticate()

            # Update state
            self._state.status = ConnectionStatus.CONNECTED
            self._state.connected_at = datetime.now()
            self._state.last_error = None
            self._connected_event.set()

            logger.info(
                "websocket_authenticated",
                connection_id=str(self._state.connection_id),
            )

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            self._state.status = ConnectionStatus.DISCONNECTED
            self._state.last_error = str(e)
            logger.error(
                "websocket_connection_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            raise ConnectionError(f"Failed to connect to Bybit WebSocket: {e}") from e

    async def _authenticate(self) -> None:
        """
        Authenticate with Bybit WebSocket.

        Raises:
            ConnectionError: If authentication fails
        """
        if not self._websocket:
            raise ConnectionError("WebSocket not connected")

        # Generate and send authentication message
        auth_message = generate_auth_message()
        await self._websocket.send(json.dumps(auth_message))

        logger.debug("websocket_auth_sent", api_key=settings.bybit_api_key[:8] + "...")

        # Wait for authentication response (with timeout)
        try:
            response_str = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            response = json.loads(response_str)

            if validate_auth_response(response):
                logger.info("websocket_auth_success")
            else:
                error_msg = response.get("ret_msg", "Unknown authentication error")
                logger.error("websocket_auth_failed", error=error_msg)
                raise ConnectionError(f"Authentication failed: {error_msg}")

        except asyncio.TimeoutError:
            logger.error("websocket_auth_timeout")
            raise ConnectionError("Authentication timeout") from None
        except json.JSONDecodeError as e:
            logger.error("websocket_auth_invalid_response", error=str(e))
            raise ConnectionError(f"Invalid authentication response: {e}") from e

    async def _receive_messages(self) -> None:
        """Continuously receive and log messages from WebSocket."""
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    # Log subscription confirmations and data messages at INFO level for visibility
                    topic = data.get("topic", data.get("op", "unknown"))
                    if data.get("op") == "subscribe" or topic != "unknown":
                        logger.info(
                            "websocket_message_received",
                            message_type=topic,
                            op=data.get("op"),
                            data=data,
                        )
                    else:
                        logger.debug(
                            "websocket_message_received",
                            message_type=topic,
                            data=data,
                        )
                    
                    # Store message for viewing via API (testing)
                    _recent_messages.append({
                        "timestamp": data.get("ts", datetime.now().isoformat()),
                        "topic": data.get("topic", ""),
                        "op": data.get("op", ""),
                        "type": data.get("type", ""),
                        "data": data
                    })
                    
                    # Message processing will be handled by event processor (Phase 4)
                except json.JSONDecodeError as e:
                    logger.warning("websocket_invalid_json", error=str(e), raw_message=message[:100])
                except Exception as e:
                    logger.error(
                        "websocket_message_processing_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.warning("websocket_connection_closed")
            self._state.status = ConnectionStatus.DISCONNECTED
            self._connected_event.clear()
            # Trigger reconnection (will be handled by reconnection manager if registered)
            await self._handle_disconnection()
        except Exception as e:
            logger.error(
                "websocket_receive_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._state.status = ConnectionStatus.DISCONNECTED
            self._connected_event.clear()
            # Trigger reconnection (will be handled by reconnection manager if registered)
            await self._handle_disconnection()

    def set_disconnection_callback(self, callback):
        """Set callback to be called on disconnection."""
        self._disconnection_callback = callback

    async def _handle_disconnection(self) -> None:
        """Handle disconnection event (triggers reconnection manager if set)."""
        if self._disconnection_callback:
            await self._disconnection_callback()

    async def send(self, message: dict) -> None:
        """
        Send a message through the WebSocket connection.

        Args:
            message: Dictionary to send as JSON

        Raises:
            ConnectionError: If WebSocket is not connected
        """
        if not self.is_connected or not self._websocket:
            raise ConnectionError("WebSocket is not connected")

        try:
            await self._websocket.send(json.dumps(message))
            logger.debug("websocket_message_sent", message=message)
        except Exception as e:
            logger.error("websocket_send_error", error=str(e), error_type=type(e).__name__)
            raise ConnectionError(f"Failed to send message: {e}") from e

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._websocket:
            try:
                if not getattr(self._websocket, 'closed', False):
                    await self._websocket.close()
            except AttributeError:
                # If closed attribute doesn't exist, try to close anyway
                try:
                    await self._websocket.close()
                except Exception:
                    pass
            logger.info("websocket_disconnected")

        self._websocket = None
        self._state.status = ConnectionStatus.DISCONNECTED
        self._connected_event.clear()

    async def wait_connected(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for connection to be established.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# Global WebSocket connection instance
_connection: Optional[WebSocketConnection] = None


def get_connection() -> WebSocketConnection:
    """Get or create global WebSocket connection instance."""
    global _connection
    if _connection is None:
        _connection = WebSocketConnection()
    return _connection

