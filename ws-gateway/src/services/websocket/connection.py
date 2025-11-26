"""WebSocket connection manager with websockets library."""

import asyncio
import json
from collections import deque
from datetime import datetime
from typing import Optional
from uuid import uuid4

import websockets
from websockets.client import WebSocketClientProtocol

from typing import Literal

from ...config.logging import get_logger
from ...config.settings import settings
from ...models.websocket_state import ConnectionStatus, WebSocketState
from ...utils.tracing import generate_trace_id, get_or_create_trace_id, set_trace_id
from ..database.subscription_repository import SubscriptionRepository
from ..subscription.subscription_service import SubscriptionService
from .auth import generate_auth_message, validate_auth_response, validate_credentials
from .event_parser import parse_events_from_message
from .event_processor import process_events

logger = get_logger(__name__)

# Global storage for recent messages (for testing/viewing)
_recent_messages: deque = deque(maxlen=100)


def get_recent_messages():
    """Get recent WebSocket messages (for testing)."""
    return list(_recent_messages)


class WebSocketConnection:
    """Manages WebSocket connection to Bybit exchange."""

    def __init__(self, endpoint_type: Literal["public", "private"] = "private"):
        """
        Initialize WebSocket connection manager.
        
        Args:
            endpoint_type: Type of endpoint to connect to - "public" for /v5/public 
                          (no auth) or "private" for /v5/private (auth required).
                          Defaults to "private" for backward compatibility.
        """
        self._endpoint_type = endpoint_type
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
        # Generate trace ID for this connection attempt
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        if self.is_connected:
            logger.warning(
                "websocket_already_connected",
                trace_id=trace_id,
            )
            return

        # Log state change with trace ID
        old_status = self._state.status
        self._state.status = ConnectionStatus.CONNECTING
        logger.info(
            "websocket_state_changed",
            old_status=old_status.value if hasattr(old_status, "value") else str(old_status),
            new_status=self._state.status.value,
            connection_id=str(self._state.connection_id),
            trace_id=trace_id,
        )

        # Get WebSocket URL based on endpoint type
        ws_url = self._get_ws_url()
        
        logger.info(
            "websocket_connecting",
            url=ws_url,
            endpoint_type=self._endpoint_type,
            environment=settings.bybit_environment,
            connection_id=str(self._state.connection_id),
            trace_id=trace_id,
        )

        # Validate credentials only for private endpoints (EC4: Handle authentication failures)
        if self._endpoint_type == "private" and not validate_credentials():
            error_msg = "Invalid or missing API credentials"
            logger.error(
                "websocket_connection_failed_credentials",
                error=error_msg,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )
            raise ConnectionError(error_msg)

        try:
            # Connect to Bybit WebSocket with timeout handling (EC8: Handle exchange endpoint timeouts)
            try:
                self._websocket = await asyncio.wait_for(
                    websockets.connect(
                        ws_url,
                        ping_interval=None,  # We'll handle ping/pong manually
                        ping_timeout=None,
                    ),
                    timeout=30.0,  # 30 second connection timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "websocket_connection_timeout",
                    url=ws_url,
                    timeout=30.0,
                    connection_id=str(self._state.connection_id),
                    trace_id=trace_id,
                )
                raise ConnectionError("WebSocket connection timeout after 30 seconds") from None

            logger.info(
                "websocket_connected",
                url=ws_url,
                endpoint_type=self._endpoint_type,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )

            # Authenticate (only for private endpoints)
            await self._authenticate()

            # Update state and log state change
            old_status = self._state.status
            self._state.status = ConnectionStatus.CONNECTED
            self._state.connected_at = datetime.now()
            self._state.last_error = None
            self._connected_event.set()

            logger.info(
                "websocket_state_changed",
                old_status=old_status.value if hasattr(old_status, "value") else str(old_status),
                new_status=self._state.status.value,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )

            logger.info(
                "websocket_authenticated",
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )

            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_messages())

        except Exception as e:
            old_status = self._state.status
            self._state.status = ConnectionStatus.DISCONNECTED
            self._state.last_error = str(e)

            logger.error(
                "websocket_connection_failed",
                error=str(e),
                error_type=type(e).__name__,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
                exc_info=True,
            )

            logger.info(
                "websocket_state_changed",
                old_status=old_status.value if hasattr(old_status, "value") else str(old_status),
                new_status=self._state.status.value,
                connection_id=str(self._state.connection_id),
                reason="connection_failed",
                trace_id=trace_id,
            )
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            raise ConnectionError(f"Failed to connect to Bybit WebSocket: {e}") from e

    def _get_ws_url(self) -> str:
        """
        Get WebSocket URL based on endpoint type.
        
        Returns:
            WebSocket URL string for the configured endpoint type
        """
        if self._endpoint_type == "public":
            return settings.bybit_ws_url_public()
        else:  # private
            return settings.bybit_ws_url_private()

    async def _authenticate(self) -> None:
        """
        Authenticate with Bybit WebSocket.
        
        For public endpoints, authentication is skipped.

        Raises:
            ConnectionError: If authentication fails
        """
        trace_id = get_or_create_trace_id()
        
        # Public endpoints do not require authentication
        if self._endpoint_type == "public":
            logger.info(
                "websocket_public_endpoint_no_auth_required",
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )
            return
        
        if not self._websocket:
            logger.error(
                "websocket_auth_error",
                error="WebSocket not connected",
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
                exc_info=True,
            )
            raise ConnectionError("WebSocket not connected")

        # Generate and send authentication message
        auth_message = generate_auth_message()
        await self._websocket.send(json.dumps(auth_message))

        logger.debug(
            "websocket_auth_sent",
            api_key=settings.bybit_api_key[:8] + "...",
            connection_id=str(self._state.connection_id),
            trace_id=trace_id,
        )

        # Wait for authentication response (with timeout)
        try:
            response_str = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            response = json.loads(response_str)

            if validate_auth_response(response):
                logger.info(
                    "websocket_auth_success",
                    connection_id=str(self._state.connection_id),
                    trace_id=trace_id,
                )
            else:
                error_msg = response.get("ret_msg", "Unknown authentication error")
                logger.error(
                    "websocket_auth_failed",
                    error=error_msg,
                    response=response,  # Full response for debugging
                    connection_id=str(self._state.connection_id),
                    environment=settings.bybit_environment,
                    trace_id=trace_id,
                    exc_info=True,
                )
                raise ConnectionError(f"Authentication failed: {error_msg}")

        except asyncio.TimeoutError:
            logger.error(
                "websocket_auth_timeout",
                timeout=10.0,
                connection_id=str(self._state.connection_id),
                environment=settings.bybit_environment,
                trace_id=trace_id,
                exc_info=True,
            )
            raise ConnectionError("Authentication timeout") from None
        except json.JSONDecodeError as e:
            logger.error(
                "websocket_auth_invalid_response",
                error=str(e),
                error_type=type(e).__name__,
                raw_response=response_str[:200] if 'response_str' in locals() else None,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
                exc_info=True,
            )
            raise ConnectionError(f"Invalid authentication response: {e}") from e
        except Exception as e:
            logger.error(
                "websocket_auth_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
                exc_info=True,
            )
            raise ConnectionError(f"Unexpected authentication error: {e}") from e

    async def _receive_messages(self) -> None:
        """Continuously receive, parse and log messages from WebSocket."""
        if not self._websocket:
            return

        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    # Generate trace ID for this message or use existing one
                    trace_id = data.get("trace_id") or get_or_create_trace_id()
                    set_trace_id(trace_id)

                    # Log subscription confirmations and data messages at INFO level for visibility
                    topic = data.get("topic", data.get("op", "unknown"))
                    if data.get("op") == "subscribe" or topic != "unknown":
                        logger.info(
                            "websocket_message_received",
                            message_type=topic,
                            op=data.get("op"),
                            topic=topic,
                            message_id=data.get("req_id"),
                            full_message=data,  # Full message details for debugging
                            trace_id=trace_id,
                        )
                    else:
                        logger.debug(
                            "websocket_message_received",
                            message_type=topic,
                            topic=topic,
                            full_message=data,  # Full message details for debugging
                            trace_id=trace_id,
                        )

                    # Store message for viewing via API (testing)
                    _recent_messages.append(
                        {
                            "timestamp": data.get("ts", datetime.now().isoformat()),
                            "topic": data.get("topic", ""),
                            "op": data.get("op", ""),
                            "type": data.get("type", ""),
                            "data": data,
                        }
                    )

                    # Parse and process events for User Story 2
                    trace_id = data.get("trace_id") or str(uuid4())
                    await self._process_message(data, trace_id)

                except json.JSONDecodeError as e:
                    trace_id = get_or_create_trace_id()
                    logger.warning(
                        "websocket_invalid_json",
                        error=str(e),
                        error_type=type(e).__name__,
                        raw_message=message[:200],  # First 200 chars for debugging
                        trace_id=trace_id,
                        exc_info=True,
                    )
                except Exception as e:
                    trace_id = get_or_create_trace_id()
                    logger.error(
                        "websocket_message_processing_error",
                        error=str(e),
                        error_type=type(e).__name__,
                        trace_id=trace_id,
                        exc_info=True,
                    )

        except websockets.exceptions.ConnectionClosed:
            trace_id = get_or_create_trace_id()
            old_status = self._state.status
            logger.warning(
                "websocket_connection_closed",
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
            )
            self._state.status = ConnectionStatus.DISCONNECTED
            self._connected_event.clear()

            logger.info(
                "websocket_state_changed",
                old_status=old_status.value if hasattr(old_status, "value") else str(old_status),
                new_status=self._state.status.value,
                connection_id=str(self._state.connection_id),
                reason="connection_closed",
                trace_id=trace_id,
            )

            # Trigger reconnection (will be handled by reconnection manager if registered)
            await self._handle_disconnection()
        except Exception as e:
            trace_id = get_or_create_trace_id()
            old_status = self._state.status
            logger.error(
                "websocket_receive_error",
                error=str(e),
                error_type=type(e).__name__,
                connection_id=str(self._state.connection_id),
                trace_id=trace_id,
                exc_info=True,
            )
            self._state.status = ConnectionStatus.DISCONNECTED
            self._connected_event.clear()

            logger.info(
                "websocket_state_changed",
                old_status=old_status.value if hasattr(old_status, "value") else str(old_status),
                new_status=self._state.status.value,
                connection_id=str(self._state.connection_id),
                reason="receive_error",
                trace_id=trace_id,
            )

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

    async def _process_message(self, data: dict, trace_id: str) -> None:
        """Parse events from a WebSocket message and update subscription state.

        This implements the core of User Story 2 for event parsing and
        subscription tracking. Queue publishing and persistence are handled
        in later phases.
        """
        topic = data.get("topic")
        if not topic:
            return

        # Look up active subscription for this topic
        subscription = await SubscriptionRepository.find_active_by_topic(topic)
        if not subscription:
            return

        events = parse_events_from_message(
            message=data,
            subscription_lookup={subscription.topic: subscription},
            trace_id=trace_id,
        )

        if not events:
            return

        # Update subscription last_event_at
        last_ts = events[-1].timestamp
        await SubscriptionService.update_last_event_at(subscription.id, last_ts)

        # Log events
        for event in events:
            logger.info(
                "subscription_event_received",
                subscription_id=str(subscription.id),
                channel_type=subscription.channel_type,
                topic=subscription.topic,
                event_type=event.event_type,
                trace_id=event.trace_id,
            )

        # Process events: publish to queues (User Story 4)
        await process_events(events)

    async def subscribe(
        self,
        channel_type: str,
        requesting_service: str,
        symbol: Optional[str] = None,
    ):
        """Create a subscription and send Bybit subscribe message."""
        subscription = await SubscriptionService.create_subscription(
            channel_type=channel_type,
            requesting_service=requesting_service,
            symbol=symbol,
        )
        from .subscription import build_subscribe_message

        msg = build_subscribe_message([subscription])
        await self.send(msg)
        logger.info(
            "websocket_subscription_sent",
            subscription_id=str(subscription.id),
            topic=subscription.topic,
            requesting_service=requesting_service,
        )


# Global WebSocket connection instance
_connection: Optional[WebSocketConnection] = None


def get_connection() -> WebSocketConnection:
    """Get or create global WebSocket connection instance."""
    global _connection
    if _connection is None:
        _connection = WebSocketConnection()
    return _connection

