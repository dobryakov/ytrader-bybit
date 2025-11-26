"""Automatic reconnection logic with 30-second timeout."""

import asyncio
from typing import Optional

from ...config.logging import get_logger
from ...models.websocket_state import ConnectionStatus
from ..subscription.subscription_service import SubscriptionService
from .connection import WebSocketConnection
from .subscription import build_subscribe_message

logger = get_logger(__name__)

# Maximum reconnection delay: 30 seconds (per requirement)
MAX_RECONNECT_DELAY = 30.0
# Initial reconnection delay: 1 second
INITIAL_RECONNECT_DELAY = 1.0
# Exponential backoff multiplier
BACKOFF_MULTIPLIER = 2.0


class ReconnectionManager:
    """Manages automatic reconnection for WebSocket connections."""

    def __init__(self, connection: WebSocketConnection):
        """
        Initialize reconnection manager.

        Args:
            connection: WebSocket connection to manage
        """
        self._connection = connection
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect = False
        self._current_delay = INITIAL_RECONNECT_DELAY

    async def start(self) -> None:
        """Start monitoring connection and automatic reconnection."""
        self._should_reconnect = True
        logger.info("reconnection_manager_started")

    async def stop(self) -> None:
        """Stop automatic reconnection."""
        self._should_reconnect = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
        logger.info("reconnection_manager_stopped")

    async def handle_disconnection(self) -> None:
        """
        Handle disconnection event and initiate reconnection.

        This should be called when the connection is lost.
        """
        if not self._should_reconnect:
            return

        # Update connection state
        self._connection.state.status = ConnectionStatus.RECONNECTING
        self._connection.state.reconnect_count += 1

        logger.warning(
            "websocket_disconnected",
            reconnect_count=self._connection.state.reconnect_count,
        )

        # Start reconnection task if not already running
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Continuously attempt to reconnect with exponential backoff."""
        while self._should_reconnect:
            try:
                # Wait for current delay before attempting reconnection
                await asyncio.sleep(self._current_delay)

                if not self._should_reconnect:
                    break

                logger.info(
                    "websocket_reconnecting",
                    delay=self._current_delay,
                    attempt=self._connection.state.reconnect_count,
                )

                # Attempt to reconnect
                try:
                    await self._connection.connect()

                    # After successful reconnection, automatically resubscribe
                    subscriptions = await SubscriptionService.get_active_subscriptions()
                    if subscriptions:
                        msg = build_subscribe_message(subscriptions)
                        await self._connection.send(msg)
                        logger.info(
                            "websocket_resubscribed_active_subscriptions",
                            subscription_count=len(subscriptions),
                        )

                    # Reset delay on successful connection
                    self._current_delay = INITIAL_RECONNECT_DELAY
                    logger.info("websocket_reconnected_successfully")
                    break  # Exit loop on successful reconnection

                except Exception as e:
                    logger.warning(
                        "websocket_reconnect_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        next_attempt_in=self._current_delay,
                    )

                    # Update error state
                    self._connection.state.last_error = str(e)

                    # Increase delay with exponential backoff, capped at MAX_RECONNECT_DELAY
                    self._current_delay = min(
                        self._current_delay * BACKOFF_MULTIPLIER, MAX_RECONNECT_DELAY
                    )

                    # Increment reconnect count for next attempt
                    self._connection.state.reconnect_count += 1

            except asyncio.CancelledError:
                logger.info("reconnection_loop_cancelled")
                break
            except Exception as e:
                logger.error(
                    "reconnection_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Continue loop even on unexpected errors
                await asyncio.sleep(1.0)

        # Mark reconnection task as complete
        self._reconnect_task = None

    def reset_delay(self) -> None:
        """Reset reconnection delay to initial value (called on successful connection)."""
        self._current_delay = INITIAL_RECONNECT_DELAY

