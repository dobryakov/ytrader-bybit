"""Heartbeat mechanism for connection maintenance."""

import asyncio
from datetime import datetime
from typing import Optional

from ...config.logging import get_logger
from .connection import WebSocketConnection

logger = get_logger(__name__)

# Heartbeat interval: 20 seconds (standard WebSocket ping interval)
HEARTBEAT_INTERVAL = 20.0
# Heartbeat timeout: 10 seconds (wait for pong response)
HEARTBEAT_TIMEOUT = 10.0


class HeartbeatManager:
    """Manages WebSocket heartbeat (ping/pong) to maintain connection."""

    def __init__(self, connection: WebSocketConnection):
        """
        Initialize heartbeat manager.

        Args:
            connection: WebSocket connection to maintain
        """
        self._connection = connection
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._should_heartbeat = False

    async def start(self) -> None:
        """Start sending periodic heartbeat messages."""
        self._should_heartbeat = True
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("heartbeat_manager_started", interval=HEARTBEAT_INTERVAL)

    async def stop(self) -> None:
        """Stop sending heartbeat messages."""
        self._should_heartbeat = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        logger.info("heartbeat_manager_stopped")

    async def _heartbeat_loop(self) -> None:
        """Continuously send ping messages at regular intervals."""
        while self._should_heartbeat:
            try:
                # Wait for heartbeat interval
                await asyncio.sleep(HEARTBEAT_INTERVAL)

                if not self._should_heartbeat:
                    break

                # Check if connection is still active
                if not self._connection.is_connected:
                    logger.warning("heartbeat_skipped_connection_not_active")
                    continue

                # Send ping message (Bybit WebSocket uses standard ping/pong)
                try:
                    # Use WebSocket ping frame (websockets library handles this)
                    # Access the websocket through a method if available, or use private attribute
                    websocket = getattr(self._connection, '_websocket', None)
                    if websocket and not getattr(websocket, 'closed', False):
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=HEARTBEAT_TIMEOUT)

                        # Update last heartbeat timestamp
                        self._connection.state.last_heartbeat_at = datetime.now()

                        logger.debug("heartbeat_sent_successfully")

                except asyncio.TimeoutError:
                    logger.warning("heartbeat_timeout_no_pong_received")
                    # Connection might be dead, but let reconnection manager handle it
                except Exception as e:
                    logger.warning(
                        "heartbeat_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Connection might be dead, but let reconnection manager handle it

            except asyncio.CancelledError:
                logger.debug("heartbeat_loop_cancelled")
                break
            except Exception as e:
                logger.error(
                    "heartbeat_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Continue loop even on unexpected errors
                await asyncio.sleep(1.0)

        # Mark heartbeat task as complete
        self._heartbeat_task = None

