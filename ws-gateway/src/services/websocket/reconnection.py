"""Automatic reconnection logic with 30-second timeout and circuit breaker pattern."""

import asyncio
from datetime import datetime, timedelta
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

# Circuit breaker thresholds
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Open circuit after 5 consecutive failures
CIRCUIT_BREAKER_TIMEOUT = 60.0  # Keep circuit open for 60 seconds before half-open


class CircuitBreakerState:
    """Circuit breaker states for handling extended API unavailability."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, not attempting connections
    HALF_OPEN = "half_open"  # Testing if service is back


class ReconnectionManager:
    """Manages automatic reconnection for WebSocket connections with circuit breaker pattern."""

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

        # Circuit breaker state
        self._circuit_state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._circuit_open_until: Optional[datetime] = None

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

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker allows reconnection attempt.

        Returns:
            True if reconnection should be attempted, False if circuit is open.
        """
        now = datetime.now()

        # Check if we should transition from OPEN to HALF_OPEN
        if self._circuit_state == CircuitBreakerState.OPEN:
            if self._circuit_open_until and now >= self._circuit_open_until:
                logger.info("circuit_breaker_half_open", failure_count=self._failure_count)
                self._circuit_state = CircuitBreakerState.HALF_OPEN
                return True
            return False  # Circuit still open

        return True  # CLOSED or HALF_OPEN states allow attempts

    def _record_failure(self) -> None:
        """Record a reconnection failure and update circuit breaker state."""
        self._failure_count += 1

        if self._failure_count >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            if self._circuit_state == CircuitBreakerState.CLOSED:
                logger.warning(
                    "circuit_breaker_opened",
                    failure_count=self._failure_count,
                    threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                )
                self._circuit_state = CircuitBreakerState.OPEN
                self._circuit_open_until = datetime.now() + timedelta(
                    seconds=CIRCUIT_BREAKER_TIMEOUT
                )

    def _record_success(self) -> None:
        """Record a successful connection and reset circuit breaker."""
        if self._circuit_state == CircuitBreakerState.HALF_OPEN:
            logger.info("circuit_breaker_closed_from_half_open")
            self._circuit_state = CircuitBreakerState.CLOSED

        self._failure_count = 0
        self._circuit_open_until = None

    async def _reconnect_loop(self) -> None:
        """Continuously attempt to reconnect with exponential backoff and circuit breaker."""
        while self._should_reconnect:
            try:
                # Check circuit breaker before attempting
                if not self._check_circuit_breaker():
                    # Circuit is open, wait before checking again
                    if self._circuit_open_until:
                        wait_seconds = (
                            self._circuit_open_until - datetime.now()
                        ).total_seconds()
                        if wait_seconds > 0:
                            logger.debug(
                                "circuit_breaker_waiting",
                                wait_seconds=wait_seconds,
                                failure_count=self._failure_count,
                            )
                            await asyncio.sleep(min(wait_seconds, CIRCUIT_BREAKER_TIMEOUT))
                    continue

                # Wait for current delay before attempting reconnection
                await asyncio.sleep(self._current_delay)

                if not self._should_reconnect:
                    break

                logger.info(
                    "websocket_reconnecting",
                    delay=self._current_delay,
                    attempt=self._connection.state.reconnect_count,
                    circuit_state=self._circuit_state,
                    failure_count=self._failure_count,
                )

                # Attempt to reconnect
                try:
                    await self._connection.connect()

                    # Record success and reset circuit breaker
                    self._record_success()

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
                    # Record failure for circuit breaker
                    self._record_failure()

                    logger.warning(
                        "websocket_reconnect_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        next_attempt_in=self._current_delay,
                        circuit_state=self._circuit_state,
                        failure_count=self._failure_count,
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

