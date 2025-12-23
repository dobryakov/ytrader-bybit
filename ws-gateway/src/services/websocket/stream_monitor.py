"""Stream activity monitor for WebSocket connections."""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from ...config.logging import get_logger
from ...config.settings import settings
from .connection_manager import get_connection_manager

logger = get_logger(__name__)

# How often to check stream activity (seconds)
STREAM_CHECK_INTERVAL_SECONDS = 30

# If no messages for this period, consider stream inactive
STREAM_INACTIVITY_THRESHOLD_SECONDS = 120


class StreamActivityMonitor:
    """Monitors WebSocket stream activity using last_message_at and triggers reconnection if needed."""

    def __init__(
        self,
        check_interval_seconds: int = STREAM_CHECK_INTERVAL_SECONDS,
        inactivity_threshold_seconds: int = STREAM_INACTIVITY_THRESHOLD_SECONDS,
    ) -> None:
        self._check_interval = check_interval_seconds
        self._inactivity_threshold = timedelta(seconds=inactivity_threshold_seconds)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._connection_manager = get_connection_manager()

    async def start(self) -> None:
        """Start monitoring stream activity."""
        if self._is_running:
            logger.warning("stream_activity_monitor_already_running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            "stream_activity_monitor_started",
            check_interval_seconds=self._check_interval,
            inactivity_threshold_seconds=self._inactivity_threshold.total_seconds(),
        )

    async def stop(self) -> None:
        """Stop monitoring stream activity."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("stream_activity_monitor_stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self._check_stream_activity()
            except Exception as e:
                logger.error(
                    "stream_activity_monitor_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
            finally:
                await asyncio.sleep(self._check_interval)

    async def _check_stream_activity(self) -> None:
        """Check activity of public WebSocket stream and trigger reconnection if necessary."""
        now = datetime.now()

        public_conn = self._connection_manager.get_public_connection_sync()
        if not public_conn:
            # No public connection yet; nothing to check
            return

        state = public_conn.state
        last_message_at = state.last_message_at

        # If we never received a message yet, don't treat it as inactivity
        if last_message_at is None:
            return

        age = now - last_message_at
        if age <= self._inactivity_threshold:
            return

        logger.warning(
            "websocket_stream_inactive",
            endpoint_type=state.endpoint_type,
            environment=state.environment,
            public_category=settings.bybit_ws_public_category,
            last_message_at=last_message_at.isoformat(),
            age_seconds=age.total_seconds(),
            threshold_seconds=self._inactivity_threshold.total_seconds(),
        )

        # Best-effort reconnection: disconnect and let reconnection manager handle it
        try:
            await public_conn.disconnect()
        except Exception as e:
            logger.error(
                "websocket_stream_inactive_disconnect_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )


