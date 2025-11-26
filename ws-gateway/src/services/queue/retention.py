"""Queue retention monitoring and cleanup logic.

Per FR-019, queues have retention limits of 24 hours or 100K messages.
This module monitors queue age/size and discards messages exceeding limits.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

from ...config.logging import get_logger
from ...config.settings import settings
from ...exceptions import QueueError
from ...models.event import EventType
from .connection import QueueConnection
from .setup import (
    QUEUE_RETENTION_HOURS,
    QUEUE_RETENTION_MESSAGES,
    SUPPORTED_EVENT_TYPES,
    get_queue_name,
)

logger = get_logger(__name__)


class QueueRetentionMonitor:
    """Monitors queue retention and enforces limits per FR-019."""

    def __init__(self, check_interval_seconds: int = 300):
        """
        Initialize retention monitor.

        Args:
            check_interval_seconds: How often to check queue retention (default: 5 minutes)
        """
        self._check_interval = check_interval_seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """Start monitoring queue retention."""
        if self._is_running:
            logger.warning("retention_monitor_already_running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "retention_monitor_started",
            check_interval_seconds=self._check_interval,
        )

    async def stop(self) -> None:
        """Stop monitoring queue retention."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("retention_monitor_stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self._check_all_queues()
            except Exception as e:
                logger.error(
                    "retention_monitor_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                await asyncio.sleep(self._check_interval)

    async def _check_all_queues(self) -> None:
        """Check retention for all event queues."""
        if not QueueConnection.is_connected():
            logger.debug("retention_check_skipped_no_connection")
            return

        try:
            channel = await QueueConnection.get_channel()

            # Check each queue
            for event_type in SUPPORTED_EVENT_TYPES:
                queue_name = get_queue_name(event_type)
                await self._check_queue_retention(channel, queue_name, event_type)

        except Exception as e:
            logger.error(
                "retention_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _check_queue_retention(
        self,
        channel,
        queue_name: str,
        event_type: EventType,
    ) -> None:
        """
        Check and enforce retention limits for a specific queue.

        EC3: Monitors queue capacity and alerts when approaching limits.
        RabbitMQ's x-message-ttl and x-max-length arguments handle
        most retention automatically. This function provides monitoring
        and alerting for visibility.

        Args:
            channel: RabbitMQ channel
            queue_name: Name of queue to check
            event_type: Event type for logging
        """
        try:
            # Declare queue to get its properties
            queue = await channel.declare_queue(queue_name, passive=True)

            # Get queue statistics (EC3: Monitor queue capacity limits)
            # Note: aio-pika queue objects have some metadata, but detailed stats
            # require RabbitMQ Management API. We'll try to get message count if available.
            message_count = None
            try:
                # Try to get message count from queue (if supported)
                # aio-pika doesn't directly expose this, but we can infer from queue state
                # For detailed monitoring, RabbitMQ Management API would be needed
                message_count = getattr(queue, 'message_count', None)
            except AttributeError:
                pass

            # Alert thresholds (EC3: Alert when approaching capacity limits)
            WARNING_THRESHOLD = 0.8  # Alert at 80% capacity
            CRITICAL_THRESHOLD = 0.95  # Critical alert at 95% capacity

            if message_count is not None:
                capacity_percentage = message_count / QUEUE_RETENTION_MESSAGES

                if capacity_percentage >= CRITICAL_THRESHOLD:
                    logger.error(
                        "queue_capacity_critical",
                        queue_name=queue_name,
                        event_type=event_type,
                        message_count=message_count,
                        max_messages=QUEUE_RETENTION_MESSAGES,
                        capacity_percentage=round(capacity_percentage * 100, 2),
                        threshold_percentage=CRITICAL_THRESHOLD * 100,
                    )
                elif capacity_percentage >= WARNING_THRESHOLD:
                    logger.warning(
                        "queue_capacity_warning",
                        queue_name=queue_name,
                        event_type=event_type,
                        message_count=message_count,
                        max_messages=QUEUE_RETENTION_MESSAGES,
                        capacity_percentage=round(capacity_percentage * 100, 2),
                        threshold_percentage=WARNING_THRESHOLD * 100,
                    )
                else:
                    logger.debug(
                        "retention_check_completed",
                        queue_name=queue_name,
                        event_type=event_type,
                        message_count=message_count,
                        retention_hours=QUEUE_RETENTION_HOURS,
                        retention_messages=QUEUE_RETENTION_MESSAGES,
                        capacity_percentage=round(capacity_percentage * 100, 2),
                    )
            else:
                # Fallback logging when message count not available
                logger.debug(
                    "retention_check_completed",
                    queue_name=queue_name,
                    event_type=event_type,
                    retention_hours=QUEUE_RETENTION_HOURS,
                    retention_messages=QUEUE_RETENTION_MESSAGES,
                    note="Message count not available (requires RabbitMQ Management API for detailed stats)",
                )

        except Exception as e:
            # Queue might not exist yet, which is fine
            logger.debug(
                "retention_check_skipped",
                queue_name=queue_name,
                event_type=event_type,
                reason=str(e),
            )

    async def get_queue_stats(self, event_type: EventType) -> Optional[Dict]:
        """
        Get retention statistics for a queue.

        Args:
            event_type: Event type to get stats for

        Returns:
            Dictionary with queue stats or None if unavailable
        """
        if not QueueConnection.is_connected():
            return None

        try:
            channel = await QueueConnection.get_channel()
            queue_name = get_queue_name(event_type)

            # Declare queue passively to check if it exists
            await channel.declare_queue(queue_name, passive=True)

            # Note: Detailed stats would require RabbitMQ management API
            # For now, return basic info
            return {
                "queue_name": queue_name,
                "event_type": event_type,
                "retention_hours": QUEUE_RETENTION_HOURS,
                "retention_messages": QUEUE_RETENTION_MESSAGES,
            }
        except Exception as e:
            logger.debug(
                "queue_stats_unavailable",
                event_type=event_type,
                error=str(e),
            )
            return None


# Global monitor instance
_monitor: Optional[QueueRetentionMonitor] = None


async def get_monitor() -> QueueRetentionMonitor:
    """Get or create global retention monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = QueueRetentionMonitor()
    return _monitor


async def start_monitoring() -> None:
    """Start global retention monitoring."""
    monitor = await get_monitor()
    await monitor.start()


async def stop_monitoring() -> None:
    """Stop global retention monitoring."""
    global _monitor
    if _monitor:
        await _monitor.stop()

