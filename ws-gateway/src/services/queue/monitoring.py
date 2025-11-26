"""Queue backlog and subscriber consumption monitoring.

EC7: Monitors queue backlog and alerts when subscribers consume slower than events arrive.
This helps identify slow consumer services that may cause queue buildup.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional

from ...config.logging import get_logger
from ...models.event import EventType
from .connection import QueueConnection
from .setup import QUEUE_RETENTION_MESSAGES, SUPPORTED_EVENT_TYPES, get_queue_name

logger = get_logger(__name__)

# Monitoring thresholds
BACKLOG_WARNING_THRESHOLD = 1000  # Warn if backlog exceeds 1000 messages
BACKLOG_CRITICAL_THRESHOLD = 10000  # Critical if backlog exceeds 10K messages
CONSUMPTION_RATE_WINDOW = 300  # 5 minutes window for consumption rate calculation


class QueueBacklogMonitor:
    """Monitors queue backlog and subscriber consumption rates (EC7)."""

    def __init__(self, check_interval_seconds: int = 60):
        """
        Initialize backlog monitor.

        Args:
            check_interval_seconds: How often to check queue backlogs (default: 1 minute)
        """
        self._check_interval = check_interval_seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Track publishing and consumption rates per queue
        # Format: {queue_name: deque([(timestamp, message_count), ...])}
        self._publish_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )  # Keep last 60 data points
        self._consumption_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=60)
        )

    async def start(self) -> None:
        """Start monitoring queue backlogs."""
        if self._is_running:
            logger.warning("backlog_monitor_already_running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "backlog_monitor_started",
            check_interval_seconds=self._check_interval,
        )

    async def stop(self) -> None:
        """Stop monitoring queue backlogs."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("backlog_monitor_stopped")

    def record_publish(self, event_type: EventType, message_count: int = 1) -> None:
        """
        Record a message publish event for rate tracking.

        Args:
            event_type: Event type being published
            message_count: Number of messages published (default: 1)
        """
        queue_name = get_queue_name(event_type)
        now = datetime.now()
        self._publish_history[queue_name].append((now, message_count))

    def record_consumption(self, event_type: EventType, message_count: int = 1) -> None:
        """
        Record a message consumption event for rate tracking.

        Note: This is called when we detect consumption (e.g., via queue stats).
        Actual consumption tracking would require integration with RabbitMQ Management API
        or consumer acknowledgment tracking.

        Args:
            event_type: Event type being consumed
            message_count: Number of messages consumed (default: 1)
        """
        queue_name = get_queue_name(event_type)
        now = datetime.now()
        self._consumption_history[queue_name].append((now, message_count))

    async def _monitor_loop(self) -> None:
        """Main monitoring loop for queue backlogs."""
        while self._is_running:
            try:
                await self._check_all_queue_backlogs()
            except Exception as e:
                logger.error(
                    "backlog_monitor_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            finally:
                await asyncio.sleep(self._check_interval)

    async def _check_all_queue_backlogs(self) -> None:
        """Check backlog for all event queues (EC7)."""
        if not QueueConnection.is_connected():
            logger.debug("backlog_check_skipped_no_connection")
            return

        try:
            channel = await QueueConnection.get_channel()

            # Check each queue
            for event_type in SUPPORTED_EVENT_TYPES:
                queue_name = get_queue_name(event_type)
                await self._check_queue_backlog(channel, queue_name, event_type)

        except Exception as e:
            logger.error(
                "backlog_check_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _check_queue_backlog(
        self,
        channel,
        queue_name: str,
        event_type: EventType,
    ) -> None:
        """
        Check queue backlog and consumption rate for a specific queue (EC7).

        Args:
            channel: RabbitMQ channel
            queue_name: Name of queue to check
            event_type: Event type for logging
        """
        try:
            # Declare queue to check if it exists
            queue = await channel.declare_queue(queue_name, passive=True)

            # Get estimated backlog (message count)
            # Note: Detailed stats require RabbitMQ Management API
            # For now, we'll estimate based on publish/consume rates and log warnings
            message_count = getattr(queue, 'message_count', None)

            if message_count is not None and message_count > 0:
                # Check backlog thresholds (EC7: Alert on slow subscriber consumption)
                if message_count >= BACKLOG_CRITICAL_THRESHOLD:
                    logger.error(
                        "queue_backlog_critical",
                        queue_name=queue_name,
                        event_type=event_type,
                        backlog_messages=message_count,
                        critical_threshold=BACKLOG_CRITICAL_THRESHOLD,
                        warning="Subscribers consuming much slower than events arrive",
                    )
                elif message_count >= BACKLOG_WARNING_THRESHOLD:
                    logger.warning(
                        "queue_backlog_warning",
                        queue_name=queue_name,
                        event_type=event_type,
                        backlog_messages=message_count,
                        warning_threshold=BACKLOG_WARNING_THRESHOLD,
                        note="Consider scaling up consumers or investigating slow subscribers",
                    )

            # Calculate publish vs consumption rates (EC7: Monitor consumption rate)
            publish_rate = self._calculate_rate(self._publish_history[queue_name])
            consumption_rate = self._calculate_rate(
                self._consumption_history[queue_name]
            )

            if publish_rate > 0 and consumption_rate > 0:
                rate_ratio = publish_rate / consumption_rate
                if rate_ratio > 1.5:  # Publishing 50% faster than consumption
                    logger.warning(
                        "queue_consumption_slow",
                        queue_name=queue_name,
                        event_type=event_type,
                        publish_rate_per_second=round(publish_rate, 2),
                        consumption_rate_per_second=round(consumption_rate, 2),
                        rate_ratio=round(rate_ratio, 2),
                        warning="Events arriving faster than consumed",
                    )

            # Log normal operation at debug level
            if message_count is not None:
                logger.debug(
                    "queue_backlog_checked",
                    queue_name=queue_name,
                    event_type=event_type,
                    backlog_messages=message_count,
                    publish_rate_per_second=round(publish_rate, 2) if publish_rate > 0 else None,
                    consumption_rate_per_second=round(consumption_rate, 2) if consumption_rate > 0 else None,
                )

        except Exception as e:
            # Queue might not exist yet, which is fine
            logger.debug(
                "backlog_check_skipped",
                queue_name=queue_name,
                event_type=event_type,
                reason=str(e),
            )

    def _calculate_rate(self, history: deque) -> float:
        """
        Calculate average rate (messages per second) from history.

        Args:
            history: Deque of (timestamp, message_count) tuples

        Returns:
            Average rate in messages per second, or 0.0 if insufficient data
        """
        if len(history) < 2:
            return 0.0

        # Get data points within the time window
        now = datetime.now()
        window_start = now - timedelta(seconds=CONSUMPTION_RATE_WINDOW)

        relevant_points = [
            (ts, count)
            for ts, count in history
            if ts >= window_start
        ]

        if len(relevant_points) < 2:
            return 0.0

        # Calculate total messages and time span
        total_messages = sum(count for _, count in relevant_points)
        time_span = (relevant_points[-1][0] - relevant_points[0][0]).total_seconds()

        if time_span <= 0:
            return 0.0

        return total_messages / time_span


# Global monitor instance
_backlog_monitor: Optional[QueueBacklogMonitor] = None


async def get_backlog_monitor() -> QueueBacklogMonitor:
    """Get or create global backlog monitor instance."""
    global _backlog_monitor
    if _backlog_monitor is None:
        _backlog_monitor = QueueBacklogMonitor()
    return _backlog_monitor


async def start_backlog_monitoring() -> None:
    """Start global backlog monitoring (EC7)."""
    monitor = await get_backlog_monitor()
    await monitor.start()


async def stop_backlog_monitoring() -> None:
    """Stop global backlog monitoring."""
    global _backlog_monitor
    if _backlog_monitor:
        await _backlog_monitor.stop()


def record_publish(event_type: EventType, message_count: int = 1) -> None:
    """Record a message publish for rate tracking."""
    global _backlog_monitor
    if _backlog_monitor:
        _backlog_monitor.record_publish(event_type, message_count)

