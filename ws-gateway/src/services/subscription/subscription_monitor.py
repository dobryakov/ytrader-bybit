"""Subscription monitoring service.

Monitors active subscriptions and alerts when last_event_at is stale (not updated recently).
This helps identify subscriptions that are not receiving events.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from ...config.logging import get_logger
from ...services.database.subscription_repository import SubscriptionRepository
from .subscription_service import SubscriptionService

logger = get_logger(__name__)

# Monitoring thresholds
STALE_SUBSCRIPTION_WARNING_MINUTES = 5  # Warn if last_event_at is older than 5 minutes
STALE_SUBSCRIPTION_CRITICAL_MINUTES = 30  # Critical if older than 30 minutes
MONITORING_CHECK_INTERVAL_SECONDS = 60  # Check every minute


class SubscriptionMonitor:
    """Monitors active subscriptions for stale last_event_at values."""

    def __init__(
        self,
        check_interval_seconds: int = MONITORING_CHECK_INTERVAL_SECONDS,
        warning_threshold_minutes: int = STALE_SUBSCRIPTION_WARNING_MINUTES,
        critical_threshold_minutes: int = STALE_SUBSCRIPTION_CRITICAL_MINUTES,
    ):
        """
        Initialize subscription monitor.

        Args:
            check_interval_seconds: How often to check subscriptions (default: 60 seconds)
            warning_threshold_minutes: Warn if last_event_at is older than this (default: 5 minutes)
            critical_threshold_minutes: Critical alert if older than this (default: 30 minutes)
        """
        self._check_interval = check_interval_seconds
        self._warning_threshold = timedelta(minutes=warning_threshold_minutes)
        self._critical_threshold = timedelta(minutes=critical_threshold_minutes)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """Start monitoring subscriptions."""
        if self._is_running:
            logger.warning("subscription_monitor_already_running")
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            "subscription_monitor_started",
            check_interval_seconds=self._check_interval,
            warning_threshold_minutes=self._warning_threshold.total_seconds() / 60,
            critical_threshold_minutes=self._critical_threshold.total_seconds() / 60,
        )

    async def stop(self) -> None:
        """Stop monitoring subscriptions."""
        self._is_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("subscription_monitor_stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self._check_stale_subscriptions()
            except Exception as e:
                logger.error(
                    "subscription_monitor_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
            finally:
                await asyncio.sleep(self._check_interval)

    async def _check_stale_subscriptions(self) -> None:
        """Check for stale subscriptions and log warnings/critical alerts."""
        now = datetime.now(timezone.utc)
        all_subscriptions = await SubscriptionService.get_active_subscriptions()

        stale_warning = []
        stale_critical = []
        never_received = []

        for subscription in all_subscriptions:
            if not subscription.last_event_at:
                # Subscription never received an event
                never_received.append(subscription)
                continue

            # Ensure last_event_at is timezone-aware (normalize to UTC)
            last_event_at = subscription.last_event_at
            if last_event_at.tzinfo is None:
                last_event_at = last_event_at.replace(tzinfo=timezone.utc)
            else:
                last_event_at = last_event_at.astimezone(timezone.utc)

            age = now - last_event_at
            if age > self._critical_threshold:
                stale_critical.append((subscription, age))
            elif age > self._warning_threshold:
                stale_warning.append((subscription, age))

        # Log warnings
        if stale_warning:
            for subscription, age in stale_warning:
                logger.warning(
                    "subscription_stale_warning",
                    subscription_id=str(subscription.id),
                    topic=subscription.topic,
                    requesting_service=subscription.requesting_service,
                    channel_type=subscription.channel_type,
                    last_event_at=last_event_at.isoformat(),
                    age_minutes=age.total_seconds() / 60,
                    threshold_minutes=self._warning_threshold.total_seconds() / 60,
                )

        # Log critical alerts
        if stale_critical:
            for subscription, age in stale_critical:
                # Normalize last_event_at for logging
                last_event_at = subscription.last_event_at
                if last_event_at:
                    if last_event_at.tzinfo is None:
                        last_event_at = last_event_at.replace(tzinfo=timezone.utc)
                    else:
                        last_event_at = last_event_at.astimezone(timezone.utc)
                
                logger.error(
                    "subscription_stale_critical",
                    subscription_id=str(subscription.id),
                    topic=subscription.topic,
                    requesting_service=subscription.requesting_service,
                    channel_type=subscription.channel_type,
                    last_event_at=last_event_at.isoformat() if last_event_at else None,
                    age_minutes=age.total_seconds() / 60,
                    threshold_minutes=self._critical_threshold.total_seconds() / 60,
                )

        # Log subscriptions that never received events (only if created more than threshold ago)
        if never_received:
            for subscription in never_received:
                if subscription.created_at:
                    # Normalize created_at to timezone-aware UTC
                    created_at = subscription.created_at
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    else:
                        created_at = created_at.astimezone(timezone.utc)
                    
                    age_since_creation = now - created_at
                    if age_since_creation > self._warning_threshold:
                        logger.warning(
                            "subscription_never_received_events",
                            subscription_id=str(subscription.id),
                            topic=subscription.topic,
                            requesting_service=subscription.requesting_service,
                            channel_type=subscription.channel_type,
                            created_at=created_at.isoformat(),
                            age_since_creation_minutes=age_since_creation.total_seconds() / 60,
                        )

        # Log summary
        if stale_warning or stale_critical or never_received:
            logger.info(
                "subscription_monitor_summary",
                total_subscriptions=len(all_subscriptions),
                stale_warning_count=len(stale_warning),
                stale_critical_count=len(stale_critical),
                never_received_count=len(never_received),
            )

