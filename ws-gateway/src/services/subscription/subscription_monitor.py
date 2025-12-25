"""Subscription monitoring service.

Monitors active subscriptions and alerts when last_event_at is stale (not updated recently).
This helps identify subscriptions that are not receiving events.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from ...config.logging import get_logger
from ...services.database.subscription_repository import SubscriptionRepository
from .subscription_service import SubscriptionService
from ..websocket.connection_manager import get_connection_manager

logger = get_logger(__name__)

# Monitoring thresholds
STALE_SUBSCRIPTION_WARNING_MINUTES = 5  # Warn if last_event_at is older than 5 minutes
STALE_SUBSCRIPTION_CRITICAL_MINUTES = 30  # Critical if older than 30 minutes
STALE_SUBSCRIPTION_AUTO_DEACTIVATE_MINUTES = 60  # Auto-deactivate if older than 60 minutes
MONITORING_CHECK_INTERVAL_SECONDS = 60  # Check every minute

# Event-driven channels that may not receive events frequently
# For these channels, we check WebSocket connection state in addition to last_event_at
EVENT_DRIVEN_CHANNELS = {"position", "balance", "order"}
# Frequent channels that should receive events regularly
FREQUENT_CHANNELS = {"ticker", "trades", "orderbook", "kline"}
# Periodic channels with scheduled updates
PERIODIC_CHANNELS = {"funding"}

# Threshold for considering WebSocket connection as active (last_message_at age)
CONNECTION_ACTIVE_THRESHOLD_MINUTES = 5  # Connection is active if last_message_at is within 5 minutes


class SubscriptionMonitor:
    """Monitors active subscriptions for stale last_event_at values."""

    def __init__(
        self,
        check_interval_seconds: int = MONITORING_CHECK_INTERVAL_SECONDS,
        warning_threshold_minutes: int = STALE_SUBSCRIPTION_WARNING_MINUTES,
        critical_threshold_minutes: int = STALE_SUBSCRIPTION_CRITICAL_MINUTES,
        auto_deactivate_threshold_minutes: int = STALE_SUBSCRIPTION_AUTO_DEACTIVATE_MINUTES,
    ):
        """
        Initialize subscription monitor.

        Args:
            check_interval_seconds: How often to check subscriptions (default: 60 seconds)
            warning_threshold_minutes: Warn if last_event_at is older than this (default: 5 minutes)
            critical_threshold_minutes: Critical alert if older than this (default: 30 minutes)
            auto_deactivate_threshold_minutes: Auto-deactivate if older than this (default: 60 minutes)
        """
        self._check_interval = check_interval_seconds
        self._warning_threshold = timedelta(minutes=warning_threshold_minutes)
        self._critical_threshold = timedelta(minutes=critical_threshold_minutes)
        self._auto_deactivate_threshold = timedelta(minutes=auto_deactivate_threshold_minutes)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
        # Cached connection manager for resubscribe operations
        self._connection_manager = get_connection_manager()
        # Last time full resubscribe was attempted (to rate-limit)
        self._last_full_resubscribe_at: Optional[datetime] = None
        self._full_resubscribe_interval = timedelta(minutes=5)

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
            auto_deactivate_threshold_minutes=self._auto_deactivate_threshold.total_seconds() / 60,
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

        stale_warning: List[Tuple[object, timedelta]] = []
        stale_critical: List[Tuple[object, timedelta]] = []
        stale_auto_deactivate: List[Tuple[object, timedelta]] = []
        never_received: List[object] = []

        # Get connection states for event-driven channels
        connection_states = await self._get_connection_states()

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
            
            # For event-driven channels, check WebSocket connection state
            # If connection is alive and receiving messages, don't deactivate even if last_event_at is stale
            is_event_driven = subscription.channel_type in EVENT_DRIVEN_CHANNELS
            connection_active = False
            
            if is_event_driven:
                connection_active = self._is_connection_active_for_subscription(
                    subscription, connection_states, now
                )
                
                # If connection is active, skip deactivation even if last_event_at is stale
                # This handles the case where position/balance/order don't change but connection is alive
                if connection_active and age > self._auto_deactivate_threshold:
                    logger.debug(
                        "subscription_stale_but_connection_active",
                        subscription_id=str(subscription.id),
                        channel_type=subscription.channel_type,
                        topic=subscription.topic,
                        age_minutes=age.total_seconds() / 60,
                        reason="WebSocket connection is active, skipping deactivation",
                    )
                    # Still log as critical for monitoring, but don't deactivate
                    stale_critical.append((subscription, age))
                    continue

            if age > self._auto_deactivate_threshold:
                stale_auto_deactivate.append((subscription, age))
            elif age > self._critical_threshold:
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
            # Group critical subscriptions by (symbol, channel_type) for nicer diagnostics
            grouped: Dict[Tuple[Optional[str], str], List[Tuple[object, timedelta]]] = {}
            for subscription, age in stale_critical:
                key = (getattr(subscription, "symbol", None), subscription.channel_type)
                grouped.setdefault(key, []).append((subscription, age))

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

            # Log aggregated symbol-level view and trigger resubscribe attempts
            for (symbol, channel_type), items in grouped.items():
                subscriptions_for_key = [sub for sub, _ in items]
                logger.warning(
                    "subscription_stale_critical_group",
                    symbol=symbol,
                    channel_type=channel_type,
                    subscription_ids=[str(sub.id) for sub in subscriptions_for_key],
                    requesting_services=[sub.requesting_service for sub in subscriptions_for_key],
                    count=len(subscriptions_for_key),
                )

                # Attempt lightweight resubscribe for these subscriptions
                await self._resubscribe_subscriptions(subscriptions_for_key)

            # If a large portion of subscriptions are critical, trigger a full resubscribe
            critical_ratio = len(stale_critical) / max(len(all_subscriptions), 1)
            now = datetime.now(timezone.utc)
            should_full_resubscribe = (
                critical_ratio >= 0.5
                and (
                    self._last_full_resubscribe_at is None
                    or (now - self._last_full_resubscribe_at) >= self._full_resubscribe_interval
                )
            )
            if should_full_resubscribe:
                try:
                    await self._connection_manager.resubscribe_all_active()
                    self._last_full_resubscribe_at = now
                    logger.info(
                        "subscription_full_resubscribe_triggered",
                        critical_count=len(stale_critical),
                        total_subscriptions=len(all_subscriptions),
                        critical_ratio=critical_ratio,
                    )
                except Exception as e:
                    logger.error(
                        "subscription_full_resubscribe_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
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

        # Auto-deactivate very stale subscriptions (older than auto_deactivate_threshold)
        if stale_auto_deactivate:
            for subscription, age in stale_auto_deactivate:
                # Normalize last_event_at for logging
                last_event_at = subscription.last_event_at
                if last_event_at:
                    if last_event_at.tzinfo is None:
                        last_event_at = last_event_at.replace(tzinfo=timezone.utc)
                    else:
                        last_event_at = last_event_at.astimezone(timezone.utc)
                
                logger.warning(
                    "subscription_auto_deactivating_stale",
                    subscription_id=str(subscription.id),
                    topic=subscription.topic,
                    requesting_service=subscription.requesting_service,
                    channel_type=subscription.channel_type,
                    last_event_at=last_event_at.isoformat() if last_event_at else None,
                    age_minutes=age.total_seconds() / 60,
                    threshold_minutes=self._auto_deactivate_threshold.total_seconds() / 60,
                )
                
                # Deactivate the subscription
                try:
                    await SubscriptionService.deactivate_subscription(subscription.id)
                    logger.info(
                        "subscription_auto_deactivated",
                        subscription_id=str(subscription.id),
                        topic=subscription.topic,
                        requesting_service=subscription.requesting_service,
                        channel_type=subscription.channel_type,
                        age_minutes=age.total_seconds() / 60,
                    )
                except Exception as e:
                    logger.error(
                        "subscription_auto_deactivate_failed",
                        subscription_id=str(subscription.id),
                        error=str(e),
                        error_type=type(e).__name__,
                        exc_info=True,
                    )

        # Log summary
        if stale_warning or stale_critical or stale_auto_deactivate or never_received:
            logger.info(
                "subscription_monitor_summary",
                total_subscriptions=len(all_subscriptions),
                stale_warning_count=len(stale_warning),
                stale_critical_count=len(stale_critical),
                stale_auto_deactivated_count=len(stale_auto_deactivate),
                never_received_count=len(never_received),
            )

    async def _resubscribe_subscriptions(self, subscriptions: List[object]) -> None:
        """
        Attempt lightweight resubscribe for a set of subscriptions.

        This is a best-effort operation: errors are logged but do not stop monitoring.
        It relies on ConnectionManager to pick the correct (public/private) connection.
        """
        if not subscriptions:
            return

        try:
            from ..websocket.subscription import build_subscribe_messages  # local import to avoid cycles
            from ..websocket.channel_types import get_endpoint_type_for_channel

            # Group subscriptions by endpoint type to send via appropriate connections
            by_endpoint: Dict[str, List[object]] = {}
            for sub in subscriptions:
                endpoint_type = get_endpoint_type_for_channel(sub.channel_type)
                by_endpoint.setdefault(endpoint_type, []).append(sub)

            for endpoint_type, subs in by_endpoint.items():
                # Get appropriate connection
                if endpoint_type == "public":
                    connection = await self._connection_manager.get_public_connection()
                else:
                    connection = await self._connection_manager.get_private_connection()

                messages = build_subscribe_messages(subs, max_topics_per_message=10)
                for msg in messages:
                    try:
                        await connection.send(msg)
                    except Exception as e:
                        logger.warning(
                            "subscription_resubscribe_send_failed",
                            error=str(e),
                            error_type=type(e).__name__,
                            endpoint_type=endpoint_type,
                        )
                        continue

                logger.info(
                    "subscription_resubscribed_group",
                    endpoint_type=endpoint_type,
                    subscription_ids=[str(s.id) for s in subs],
                    topics=[s.topic for s in subs],
                    count=len(subs),
                    message_batches=len(messages),
                )

        except Exception as e:
            logger.error(
                "subscription_resubscribe_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    async def _get_connection_states(self) -> Dict[str, Dict]:
        """Get connection states for public and private connections.
        
        Returns:
            Dictionary with 'public' and 'private' keys, each containing connection state info
        """
        states = {}
        
        try:
            # Get public connection state
            public_conn = self._connection_manager.get_public_connection_sync()
            if public_conn:
                state = public_conn.state
                states["public"] = {
                    "is_connected": public_conn.is_connected,
                    "last_message_at": state.last_message_at,
                    "last_heartbeat_at": state.last_heartbeat_at,
                }
            
            # Get private connection state
            private_conn = self._connection_manager.get_private_connection_sync()
            if private_conn:
                state = private_conn.state
                states["private"] = {
                    "is_connected": private_conn.is_connected,
                    "last_message_at": state.last_message_at,
                    "last_heartbeat_at": state.last_heartbeat_at,
                }
        except Exception as e:
            logger.warning(
                "subscription_monitor_connection_state_error",
                error=str(e),
                error_type=type(e).__name__,
            )
        
        return states

    def _is_connection_active_for_subscription(
        self,
        subscription: object,
        connection_states: Dict[str, Dict],
        now: datetime,
    ) -> bool:
        """Check if WebSocket connection is active for an event-driven subscription.
        
        Args:
            subscription: Subscription object
            connection_states: Connection states dictionary from _get_connection_states()
            now: Current datetime (timezone-aware UTC)
            
        Returns:
            True if connection is active and receiving messages, False otherwise
        """
        from ..websocket.channel_types import get_endpoint_type_for_channel
        
        try:
            endpoint_type = get_endpoint_type_for_channel(subscription.channel_type)
            state = connection_states.get(endpoint_type)
            
            if not state:
                return False
            
            # Check if connection is connected
            if not state.get("is_connected", False):
                return False
            
            # Check last_message_at (any message, including ping/pong)
            last_message_at = state.get("last_message_at")
            if last_message_at:
                if last_message_at.tzinfo is None:
                    last_message_at = last_message_at.replace(tzinfo=timezone.utc)
                else:
                    last_message_at = last_message_at.astimezone(timezone.utc)
                
                message_age = now - last_message_at
                if message_age.total_seconds() / 60 <= CONNECTION_ACTIVE_THRESHOLD_MINUTES:
                    return True
            
            # Fallback: check last_heartbeat_at
            last_heartbeat_at = state.get("last_heartbeat_at")
            if last_heartbeat_at:
                if last_heartbeat_at.tzinfo is None:
                    last_heartbeat_at = last_heartbeat_at.replace(tzinfo=timezone.utc)
                else:
                    last_heartbeat_at = last_heartbeat_at.astimezone(timezone.utc)
                
                heartbeat_age = now - last_heartbeat_at
                if heartbeat_age.total_seconds() / 60 <= CONNECTION_ACTIVE_THRESHOLD_MINUTES:
                    return True
            
            return False
        except Exception as e:
            logger.warning(
                "subscription_monitor_connection_check_error",
                subscription_id=str(subscription.id),
                channel_type=subscription.channel_type,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

