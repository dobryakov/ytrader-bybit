"""Subscription management service."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from typing import TYPE_CHECKING

from ...config.logging import get_logger
from ...exceptions import ValidationError
from ...models.subscription import ChannelType, Subscription
from ..database.subscription_repository import SubscriptionRepository

if TYPE_CHECKING:
    from ..websocket.connection_manager import ConnectionManager

logger = get_logger(__name__)


class SubscriptionService:
    """High-level subscription management operations."""

    VALID_CHANNEL_TYPES: List[ChannelType] = [
        "trades",
        "ticker",
        "orderbook",
        "order",
        "balance",
        "position",
        "kline",
        "liquidation",
    ]

    @classmethod
    def _build_topic(cls, channel_type: ChannelType, symbol: Optional[str]) -> str:
        """Build Bybit topic string from channel type and symbol."""
        if channel_type == "balance":
            return "wallet"
        if channel_type == "order":
            return "order"
        if channel_type == "position":
            return "position"
        if channel_type == "trades":
            if not symbol:
                raise ValidationError("Symbol is required for trades channel")
            # Bybit v5 WebSocket API uses "publicTrade.{symbol}" format for trades
            return f"publicTrade.{symbol}"
        if channel_type == "ticker":
            if not symbol:
                raise ValidationError("Symbol is required for ticker channel")
            return f"tickers.{symbol}"
        if channel_type == "orderbook":
            if not symbol:
                raise ValidationError("Symbol is required for orderbook channel")
            # Use level 1 orderbook as default; can be parameterized later
            return f"orderbook.1.{symbol}"
        if channel_type == "kline":
            if not symbol:
                raise ValidationError("Symbol is required for kline channel")
            # Default interval 1m; can be made configurable later
            return f"kline.1.{symbol}"
        if channel_type == "liquidation":
            return "liquidation"
        raise ValidationError(f"Unsupported channel_type: {channel_type}")

    @classmethod
    async def create_subscription(
        cls,
        channel_type: ChannelType,
        requesting_service: str,
        symbol: Optional[str] = None,
    ) -> Subscription:
        """
        Create and persist a new subscription.

        Handles conflicting subscription configurations (EC5: Handle conflicting subscription configurations).
        If multiple services request the same topic, all subscriptions are allowed (fan-out pattern).
        """
        if channel_type not in cls.VALID_CHANNEL_TYPES:
            raise ValidationError(f"Invalid channel_type: {channel_type}")
        if not requesting_service:
            raise ValidationError("requesting_service is required")

        topic = cls._build_topic(channel_type, symbol)

        # Check for existing active subscriptions with same topic (EC5: Conflict detection)
        existing_subscriptions = await SubscriptionRepository.get_active_subscriptions_by_topic(
            topic
        )

        # Check if this service already has an active subscription for this topic
        existing_for_service = [
            sub
            for sub in existing_subscriptions
            if sub.requesting_service == requesting_service
        ]

        if existing_for_service:
            # Service already has active subscription - return existing instead of creating duplicate
            logger.info(
                "subscription_already_exists_for_service",
                topic=topic,
                requesting_service=requesting_service,
                existing_subscription_id=str(existing_for_service[0].id),
            )
            return existing_for_service[0]

        # If other services have subscriptions for this topic, log for visibility
        if existing_subscriptions:
            other_services = [
                sub.requesting_service
                for sub in existing_subscriptions
                if sub.requesting_service != requesting_service
            ]
            logger.info(
                "subscription_topic_already_subscribed",
                topic=topic,
                requesting_service=requesting_service,
                other_services=other_services,
                total_subscriptions_for_topic=len(existing_subscriptions),
            )
            # Allow multiple services to subscribe to same topic (fan-out pattern)

        subscription = Subscription.create(
            channel_type=channel_type,
            topic=topic,
            requesting_service=requesting_service,
            symbol=symbol,
        )
        subscription = await SubscriptionRepository.create_subscription(subscription)
        logger.info(
            "subscription_service_created",
            subscription=asdict(subscription),
            existing_subscriptions_count=len(existing_subscriptions),
        )
        return subscription

    @staticmethod
    async def get_active_subscriptions() -> List[Subscription]:
        """Return all active subscriptions."""
        return await SubscriptionRepository.get_active_subscriptions()

    @staticmethod
    async def get_active_subscriptions_by_service(
        service_name: str,
    ) -> List[Subscription]:
        """Return active subscriptions for a given service."""
        return await SubscriptionRepository.get_active_subscriptions_by_service(
            service_name
        )

    @staticmethod
    async def deactivate_subscription(subscription_id) -> None:
        """Deactivate a subscription."""
        await SubscriptionRepository.deactivate_subscription(subscription_id)

    @staticmethod
    async def update_last_event_at(subscription_id, ts: datetime) -> None:
        """Update last_event_at for a subscription."""
        await SubscriptionRepository.update_last_event_at(subscription_id, ts)

    # --- New helpers for REST API layer ---

    @staticmethod
    async def list_subscriptions(
        requesting_service: Optional[str] = None,
        is_active: Optional[bool] = None,
        channel_type: Optional[str] = None,
    ) -> List[Subscription]:
        """List subscriptions with optional filters."""
        return await SubscriptionRepository.list_subscriptions(
            requesting_service=requesting_service,
            is_active=is_active,
            channel_type=channel_type,
        )

    @staticmethod
    async def get_subscription_by_id(subscription_id) -> Optional[Subscription]:
        """Return a subscription by ID."""
        return await SubscriptionRepository.get_by_id(subscription_id)

    @staticmethod
    async def deactivate_subscription_if_exists(subscription_id) -> bool:
        """Deactivate a subscription if it exists. Returns True if it existed."""
        existing = await SubscriptionRepository.get_by_id(subscription_id)
        if not existing:
            return False
        await SubscriptionRepository.deactivate_subscription(subscription_id)
        return True

    @staticmethod
    async def deactivate_subscriptions_by_service(service_name: str) -> int:
        """Deactivate all subscriptions for a given service."""
        return await SubscriptionRepository.deactivate_subscriptions_by_service(
            service_name
        )

    @classmethod
    async def subscribe(
        cls,
        channel_type: ChannelType,
        requesting_service: str,
        symbol: Optional[str] = None,
    ) -> Subscription:
        """
        Create a subscription and send it to the appropriate WebSocket connection.
        
        This method creates the subscription in the database and sends the subscribe
        message to the correct endpoint (public or private) based on channel type.
        
        Args:
            channel_type: Type of channel to subscribe to
            requesting_service: Name of the service requesting the subscription
            symbol: Optional trading pair symbol
            
        Returns:
            Created Subscription object
        """
        # Create subscription in database
        subscription = await cls.create_subscription(
            channel_type=channel_type,
            requesting_service=requesting_service,
            symbol=symbol,
        )
        
        # Get appropriate connection for this subscription
        # Import here to avoid circular import
        from ..websocket.connection_manager import get_connection_manager
        from ..websocket.subscription import build_subscribe_message
        
        connection_manager = get_connection_manager()
        
        # Try to get connection, but don't fail subscription creation if connection fails
        # Connection will be retried on reconnection
        try:
            connection = await connection_manager.get_connection_for_subscription(subscription)
            
            # Build and send subscribe message
            msg = build_subscribe_message([subscription])
            await connection.send(msg)
            
            endpoint_type = "public" if channel_type in {"trades", "ticker", "orderbook", "kline", "liquidation"} else "private"
            logger.info(
                "subscription_sent_to_websocket",
                subscription_id=str(subscription.id),
                topic=subscription.topic,
                channel_type=subscription.channel_type,
                requesting_service=requesting_service,
                endpoint_type=endpoint_type,
            )
        except Exception as e:
            # Log error but don't fail subscription creation
            # Subscription will be sent on reconnection
            endpoint_type = "public" if channel_type in {"trades", "ticker", "orderbook", "kline", "liquidation"} else "private"
            logger.warning(
                "subscription_connection_failed_will_retry",
                subscription_id=str(subscription.id),
                topic=subscription.topic,
                channel_type=subscription.channel_type,
                endpoint_type=endpoint_type,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Subscription is still created in DB, will be sent on reconnection
        
        return subscription



