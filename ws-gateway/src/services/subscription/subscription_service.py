"""Subscription management service."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

from ...config.logging import get_logger
from ...exceptions import ValidationError
from ...models.subscription import ChannelType, Subscription
from ..database.subscription_repository import SubscriptionRepository

logger = get_logger(__name__)


class SubscriptionService:
    """High-level subscription management operations."""

    VALID_CHANNEL_TYPES: List[ChannelType] = [
        "trades",
        "ticker",
        "orderbook",
        "order",
        "balance",
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
        if channel_type == "trades":
            if not symbol:
                raise ValidationError("Symbol is required for trades channel")
            return f"trade.{symbol}"
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
        """Create and persist a new subscription."""
        if channel_type not in cls.VALID_CHANNEL_TYPES:
            raise ValidationError(f"Invalid channel_type: {channel_type}")
        if not requesting_service:
            raise ValidationError("requesting_service is required")

        topic = cls._build_topic(channel_type, symbol)
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


