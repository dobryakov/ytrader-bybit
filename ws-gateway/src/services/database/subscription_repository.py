"""Database operations for Subscription entity."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from .connection import DatabaseConnection
from ...config.logging import get_logger
from ...models.subscription import Subscription

logger = get_logger(__name__)


class SubscriptionRepository:
    """Repository for CRUD operations on subscriptions table."""

    @staticmethod
    async def create_subscription(subscription: Subscription) -> Subscription:
        """Insert a new subscription into the database."""
        query = """
            INSERT INTO subscriptions (
                id, channel_type, symbol, topic, requesting_service,
                is_active, created_at, updated_at, last_event_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        await DatabaseConnection.execute(
            query,
            subscription.id,
            subscription.channel_type,
            subscription.symbol,
            subscription.topic,
            subscription.requesting_service,
            subscription.is_active,
            subscription.created_at or datetime.utcnow(),
            subscription.updated_at or datetime.utcnow(),
            subscription.last_event_at,
        )
        logger.info(
            "subscription_created",
            subscription_id=str(subscription.id),
            channel_type=subscription.channel_type,
            topic=subscription.topic,
            requesting_service=subscription.requesting_service,
        )
        return subscription

    @staticmethod
    async def get_active_subscriptions() -> List[Subscription]:
        """Return all active subscriptions."""
        query = """
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            WHERE is_active = true
        """
        rows = await DatabaseConnection.fetch(query)
        return [Subscription(**dict(row)) for row in rows]

    @staticmethod
    async def get_active_subscriptions_by_service(
        service_name: str,
    ) -> List[Subscription]:
        """Return active subscriptions for a specific requesting service."""
        query = """
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            WHERE is_active = true AND requesting_service = $1
        """
        rows = await DatabaseConnection.fetch(query, service_name)
        return [Subscription(**dict(row)) for row in rows]

    @staticmethod
    async def deactivate_subscription(subscription_id: UUID) -> None:
        """Mark a subscription as inactive."""
        query = """
            UPDATE subscriptions
            SET is_active = false,
                updated_at = NOW()
            WHERE id = $1
        """
        await DatabaseConnection.execute(query, subscription_id)
        logger.info("subscription_deactivated", subscription_id=str(subscription_id))

    @staticmethod
    async def update_last_event_at(subscription_id: UUID, ts: datetime) -> None:
        """Update last_event_at for a subscription."""
        query = """
            UPDATE subscriptions
            SET last_event_at = $1,
                updated_at = NOW()
            WHERE id = $2
        """
        await DatabaseConnection.execute(query, ts, subscription_id)
        logger.debug(
            "subscription_last_event_updated",
            subscription_id=str(subscription_id),
            last_event_at=ts.isoformat(),
        )

    @staticmethod
    async def find_active_by_topic(topic: str) -> Optional[Subscription]:
        """Find an active subscription by topic, if any."""
        query = """
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            WHERE is_active = true AND topic = $1
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, topic)
        if not row:
            return None
        return Subscription(**dict(row))

    @staticmethod
    async def count_active_subscriptions() -> int:
        """Return the number of active subscriptions."""
        query = "SELECT COUNT(*) AS count FROM subscriptions WHERE is_active = true"
        row = await DatabaseConnection.fetchrow(query)
        if not row:
            return 0
        return int(row["count"])



