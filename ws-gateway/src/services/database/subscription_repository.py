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

    @staticmethod
    async def get_by_id(subscription_id: UUID) -> Optional[Subscription]:
        """Fetch a subscription by its ID."""
        query = """
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            WHERE id = $1
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, subscription_id)
        if not row:
            return None
        return Subscription(**dict(row))

    @staticmethod
    async def list_subscriptions(
        requesting_service: Optional[str] = None,
        is_active: Optional[bool] = None,
        channel_type: Optional[str] = None,
    ) -> List[Subscription]:
        """List subscriptions with optional filters."""
        conditions = []
        params = []

        if requesting_service is not None:
            conditions.append(f"requesting_service = ${len(params) + 1}")
            params.append(requesting_service)
        if is_active is not None:
            conditions.append(f"is_active = ${len(params) + 1}")
            params.append(is_active)
        if channel_type is not None:
            conditions.append(f"channel_type = ${len(params) + 1}")
            params.append(channel_type)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            {where_clause}
        """
        rows = await DatabaseConnection.fetch(query, *params)
        return [Subscription(**dict(row)) for row in rows]

    @staticmethod
    async def find_active_by_topic_and_service(
        topic: str,
        service_name: str,
    ) -> Optional[Subscription]:
        """Find an active subscription by topic and requesting service."""
        query = """
            SELECT id, channel_type, symbol, topic, requesting_service,
                   is_active, created_at, updated_at, last_event_at
            FROM subscriptions
            WHERE is_active = true AND topic = $1 AND requesting_service = $2
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, topic, service_name)
        if not row:
            return None
        return Subscription(**dict(row))

    @staticmethod
    async def deactivate_subscriptions_by_service(service_name: str) -> int:
        """Deactivate all subscriptions for a service, returning number of affected rows."""
        query = """
            UPDATE subscriptions
            SET is_active = false,
                updated_at = NOW()
            WHERE requesting_service = $1 AND is_active = true
        """
        result = await DatabaseConnection.execute(query, service_name)
        # asyncpg returns a string like "UPDATE <n>"
        try:
            _, count_str = result.split()
            return int(count_str)
        except Exception:
            logger.warning(
                "subscription_deactivate_service_rowcount_parse_failed",
                result=result,
            )
            return 0



