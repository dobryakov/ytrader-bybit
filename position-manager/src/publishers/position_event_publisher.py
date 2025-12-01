"""RabbitMQ publishers for position and portfolio events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aio_pika import Message

from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..models import PortfolioMetrics, Position

logger = get_logger(__name__)


class PositionEventPublisher:
    """Publish position and portfolio events to RabbitMQ."""

    POSITION_QUEUE = "position-manager.position_updated"
    PORTFOLIO_QUEUE = "position-manager.portfolio_updated"

    @classmethod
    async def _publish(cls, queue_name: str, payload: Dict[str, Any]) -> None:
        """Low-level helper to publish a JSON message."""
        try:
            channel = await RabbitMQConnection.get_channel()
            queue = await channel.declare_queue(queue_name, durable=True)
            body = json.dumps(payload, default=str).encode("utf-8")
            message = Message(body=body, delivery_mode=2)
            await channel.default_exchange.publish(message, routing_key=queue.name)
            logger.debug("event_published", queue=queue.name, event_type=payload.get("event_type"))
        except Exception as e:  # pragma: no cover - best-effort path
            logger.error("event_publishing_failed", queue=queue_name, error=str(e), exc_info=True)

    @classmethod
    async def publish_position_updated(
        cls,
        position: Position,
        update_source: str,
        trace_id: Optional[str],
    ) -> None:
        """Publish position_updated event with ML features."""
        payload: Dict[str, Any] = {
            "event_type": "position_updated",
            "position_id": str(position.id),
            "asset": position.asset,
            "size": str(position.size),
            "unrealized_pnl": str(position.unrealized_pnl),
            "realized_pnl": str(position.realized_pnl),
            "mode": position.mode,
            "unrealized_pnl_pct": (
                str(position.unrealized_pnl_pct) if position.unrealized_pnl_pct is not None else None
            ),
            "time_held_minutes": position.time_held_minutes,
            "position_size_norm": None,  # can be enriched by consumers if needed
            "update_source": update_source,
            "trace_id": trace_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        await cls._publish(cls.POSITION_QUEUE, payload)

    @classmethod
    async def publish_portfolio_updated(
        cls,
        metrics: PortfolioMetrics,
        trace_id: Optional[str],
    ) -> None:
        """Publish portfolio_updated event with aggregate metrics."""
        payload: Dict[str, Any] = {
            "event_type": "portfolio_updated",
            "total_exposure_usdt": str(metrics.total_exposure_usdt),
            "total_unrealized_pnl_usdt": str(metrics.total_unrealized_pnl_usdt),
            "total_realized_pnl_usdt": str(metrics.total_realized_pnl_usdt),
            "open_positions_count": metrics.open_positions_count,
            "trace_id": trace_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        await cls._publish(cls.PORTFOLIO_QUEUE, payload)


