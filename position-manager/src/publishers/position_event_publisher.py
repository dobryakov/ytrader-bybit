"""RabbitMQ publishers for position and portfolio events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aio_pika import Message

from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..models import PortfolioMetrics, Position, PositionSnapshot
from common.trading_events import trading_events_publisher

logger = get_logger(__name__)


class PositionEventPublisher:
    """Publish position and portfolio events to RabbitMQ."""

    POSITION_QUEUE = "position-manager.position_updated"
    PORTFOLIO_QUEUE = "position-manager.portfolio_updated"
    SNAPSHOT_QUEUE = "position-manager.position_snapshot_created"

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

        # Best-effort публикация события в trading_events для продуктовой аналитики
        try:
            event_type = "position_closed" if position.size == 0 and position.closed_at else "position_updated"
            trading_payload: Dict[str, Any] = {
                "position_id": str(position.id),
                "asset": position.asset,
                "mode": position.mode,
                "size": str(position.size),
                "average_entry_price": str(position.average_entry_price)
                if position.average_entry_price is not None
                else None,
                "current_price": str(position.current_price) if position.current_price is not None else None,
                "unrealized_pnl": str(position.unrealized_pnl),
                "realized_pnl": str(position.realized_pnl),
                "unrealized_pnl_pct": str(position.unrealized_pnl_pct)
                if position.unrealized_pnl_pct is not None
                else None,
                "time_held_minutes": position.time_held_minutes,
                "closed_at": position.closed_at.replace(tzinfo=timezone.utc).isoformat()
                if position.closed_at
                else None,
                "created_at": position.created_at.replace(tzinfo=timezone.utc).isoformat()
                if hasattr(position.created_at, "isoformat")
                else None,
                "update_source": update_source,
            }

            await trading_events_publisher.publish_event(
                {
                    "event_type": event_type,
                    "service": "position-manager",
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "level": "info",
                    "payload": trading_payload,
                    "trace_id": trace_id,
                }
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "trading_event_position_publish_failed",
                position_id=str(position.id),
                asset=position.asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )

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

    @classmethod
    async def publish_snapshot_created(
        cls,
        snapshot: PositionSnapshot,
        trace_id: Optional[str],
    ) -> None:
        """Publish position_snapshot_created event with full snapshot payload."""
        payload: Dict[str, Any] = {
            "event_type": "position_snapshot_created",
            "snapshot_id": str(snapshot.id),
            "position_id": str(snapshot.position_id),
            "asset": snapshot.asset,
            "mode": snapshot.mode,
            "snapshot_data": snapshot.snapshot_data,
            "created_at": snapshot.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "trace_id": trace_id,
        }
        await cls._publish(cls.SNAPSHOT_QUEUE, payload)


