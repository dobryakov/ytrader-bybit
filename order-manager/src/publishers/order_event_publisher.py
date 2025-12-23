"""Order event publisher service for publishing enriched order events to RabbitMQ."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from aio_pika import Message

from ..config.rabbitmq import RabbitMQConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..models.order import Order
from ..exceptions import QueueError
from ..utils.tracing import get_or_create_trace_id
from common.trading_events import trading_events_publisher

logger = get_logger(__name__)


class OrderEventPublisher:
    """Service for publishing enriched order execution events to RabbitMQ."""

    def __init__(self):
        """Initialize order event publisher."""
        self.queue_name = "order-manager.order_events"
        self._exchange_name = ""  # Default exchange (direct routing)

    async def publish_order_event(
        self,
        order: Order,
        event_type: str,
        trace_id: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish enriched order event to RabbitMQ queue.

        Args:
            order: Order object
            event_type: Event type ('filled', 'partially_filled', 'cancelled', 'rejected', 'created', 'modified')
            trace_id: Optional trace ID
            rejection_reason: Optional rejection reason (for rejected orders)
            before_state: Optional before state (for modifications)
            after_state: Optional after state (for modifications)
            market_conditions: Optional market conditions at time of event
            signal_info: Optional signal information (signal_id, signal_type, confidence)
        """
        trace_id = trace_id or order.trace_id or get_or_create_trace_id()

        try:
            # Enrich event with additional data
            enriched_event = self._enrich_event(
                order=order,
                event_type=event_type,
                trace_id=trace_id,
                rejection_reason=rejection_reason,
                before_state=before_state,
                after_state=after_state,
                market_conditions=market_conditions,
                signal_info=signal_info,
            )

            # Publish to RabbitMQ
            await self._publish_to_queue(enriched_event, trace_id)

            logger.info(
                "order_event_published",
                order_id=order.order_id,
                event_type=event_type,
                queue_name=self.queue_name,
                trace_id=trace_id,
            )

            # Publish trading event for product analytics (non-blocking best-effort)
            try:
                await self._publish_trading_event(
                    order=order,
                    event_type=event_type,
                    trace_id=trace_id,
                    signal_info=signal_info,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "trading_event_publish_failed",
                    order_id=order.order_id,
                    event_type=event_type,
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )

        except Exception as e:
            logger.error(
                "order_event_publish_failed",
                order_id=order.order_id,
                event_type=event_type,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            # Don't raise - event publishing failure shouldn't block order processing
            # Log error but continue execution

    def _enrich_event(
        self,
        order: Order,
        event_type: str,
        trace_id: str,
        rejection_reason: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        signal_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enrich order event with execution details, market conditions, and timing information.

        Args:
            order: Order object
            event_type: Event type
            trace_id: Trace ID
            rejection_reason: Optional rejection reason
            before_state: Optional before state
            after_state: Optional after state
            market_conditions: Optional market conditions
            signal_info: Optional signal information

        Returns:
            Enriched event dictionary
        """
        # Calculate execution latency (time from order creation to execution)
        execution_latency: Optional[float] = None
        if order.executed_at and order.created_at:
            latency_delta = order.executed_at - order.created_at
            execution_latency = latency_delta.total_seconds()

        # Calculate fill percentage
        fill_percentage: Optional[float] = None
        if order.quantity > 0:
            fill_percentage = float(order.filled_quantity / order.quantity * 100)

        # Build enriched event
        enriched_event = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "order": {
                "id": str(order.id),
                "order_id": order.order_id,
                "signal_id": str(order.signal_id),
                "asset": order.asset,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": str(order.quantity),
                "price": str(order.price) if order.price else None,
                "status": order.status,
                "filled_quantity": str(order.filled_quantity),
                "average_price": str(order.average_price) if order.average_price else None,
                "fees": str(order.fees) if order.fees else None,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                "executed_at": order.executed_at.isoformat() if order.executed_at else None,
                "is_dry_run": order.is_dry_run,
            },
            "execution_details": {
                "execution_latency_seconds": execution_latency,
                "fill_percentage": fill_percentage,
                "remaining_quantity": str(order.quantity - order.filled_quantity),
            },
        }

        # Add rejection reason if provided
        if rejection_reason:
            enriched_event["rejection_reason"] = rejection_reason

        # Add modification details if provided
        if before_state or after_state:
            enriched_event["modification"] = {
                "before": before_state,
                "after": after_state,
            }

        # Add market conditions if provided
        if market_conditions:
            enriched_event["market_conditions"] = market_conditions

        # Add signal information if provided
        if signal_info:
            enriched_event["signal"] = signal_info

        return enriched_event

    async def _publish_trading_event(
        self,
        *,
        order: Order,
        event_type: str,
        trace_id: str,
        signal_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish simplified trading event to trading_events exchange for Kibana analytics.

        Maps internal order events to product-oriented event types and payload.
        """
        # Map internal event_type to product-level name
        event_type_map = {
            "created": "order_created",
            "filled": "order_filled",
            "partially_filled": "order_partially_filled",
            "cancelled": "order_cancelled",
            "rejected": "order_rejected",
            "modified": "order_modified",
        }
        product_event_type = event_type_map.get(event_type, f"order_{event_type}")

        payload: Dict[str, Any] = {
            "signal_id": str(order.signal_id),
            "order_id": order.order_id,
            "order_internal_id": str(order.id),
            "asset": order.asset,
            "side": order.side,
            "type": order.order_type,
            "status": order.status,
            "quantity": str(order.quantity),
            "filled_quantity": str(order.filled_quantity),
            "price": str(order.price) if order.price is not None else None,
            "average_price": str(order.average_price) if order.average_price is not None else None,
            "fees": str(order.fees) if order.fees is not None else None,
            "is_dry_run": order.is_dry_run,
            "rejection_reason": order.rejection_reason,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            "executed_at": order.executed_at.isoformat() if order.executed_at else None,
        }

        if signal_info:
            payload["strategy_id"] = signal_info.get("strategy_id")
            payload["signal_type"] = signal_info.get("signal_type")
            payload["signal_confidence"] = signal_info.get("confidence")

        await trading_events_publisher.publish_trading_signal_event(
            event_type=product_event_type,
            service="order-manager",
            signal_payload=payload,
            trace_id=trace_id,
        )

    async def _publish_to_queue(self, event: Dict[str, Any], trace_id: str) -> None:
        """
        Publish event to RabbitMQ queue.

        Args:
            event: Event dictionary
            trace_id: Trace ID
        """
        try:
            channel = await RabbitMQConnection.get_channel()

            # Declare queue (ensure it exists)
            queue = await channel.declare_queue(
                self.queue_name,
                durable=True,  # Queue survives broker restart
            )

            # Serialize event to JSON
            event_json = json.dumps(event, default=str)

            # Create message with trace ID in headers
            message = Message(
                event_json.encode("utf-8"),
                headers={"trace_id": trace_id},
                content_type="application/json",
            )

            # Publish message
            await channel.default_exchange.publish(
                message,
                routing_key=self.queue_name,
            )

            logger.debug(
                "order_event_published_to_queue",
                queue_name=self.queue_name,
                event_id=event.get("event_id"),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "order_event_queue_publish_failed",
                queue_name=self.queue_name,
                error=str(e),
                trace_id=trace_id,
            )
            raise QueueError(f"Failed to publish order event to queue: {e}") from e

