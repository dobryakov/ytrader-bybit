"""Event subscriber service for WebSocket order execution events."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

import aio_pika
import httpx
from aio_pika import Message

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..config.settings import settings
from ..models.order import Order
from ..services.position_manager import PositionManager
from ..publishers.order_event_publisher import OrderEventPublisher
from ..exceptions import DatabaseError, QueueError
from ..utils.tracing import generate_trace_id, set_trace_id

logger = get_logger(__name__)


class EventSubscriber:
    """Service for subscribing to and processing order execution events from WebSocket gateway."""

    def __init__(self):
        """Initialize event subscriber service."""
        self.queue_name = "ws-gateway.order"
        self._consumer_tag = f"order-manager-event-subscriber-{id(self)}"
        self.position_manager = PositionManager()
        self.event_publisher = OrderEventPublisher()
        self._subscription_id: Optional[str] = None

    async def subscribe_to_order_events(self, trace_id: Optional[str] = None) -> None:
        """
        Subscribe to order execution events from WebSocket gateway.

        Creates a subscription via REST API to receive order status updates.

        Args:
            trace_id: Optional trace ID for request tracking
        """
        trace_id = trace_id or generate_trace_id()

        try:
            # Subscribe to order events via WebSocket gateway REST API
            # Channel type: "order" (no symbol required - receives all order events)
            subscription_data = {
                "channel_type": "order",
                "requesting_service": settings.order_manager_service_name,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.ws_gateway_url}/api/v1/subscriptions",
                    headers={
                        "X-API-Key": settings.ws_gateway_api_key,
                        "Content-Type": "application/json",
                    },
                    json=subscription_data,
                )

                if response.status_code == 201:
                    result = response.json()
                    self._subscription_id = result.get("id")
                    logger.info(
                        "order_events_subscription_created",
                        subscription_id=self._subscription_id,
                        trace_id=trace_id,
                    )
                elif response.status_code == 409:
                    # Subscription already exists - get existing subscription
                    logger.info(
                        "order_events_subscription_exists",
                        trace_id=trace_id,
                    )
                    # Try to get existing subscriptions for this service
                    list_response = await client.get(
                        f"{settings.ws_gateway_url}/api/v1/subscriptions",
                        headers={"X-API-Key": settings.ws_gateway_api_key},
                        params={
                            "requesting_service": settings.order_manager_service_name,
                            "channel_type": "order",
                            "is_active": True,
                        },
                    )
                    if list_response.status_code == 200:
                        subscriptions = list_response.json().get("subscriptions", [])
                        if subscriptions:
                            self._subscription_id = subscriptions[0].get("id")
                            logger.info(
                                "order_events_subscription_found",
                                subscription_id=self._subscription_id,
                                trace_id=trace_id,
                            )
                else:
                    error_msg = f"Failed to create subscription: {response.status_code} - {response.text}"
                    logger.error(
                        "order_events_subscription_failed",
                        status_code=response.status_code,
                        error=error_msg,
                        trace_id=trace_id,
                    )
                    raise QueueError(error_msg)

        except httpx.HTTPError as e:
            logger.error(
                "order_events_subscription_http_error",
                error=str(e),
                trace_id=trace_id,
            )
            raise QueueError(f"HTTP error creating subscription: {e}") from e
        except Exception as e:
            logger.error(
                "order_events_subscription_error",
                error=str(e),
                trace_id=trace_id,
            )
            raise QueueError(f"Failed to subscribe to order events: {e}") from e

    async def start(self) -> None:
        """Start consuming order execution events from RabbitMQ queue."""
        try:
            # First, subscribe to order events via WebSocket gateway
            await self.subscribe_to_order_events()

            # Then, start consuming from RabbitMQ queue
            channel = await RabbitMQConnection.get_channel()

            # Declare queue (ensure it exists)
            queue = await channel.declare_queue(
                self.queue_name,
                durable=True,  # Queue survives broker restart
            )

            logger.info(
                "event_subscriber_starting",
                queue_name=self.queue_name,
            )

            # Start consuming messages
            await queue.consume(
                self._process_message,
                consumer_tag=self._consumer_tag,
            )

            logger.info(
                "event_subscriber_started",
                queue_name=self.queue_name,
            )

        except Exception as e:
            logger.error(
                "event_subscriber_start_failed",
                queue_name=self.queue_name,
                error=str(e),
            )
            raise QueueError(f"Failed to start event subscriber: {e}") from e

    async def stop(self) -> None:
        """Stop consuming order execution events."""
        try:
            channel = await RabbitMQConnection.get_channel()

            if self._consumer_tag:
                await channel.cancel(self._consumer_tag)

            logger.info(
                "event_subscriber_stopped",
                queue_name=self.queue_name,
            )

        except Exception as e:
            logger.error(
                "event_subscriber_stop_failed",
                queue_name=self.queue_name,
                error=str(e),
            )

    async def _process_message(self, message: aio_pika.IncomingMessage) -> None:
        """Process a single order execution event message.

        Args:
            message: RabbitMQ message containing order execution event
        """
        trace_id = None
        async with message.process():
            try:
                # Extract trace ID from message headers if available
                if message.headers:
                    trace_id = message.headers.get("trace_id")
                    if isinstance(trace_id, bytes):
                        trace_id = trace_id.decode("utf-8")

                # Generate new trace ID if not present
                if not trace_id:
                    trace_id = generate_trace_id()

                set_trace_id(trace_id)

                # Parse message body
                body = message.body.decode("utf-8")
                event_data = json.loads(body)

                logger.debug(
                    "order_event_received",
                    event_id=event_data.get("event_id"),
                    event_type=event_data.get("event_type"),
                    trace_id=trace_id,
                )

                # Process order execution event
                await self._handle_order_event(event_data, trace_id)

            except json.JSONDecodeError as e:
                logger.error(
                    "order_event_json_decode_failed",
                    error=str(e),
                    body=message.body.decode("utf-8")[:500],  # Log first 500 chars
                    trace_id=trace_id,
                )
            except Exception as e:
                logger.error(
                    "order_event_processing_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )

    async def _handle_order_event(
        self, event_data: dict, trace_id: Optional[str] = None
    ) -> None:
        """
        Handle order execution event and update order state.

        Args:
            event_data: Event data from WebSocket gateway
            trace_id: Optional trace ID
        """
        try:
            # Extract event payload
            payload = event_data.get("payload", {})
            event_type = event_data.get("event_type", "")

            # Extract order information from payload
            bybit_order_id = payload.get("orderId") or payload.get("order_id")
            if not bybit_order_id:
                logger.warning(
                    "order_event_missing_order_id",
                    event_data=event_data,
                    trace_id=trace_id,
                )
                return

            # Find order in database by Bybit order ID
            order = await self._get_order_by_bybit_id(bybit_order_id, trace_id)
            if not order:
                logger.warning(
                    "order_not_found_for_event",
                    bybit_order_id=bybit_order_id,
                    event_type=event_type,
                    trace_id=trace_id,
                )
                return

            # Update order state based on event type
            await self._update_order_state_from_event(
                order, event_data, payload, trace_id
            )

        except Exception as e:
            logger.error(
                "order_event_handle_failed",
                error=str(e),
                event_data=event_data,
                trace_id=trace_id,
                exc_info=True,
            )

    async def _get_order_by_bybit_id(
        self, bybit_order_id: str, trace_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Get order from database by Bybit order ID.

        Args:
            bybit_order_id: Bybit order ID
            trace_id: Optional trace ID

        Returns:
            Order object if found, None otherwise
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                       status, filled_quantity, average_price, fees, created_at, updated_at,
                       executed_at, trace_id, is_dry_run
                FROM orders
                WHERE order_id = $1
            """
            row = await pool.fetchrow(query, bybit_order_id)

            if row is None:
                return None

            return Order.from_dict(dict(row))

        except Exception as e:
            logger.error(
                "order_query_by_bybit_id_failed",
                bybit_order_id=bybit_order_id,
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to query order: {e}") from e

    async def _update_order_state_from_event(
        self,
        order: Order,
        event_data: dict,
        payload: dict,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Update order state based on execution event.

        Handles filled, partially_filled, cancelled, rejected events.

        Args:
            order: Order object to update
            event_data: Full event data
            payload: Event payload with order details
            trace_id: Optional trace ID
        """
        try:
            # Extract order status and execution details from payload
            bybit_status = payload.get("orderStatus") or payload.get("status", "")
            filled_qty = Decimal(str(payload.get("cumExecQty") or payload.get("executed_qty") or "0"))
            avg_price = (
                Decimal(str(payload.get("avgPrice") or payload.get("avg_price") or "0"))
                if payload.get("avgPrice") or payload.get("avg_price")
                else None
            )
            fees = (
                Decimal(str(payload.get("cumExecFee") or payload.get("fees") or "0")))
                if payload.get("cumExecFee") or payload.get("fees")
                else None
            )

            # Map Bybit status to our status
            status_map = {
                "New": "pending",
                "PartiallyFilled": "partially_filled",
                "Filled": "filled",
                "Cancelled": "cancelled",
                "Rejected": "rejected",
            }
            new_status = status_map.get(bybit_status, bybit_status.lower())

            # Determine if status changed
            status_changed = order.status != new_status

            # Update order in database
            pool = await DatabaseConnection.get_pool()

            # Determine executed_at timestamp if order is filled
            executed_at = order.executed_at
            if new_status == "filled" and order.status != "filled":
                executed_at = datetime.utcnow()

            update_query = """
                UPDATE orders
                SET status = $1,
                    filled_quantity = $2,
                    average_price = $3,
                    fees = $4,
                    updated_at = NOW(),
                    executed_at = $5
                WHERE id = $6
            """
            await pool.execute(
                update_query,
                new_status,
                str(filled_qty),
                str(avg_price) if avg_price else None,
                str(fees) if fees else None,
                executed_at,
                str(order.id),
            )

            logger.info(
                "order_state_updated_from_event",
                order_id=order.order_id,
                old_status=order.status,
                new_status=new_status,
                filled_quantity=float(filled_qty),
                status_changed=status_changed,
                trace_id=trace_id,
            )

            # Get updated order from database to publish event with latest state
            updated_order = await self._get_order_by_bybit_id(order.order_id, trace_id)
            if not updated_order:
                logger.warning(
                    "order_not_found_after_update",
                    order_id=order.order_id,
                    trace_id=trace_id,
                )
                updated_order = order  # Fallback to original order

            # Publish order event if status changed
            if status_changed:
                event_type = new_status
                # Map status to event type
                if new_status == "partially_filled":
                    event_type = "partially_filled"
                elif new_status == "filled":
                    event_type = "filled"
                elif new_status == "cancelled":
                    event_type = "cancelled"
                elif new_status == "rejected":
                    event_type = "rejected"

                # Extract market conditions from event_data if available
                market_conditions = None
                if event_data.get("payload"):
                    payload = event_data.get("payload", {})
                    # Try to extract market data if available
                    if "price" in payload or "avgPrice" in payload:
                        market_conditions = {
                            "price": float(avg_price) if avg_price else None,
                            "timestamp": event_data.get("timestamp"),
                        }

                # Publish enriched order event
                await self.event_publisher.publish_order_event(
                    order=updated_order,
                    event_type=event_type,
                    trace_id=trace_id,
                    market_conditions=market_conditions,
                )

            # Update position if order was filled (fully or partially)
            if new_status in ["filled", "partially_filled"] and status_changed:
                await self._update_position_from_order_fill(
                    order, filled_qty, avg_price, trace_id
                )

        except Exception as e:
            logger.error(
                "order_state_update_from_event_failed",
                order_id=order.order_id,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to update order state: {e}") from e

    async def _update_position_from_order_fill(
        self,
        order: Order,
        filled_quantity: Decimal,
        execution_price: Decimal,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Update position when order is filled.

        Args:
            order: Order that was filled
            filled_quantity: Quantity that was filled
            execution_price: Price at which order was executed
            trace_id: Optional trace ID
        """
        try:
            # Calculate size delta based on order side
            # Buy orders increase position (positive), Sell orders decrease position (negative)
            if order.side.upper() == "BUY":
                size_delta = filled_quantity
            else:  # SELL
                size_delta = -filled_quantity

            # Update position
            await self.position_manager.update_position(
                asset=order.asset,
                size_delta=size_delta,
                execution_price=execution_price,
                mode="one-way",  # Default to one-way mode
                trace_id=trace_id,
            )

            logger.info(
                "position_updated_from_order_fill",
                order_id=order.order_id,
                asset=order.asset,
                side=order.side,
                filled_quantity=float(filled_quantity),
                execution_price=float(execution_price),
                size_delta=float(size_delta),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "position_update_from_order_fill_failed",
                order_id=order.order_id,
                asset=order.asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            # Don't raise - position update failure shouldn't block order state update

