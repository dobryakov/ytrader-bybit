"""Event routing logic to determine target queue by event class."""

from ...config.logging import get_logger
from ...models.event import Event, EventType

logger = get_logger(__name__)


def get_queue_name_for_event(event: Event) -> str:
    """
    Determine target queue name for an event based on event class.

    Queue naming convention: ws-gateway.{event_class}
    Examples:
        - trade events -> ws-gateway.trade
        - order events -> ws-gateway.order
        - balance events -> ws-gateway.balance

    Args:
        event: Event to route

    Returns:
        Queue name for the event
    """
    queue_name = f"ws-gateway.{event.event_type}"
    logger.debug(
        "event_routed",
        event_id=str(event.event_id),
        event_type=event.event_type,
        queue_name=queue_name,
        trace_id=event.trace_id,
    )
    return queue_name


def get_queue_name_for_event_type(event_type: EventType) -> str:
    """
    Get queue name for an event type.

    Args:
        event_type: Event type

    Returns:
        Queue name for the event type
    """
    return f"ws-gateway.{event_type}"

