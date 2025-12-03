"""Event processing pipeline that integrates queue publishing and balance persistence."""

from typing import List

from ...config.logging import get_logger
from ...models.event import Event
from ..database.balance_service import BalanceService
from ..positions.position_event_normalizer import PositionEventNormalizer
from ..queue.publisher import get_publisher
from ..queue.router import get_queue_name_for_event

logger = get_logger(__name__)


async def process_events(events: List[Event]) -> None:
    """
    Process events by publishing them to appropriate queues and persisting balance events.

    This function implements the core event processing pipeline for User Stories 4 and 5:
    - Routes events to queues based on event class (User Story 4)
    - Persists balance events to PostgreSQL (User Story 5)
    - Publishes events to RabbitMQ
    - Handles failures gracefully (per FR-017)

    Args:
        events: List of events to process
    """
    if not events:
        return

    publisher = await get_publisher()

    for event in events:
        try:
            # Persist balance events to database (User Story 5)
            # This happens before queue publishing to ensure critical data is stored.
            # Per FR-017, database failures don't block queue publishing.
            if event.event_type == "balance":
                await BalanceService.persist_balance_from_event(event)

            # Position events are normalized and published via PositionEventNormalizer.
            # They are not written directly to the positions table in this service.
            if event.event_type == "position":
                success = await PositionEventNormalizer.normalize_and_publish(event)
                queue_name = get_queue_name_for_event(event)
            else:
                # Determine target queue based on event class
                queue_name = get_queue_name_for_event(event)

                # Publish event to queue
                success = await publisher.publish_event(event, queue_name)

            if success:
                logger.debug(
                    "event_processed",
                    event_id=str(event.event_id),
                    event_type=event.event_type,
                    topic=event.topic,
                    queue_name=queue_name,
                    trace_id=event.trace_id,
                )
            else:
                # Publisher already logged the error
                # Per FR-017, we continue processing other events
                logger.warning(
                    "event_processing_failed",
                    event_id=str(event.event_id),
                    event_type=event.event_type,
                    topic=event.topic,
                    queue_name=queue_name,
                    trace_id=event.trace_id,
                )

        except Exception as e:
            # Per FR-017, queue publishing failures should not block event processing
            logger.error(
                "event_processing_exception",
                event_id=str(event.event_id),
                event_type=event.event_type,
                topic=event.topic,
                trace_id=event.trace_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Continue with next event


async def process_event(event: Event) -> None:
    """
    Process a single event by publishing it to the appropriate queue.

    Args:
        event: Event to process
    """
    await process_events([event])

