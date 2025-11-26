"""
Trading signal publisher.

Publishes trading signals to RabbitMQ queue for order manager consumption.
"""

import json
from typing import Optional

import aio_pika

from ..models.signal import TradingSignal
from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger, bind_context
from ..config.exceptions import MessageQueueError

logger = get_logger(__name__)


class SignalPublisher:
    """Publishes trading signals to RabbitMQ."""

    def __init__(self, queue_name: str = "model-service.trading_signals"):
        """
        Initialize signal publisher.

        Args:
            queue_name: Name of the RabbitMQ queue to publish to
        """
        self.queue_name = queue_name
        self._queue: Optional[aio_pika.Queue] = None

    async def initialize(self) -> None:
        """Initialize publisher (declare queue)."""
        try:
            channel = await rabbitmq_manager.get_channel()
            self._queue = await rabbitmq_manager.get_queue(self.queue_name, durable=True)
            logger.info("Signal publisher initialized", queue=self.queue_name)
        except Exception as e:
            logger.error("Failed to initialize signal publisher", queue=self.queue_name, error=str(e))
            raise MessageQueueError(f"Failed to initialize signal publisher: {e}") from e

    async def publish(self, signal: TradingSignal) -> None:
        """
        Publish a trading signal to RabbitMQ.

        Args:
            signal: TradingSignal to publish

        Raises:
            MessageQueueError: If publishing fails
        """
        if not self._queue:
            await self.initialize()

        bind_context(
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            trace_id=signal.trace_id,
        )

        try:
            # Convert signal to dictionary
            signal_dict = signal.to_dict()

            # Serialize to JSON
            message_body = json.dumps(signal_dict).encode("utf-8")

            # Publish message
            channel = await rabbitmq_manager.get_channel()
            await channel.default_exchange.publish(
                aio_pika.Message(
                    message_body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.queue_name,
            )

            logger.info(
                "Published trading signal",
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                asset=signal.asset,
                amount=signal.amount,
                strategy_id=signal.strategy_id,
                is_warmup=signal.is_warmup,
                queue=self.queue_name,
            )

        except Exception as e:
            logger.error(
                "Failed to publish trading signal",
                signal_id=signal.signal_id,
                error=str(e),
                exc_info=True,
            )
            raise MessageQueueError(f"Failed to publish signal {signal.signal_id}: {e}") from e

    async def publish_batch(self, signals: list[TradingSignal]) -> int:
        """
        Publish multiple trading signals.

        Args:
            signals: List of TradingSignals to publish

        Returns:
            Number of successfully published signals

        Raises:
            MessageQueueError: If publishing fails for all signals
        """
        if not self._queue:
            await self.initialize()

        published_count = 0
        errors = []

        for signal in signals:
            try:
                await self.publish(signal)
                published_count += 1
            except MessageQueueError as e:
                errors.append(f"Signal {signal.signal_id}: {e}")
                logger.warning("Failed to publish signal in batch", signal_id=signal.signal_id, error=str(e))

        if published_count == 0 and signals:
            error_message = f"Failed to publish all signals: {', '.join(errors)}"
            logger.error("Failed to publish any signals in batch", total=len(signals), errors=errors)
            raise MessageQueueError(error_message)

        if errors:
            logger.warning(
                "Some signals failed to publish",
                total=len(signals),
                published=published_count,
                failed=len(errors),
            )

        return published_count


# Global signal publisher instance
signal_publisher = SignalPublisher()

