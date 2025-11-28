"""
Trading signal publisher.

Publishes trading signals to RabbitMQ queue for order manager consumption
and persists them to PostgreSQL database for Grafana monitoring dashboard visibility.
"""

import json
from typing import Optional

import aio_pika

from ..models.signal import TradingSignal
from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger, bind_context
from ..config.exceptions import MessageQueueError
from ..database.repositories.trading_signal_repo import TradingSignalRepository

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
        self.trading_signal_repo = TradingSignalRepository()

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
        Publish a trading signal to RabbitMQ with retry logic.

        Args:
            signal: TradingSignal to publish

        Raises:
            MessageQueueError: If publishing fails after retries
        """
        from ..config.retry import retry_async

        if not self._queue:
            await self.initialize()

        bind_context(
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            trace_id=signal.trace_id,
        )

        async def _publish_attempt():
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

        try:
            await retry_async(
                _publish_attempt,
                max_retries=3,
                initial_delay=0.5,
                max_delay=5.0,
                operation_name="publish_signal",
            )

            logger.info(
                "Published trading signal",
                signal_id=signal.signal_id,
                signal_type=signal.signal_type,
                asset=signal.asset,
                amount=signal.amount,
                strategy_id=signal.strategy_id,
                model_version=signal.model_version,
                is_warmup=signal.is_warmup,
                confidence=signal.confidence,
                queue=self.queue_name,
            )

            # Persist trading signal to database after successful RabbitMQ publish
            try:
                logger.info("Attempting to persist trading signal to database", signal_id=signal.signal_id)
                await self._persist_signal(signal)
                logger.info("Successfully persisted trading signal to database", signal_id=signal.signal_id)
            except Exception as e:
                # Log error but don't raise - continue processing on persistence failures
                # This ensures signals are still published to RabbitMQ even if database persistence fails
                logger.warning(
                    "Failed to persist trading signal to database (continuing)",
                    signal_id=signal.signal_id,
                    error=str(e),
                    exc_info=True,
                )

        except Exception as e:
            logger.error(
                "Failed to publish trading signal after retries",
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

    async def _persist_signal(self, signal: TradingSignal) -> None:
        """
        Persist trading signal to PostgreSQL database.

        Args:
            signal: TradingSignal to persist

        Note:
            This method handles database errors gracefully and continues processing
            even if persistence fails, as per T090 requirements. Signals are persisted
            for Grafana monitoring dashboard visibility.
        """
        try:
            # Extract price from market_data_snapshot (required field)
            price = signal.market_data_snapshot.price

            # Convert market_data_snapshot to dict for JSONB storage
            market_data_dict = {
                "price": signal.market_data_snapshot.price,
                "spread": signal.market_data_snapshot.spread,
                "volume_24h": signal.market_data_snapshot.volume_24h,
                "volatility": signal.market_data_snapshot.volatility,
            }
            if signal.market_data_snapshot.orderbook_depth:
                market_data_dict["orderbook_depth"] = signal.market_data_snapshot.orderbook_depth
            if signal.market_data_snapshot.technical_indicators:
                market_data_dict["technical_indicators"] = signal.market_data_snapshot.technical_indicators

            await self.trading_signal_repo.create(
                signal_id=signal.signal_id,
                strategy_id=signal.strategy_id,
                asset=signal.asset,
                side=signal.signal_type,  # signal_type is 'buy' or 'sell'
                price=price,
                confidence=signal.confidence,
                timestamp=signal.timestamp,
                model_version=signal.model_version,
                is_warmup=signal.is_warmup,
                market_data_snapshot=market_data_dict,
                metadata=signal.metadata,
                trace_id=signal.trace_id,
            )
            logger.info(
                "Trading signal persisted to database",
                signal_id=signal.signal_id,
                strategy_id=signal.strategy_id,
                asset=signal.asset,
            )
        except Exception as e:
            # Log error but don't raise - continue processing on persistence failures
            logger.warning(
                "Failed to persist trading signal (continuing)",
                signal_id=signal.signal_id,
                error=str(e),
            )


# Global signal publisher instance
signal_publisher = SignalPublisher()

