"""Signal consumer for processing trading signals from RabbitMQ queue."""

import json
from typing import Optional

import aio_pika
from aio_pika import Message

from ..config.rabbitmq import RabbitMQConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..models.trading_signal import TradingSignal
from ..services.signal_processor import SignalProcessor
from ..utils.tracing import generate_trace_id, set_trace_id
from ..exceptions import QueueError, OrderExecutionError

logger = get_logger(__name__)


class SignalConsumer:
    """RabbitMQ consumer for trading signals from model-service.trading_signals queue."""

    def __init__(self):
        """Initialize signal consumer."""
        self.signal_processor = SignalProcessor()
        self.queue_name = "model-service.trading_signals"
        self._consumer_tag = f"order-manager-signal-consumer-{id(self)}"

    async def start(self) -> None:
        """Start consuming signals from RabbitMQ queue."""
        try:
            channel = await RabbitMQConnection.get_channel()

            # Declare queue (ensure it exists)
            queue = await channel.declare_queue(
                self.queue_name,
                durable=True,  # Queue survives broker restart
            )

            logger.info(
                "signal_consumer_starting",
                queue_name=self.queue_name,
            )

            # Start consuming messages
            await queue.consume(
                self._process_message,
                consumer_tag=self._consumer_tag,
            )

            logger.info(
                "signal_consumer_started",
                queue_name=self.queue_name,
            )

        except Exception as e:
            logger.error(
                "signal_consumer_start_failed",
                queue_name=self.queue_name,
                error=str(e),
            )
            raise QueueError(f"Failed to start signal consumer: {e}") from e

    async def stop(self) -> None:
        """Stop consuming signals from RabbitMQ queue."""
        try:
            channel = await RabbitMQConnection.get_channel()

            if self._consumer_tag:
                await channel.cancel(self._consumer_tag)

            logger.info(
                "signal_consumer_stopped",
                queue_name=self.queue_name,
            )

        except Exception as e:
            logger.error(
                "signal_consumer_stop_failed",
                queue_name=self.queue_name,
                error=str(e),
            )

    async def _process_message(self, message: aio_pika.IncomingMessage) -> None:
        """Process a single message from the queue.

        Args:
            message: RabbitMQ message containing trading signal
        """
        trace_id = None
        try:
            # Extract trace ID from message headers if available
            if message.headers:
                trace_id = message.headers.get("trace_id")
                if isinstance(trace_id, bytes):
                    trace_id = trace_id.decode("utf-8")

            # Generate new trace ID if not present
            if not trace_id:
                trace_id = generate_trace_id()

            # Set trace ID in context
            set_trace_id(trace_id)

            # Parse message body
            body = message.body.decode("utf-8")
            signal_data = json.loads(body)

            logger.info(
                "signal_message_received",
                queue_name=self.queue_name,
                message_id=message.message_id,
                trace_id=trace_id,
            )

            # Create TradingSignal object
            signal = TradingSignal.from_dict(signal_data)
            # Override trace_id from message if available
            if trace_id:
                signal.trace_id = trace_id

            # Process signal
            order = await self.signal_processor.process_signal(signal)

            if order:
                logger.info(
                    "signal_processed_successfully",
                    signal_id=str(signal.signal_id),
                    order_id=str(order.id),
                    trace_id=trace_id,
                )
            else:
                logger.warning(
                    "signal_processed_rejected",
                    signal_id=str(signal.signal_id),
                    trace_id=trace_id,
                )

            # Acknowledge message on successful processing
            await message.ack()

        except json.JSONDecodeError as e:
            logger.error(
                "signal_message_parse_error",
                queue_name=self.queue_name,
                error=str(e),
                trace_id=trace_id,
            )
            # Message will be rejected and sent to dead letter queue
            await message.nack(requeue=False)

        except OrderExecutionError as e:
            logger.error(
                "signal_processing_error",
                queue_name=self.queue_name,
                error=str(e),
                trace_id=trace_id,
            )
            # Message will be rejected and sent to dead letter queue
            await message.nack(requeue=False)

        except Exception as e:
            logger.error(
                "signal_processing_unexpected_error",
                queue_name=self.queue_name,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            # Message will be rejected and sent to dead letter queue
            await message.nack(requeue=False)

        finally:
            # Clear trace ID from context
            set_trace_id(None)

