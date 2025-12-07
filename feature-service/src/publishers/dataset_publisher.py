"""
Dataset completion publisher for RabbitMQ.
"""
from typing import Optional
import json
import structlog
import aio_pika
import aiormq.exceptions

from src.mq.connection import MQConnectionManager

logger = structlog.get_logger(__name__)


class DatasetPublisher:
    """Publishes dataset completion notifications to RabbitMQ."""
    
    def __init__(self, mq_manager: MQConnectionManager):
        """
        Initialize dataset publisher.
        
        Args:
            mq_manager: RabbitMQ connection manager
        """
        self._mq_manager = mq_manager
        self._channel: Optional[aio_pika.Channel] = None
    
    async def initialize(self) -> None:
        """Initialize publisher channel."""
        self._channel = await self._mq_manager.get_channel()
        logger.info("dataset_publisher_initialized", queue="features.dataset.ready")
    
    async def publish_dataset_ready(
        self,
        dataset_id: str,
        symbol: str,
        status: str,
        train_records: int,
        validation_records: int,
        test_records: int,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Publish dataset completion notification.
        
        Args:
            dataset_id: Dataset ID
            symbol: Trading pair symbol
            status: Dataset status (ready, failed)
            train_records: Number of train records
            validation_records: Number of validation records
            test_records: Number of test records
            trace_id: Optional trace ID
        """
        # Check if channel is None, reinitialize if needed
        if self._channel is None:
            await self.initialize()
        
        message_data = {
            "dataset_id": dataset_id,
            "symbol": symbol,
            "status": status,
            "train_records": train_records,
            "validation_records": validation_records,
            "test_records": test_records,
            "trace_id": trace_id,
        }
        
        message_body = json.dumps(message_data).encode()
        
        # Try to publish - handle channel errors gracefully
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Publish message
                await self._channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message_body,
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key="features.dataset.ready",
                )
                
                logger.info(
                    "dataset_ready_published",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    status=status,
                    queue="features.dataset.ready",
                )
                return  # Success - exit function
            
            except (aiormq.exceptions.ChannelInvalidStateError, AttributeError) as e:
                # Channel is closed or invalid - reinitialize and retry
                error_type = type(e).__name__
                logger.warning(
                    "channel_closed_during_publish",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    error=str(e),
                    error_type=error_type,
                    queue="features.dataset.ready",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                
                if attempt < max_retries - 1:
                    # Reinitialize channel before retry
                    try:
                        await self.initialize()
                    except Exception as init_error:
                        logger.error(
                            "failed_to_reinitialize_channel",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            error=str(init_error),
                            exc_info=True,
                        )
                        # If reinitialization fails, log error and give up
                        break
                else:
                    # Last attempt failed - log error
                    logger.error(
                        "dataset_publish_error_after_retries",
                        dataset_id=dataset_id,
                        symbol=symbol,
                        error=str(e),
                        error_type=error_type,
                        exc_info=True,
                    )
            
            except Exception as e:
                # Other error - log and don't retry
                error_type = type(e).__name__
                logger.error(
                    "dataset_publish_error",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    error=str(e),
                    error_type=error_type,
                    exc_info=True,
                )
                return  # Don't retry for other errors
