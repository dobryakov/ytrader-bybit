"""
Feature publisher for publishing computed features to RabbitMQ queue.
"""
import json
from typing import Optional
from datetime import datetime
import asyncio
import structlog
import aio_pika
import aiormq.exceptions

from src.mq.connection import MQConnectionManager
from src.models.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)


class FeaturePublisher:
    """Publishes computed features to RabbitMQ queue."""
    
    def __init__(
        self,
        mq_manager: MQConnectionManager,
        queue_name: str = "features.live",
    ):
        """Initialize feature publisher."""
        self._mq_manager = mq_manager
        self._queue_name = queue_name
        self._channel: Optional[aio_pika.Channel] = None
        self._queue: Optional[aio_pika.Queue] = None
    
    async def initialize(self) -> None:
        """Initialize publisher (declare queue)."""
        self._channel = await self._mq_manager.get_channel()
        self._queue = await self._channel.declare_queue(self._queue_name, durable=True)
        logger.info("feature_publisher_initialized", queue=self._queue_name)
    
    async def publish(self, feature_vector: FeatureVector) -> None:
        """Publish feature vector to queue."""
        # Check if channel/queue is None, reinitialize if needed
        # Note: We don't check is_closed here as it may be unreliable for RobustChannel
        if self._queue is None or self._channel is None:
            await self.initialize()
        
        # Serialize feature vector once
        try:
            data = feature_vector.model_dump(mode='json')
        except (TypeError, AttributeError):
            # Fallback for Pydantic v1 or if mode='json' is not supported
            data = feature_vector.model_dump()
            # Manually convert datetime to ISO string
            if isinstance(data.get("timestamp"), datetime):
                data["timestamp"] = data["timestamp"].isoformat()
        message_body = json.dumps(data).encode()
        
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
                    routing_key=self._queue_name,
                )
                
                logger.debug(
                    "feature_vector_published",
                    symbol=feature_vector.symbol,
                    timestamp=feature_vector.timestamp.isoformat(),
                    features_count=len(feature_vector.features),
                    trace_id=feature_vector.trace_id,
                )
                return  # Success - exit function
            
            except (aiormq.exceptions.ChannelInvalidStateError, AttributeError, RuntimeError) as e:
                # Channel is closed or invalid - reinitialize and retry
                error_type = type(e).__name__
                logger.warning(
                    "channel_closed_during_publish",
                    symbol=feature_vector.symbol,
                    error=str(e),
                    error_type=error_type,
                    queue=self._queue_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                
                if attempt < max_retries - 1:
                    # Reset channel and queue references before reinitialization
                    self._channel = None
                    self._queue = None
                    
                    # Add small delay to allow connection to stabilize
                    await asyncio.sleep(0.1 * (attempt + 1))
                    
                    # Reinitialize channel and queue before retry
                    try:
                        await self.initialize()
                    except Exception as init_error:
                        logger.error(
                            "failed_to_reinitialize_channel",
                            symbol=feature_vector.symbol,
                            error=str(init_error),
                            exc_info=True,
                        )
                        # If reinitialization fails, log error and give up
                        break
                else:
                    # Last attempt failed - log error
                    logger.error(
                        "feature_publish_error_after_retries",
                        symbol=feature_vector.symbol,
                        error=str(e),
                        error_type=error_type,
                        exc_info=True,
                    )
            
            except Exception as e:
                # Other error - log and don't retry
                error_type = type(e).__name__
                logger.error(
                    "feature_publish_error",
                    symbol=feature_vector.symbol,
                    error=str(e),
                    error_type=error_type,
                    exc_info=True,
                )
                return  # Don't retry for other errors

