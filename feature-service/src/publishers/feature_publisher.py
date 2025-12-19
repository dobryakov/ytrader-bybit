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
        """Publish feature vector to queue.
        
        Handles connection errors gracefully - logs errors but doesn't raise exceptions
        to avoid breaking feature computation loop.
        """
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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if channel/queue is None or invalid, reinitialize if needed
                if self._queue is None or self._channel is None:
                    await self.initialize()
                
                # Publish message
                await self._channel.default_exchange.publish(
                    aio_pika.Message(
                        body=message_body,
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=self._queue_name,
                )
                
                logger.info(
                    "feature_vector_published",
                    symbol=feature_vector.symbol,
                    timestamp=feature_vector.timestamp.isoformat(),
                    features_count=len(feature_vector.features),
                    feature_names=sorted(list(feature_vector.features.keys())),
                    trace_id=feature_vector.trace_id,
                    queue=self._queue_name,
                )
                return  # Success - exit function
            
            except (aiormq.exceptions.ChannelInvalidStateError, AttributeError, RuntimeError) as e:
                # Channel is closed or invalid - reinitialize and retry
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a connection closed error
                is_connection_error = (
                    "closed" in error_msg.lower() or
                    "Connection" in error_type or
                    "Connection was not opened" in error_msg
                )
                
                logger.warning(
                    "channel_closed_during_publish",
                    symbol=feature_vector.symbol,
                    error=error_msg,
                    error_type=error_type,
                    queue=self._queue_name,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    is_connection_error=is_connection_error,
                )
                
                if attempt < max_retries - 1:
                    # Reset channel and queue references before reinitialization
                    self._channel = None
                    self._queue = None
                    
                    # Add delay to allow connection to stabilize (exponential backoff)
                    await asyncio.sleep(0.5 * (attempt + 1))
                    
                    # Reinitialize channel and queue before retry
                    try:
                        # Check if mq_manager is valid
                        if self._mq_manager is None:
                            raise RuntimeError("MQ manager is None")
                        await self.initialize()
                    except (asyncio.CancelledError, KeyboardInterrupt) as init_error:
                        # Don't retry on cancellation - re-raise immediately
                        logger.warning(
                            "channel_reinitialization_cancelled",
                            symbol=feature_vector.symbol,
                            error=str(init_error),
                            error_type=type(init_error).__name__,
                        )
                        raise
                    except Exception as init_error:
                        error_msg = str(init_error)
                        error_type = type(init_error).__name__
                        logger.warning(
                            "failed_to_reinitialize_channel",
                            symbol=feature_vector.symbol,
                            error=error_msg,
                            error_type=error_type,
                            attempt=attempt + 1,
                            mq_manager_is_none=self._mq_manager is None,
                        )
                        # Continue to next retry attempt
                        continue
                else:
                    # Last attempt failed - log error but don't raise
                    logger.error(
                        "feature_publish_failed_after_retries",
                        symbol=feature_vector.symbol,
                        error=error_msg,
                        error_type=error_type,
                        queue=self._queue_name,
                    )
                    # Don't raise - just return to avoid breaking feature computation
                    return
            
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
                # Don't raise - just return to avoid breaking feature computation
                return

