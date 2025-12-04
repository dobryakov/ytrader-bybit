"""
Feature publisher for publishing computed features to RabbitMQ queue.
"""
import json
from typing import Optional
from datetime import datetime
import structlog
import aio_pika

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
        if self._queue is None:
            await self.initialize()
        
        try:
            # Serialize feature vector
            # Use mode='json' to automatically convert datetime to ISO string
            # This handles all datetime fields, not just timestamp
            try:
                data = feature_vector.model_dump(mode='json')
            except TypeError:
                # Fallback for Pydantic v1 or if mode='json' is not supported
                data = feature_vector.model_dump()
                # Manually convert datetime to ISO string
                if isinstance(data.get("timestamp"), datetime):
                    data["timestamp"] = data["timestamp"].isoformat()
            message_body = json.dumps(data).encode()
            
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
        
        except Exception as e:
            logger.error(
                "feature_publish_error",
                symbol=feature_vector.symbol,
                error=str(e),
                exc_info=True,
            )

