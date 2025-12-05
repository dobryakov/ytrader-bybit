"""
Dataset completion publisher for RabbitMQ.
"""
from typing import Optional
import json
import structlog
from aio_pika import Connection, Channel, Message

from src.mq.connection import MQConnection

logger = structlog.get_logger(__name__)


class DatasetPublisher:
    """Publishes dataset completion notifications to RabbitMQ."""
    
    def __init__(self, mq_connection: MQConnection):
        """
        Initialize dataset publisher.
        
        Args:
            mq_connection: RabbitMQ connection manager
        """
        self._mq_connection = mq_connection
        self._channel: Optional[Channel] = None
    
    async def initialize(self) -> None:
        """Initialize publisher channel."""
        connection = await self._mq_connection.get_connection()
        self._channel = await connection.channel()
        logger.info("Dataset publisher initialized")
    
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
        
        message = Message(
            body=json.dumps(message_data).encode(),
            content_type="application/json",
        )
        
        await self._channel.default_exchange.publish(
            message,
            routing_key="features.dataset.ready",
        )
        
        logger.info(
            "dataset_ready_published",
            dataset_id=dataset_id,
            symbol=symbol,
            status=status,
            queue="features.dataset.ready",
        )
