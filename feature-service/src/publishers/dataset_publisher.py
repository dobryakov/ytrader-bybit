"""
Dataset completion publisher for RabbitMQ.
"""
from typing import Optional
import json
import asyncio
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
        strategy_id: Optional[str] = None,
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
            strategy_id: Optional strategy ID (for model training)
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
        # Add strategy_id if provided
        if strategy_id is not None:
            message_data["strategy_id"] = strategy_id
        
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
            
            except (
                aiormq.exceptions.ChannelInvalidStateError,
                aiormq.exceptions.AMQPConnectionError,
                ConnectionResetError,
                AttributeError,
                RuntimeError,
            ) as e:
                # Channel is closed or invalid - reinitialize and retry
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Check if it's a connection closed error
                is_connection_error = (
                    isinstance(e, (aiormq.exceptions.AMQPConnectionError, ConnectionResetError)) or
                    "closed" in error_msg.lower() or
                    "Connection" in error_type or
                    "Connection was not opened" in error_msg or
                    "Connection reset" in error_msg
                )
                
                logger.warning(
                    "channel_closed_during_publish",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    error=error_msg,
                    error_type=error_type,
                    queue="features.dataset.ready",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    is_connection_error=is_connection_error,
                )
                
                if attempt < max_retries - 1:
                    # For connection errors, reset the connection in mq_manager
                    # Use reset_connection() method to properly close channels before connection
                    if is_connection_error and self._mq_manager is not None:
                        try:
                            await self._mq_manager.reset_connection()
                        except Exception as reset_error:
                            logger.debug(
                                "Error resetting connection during publish retry",
                                error=str(reset_error),
                                error_type=type(reset_error).__name__,
                            )
                    
                    # Reset channel reference before reinitialization
                    self._channel = None
                    
                    # Add delay to allow connection to stabilize (exponential backoff)
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    
                    # Reinitialize channel before retry
                    try:
                        # Check if mq_manager is valid
                        if self._mq_manager is None:
                            raise RuntimeError("MQ manager is None")
                        await self.initialize()
                    except (asyncio.CancelledError, KeyboardInterrupt) as init_error:
                        # Don't retry on cancellation - re-raise immediately
                        logger.warning(
                            "channel_reinitialization_cancelled",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            error=str(init_error),
                            error_type=type(init_error).__name__,
                        )
                        raise
                    except Exception as init_error:
                        error_msg = str(init_error)
                        error_type = type(init_error).__name__
                        logger.warning(
                            "failed_to_reinitialize_channel",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            error=error_msg,
                            error_type=error_type,
                            attempt=attempt + 1,
                            mq_manager_is_none=self._mq_manager is None,
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
            
            except (asyncio.CancelledError, KeyboardInterrupt) as e:
                # Don't retry on cancellation - re-raise immediately
                logger.warning(
                    "dataset_publish_cancelled",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            
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
