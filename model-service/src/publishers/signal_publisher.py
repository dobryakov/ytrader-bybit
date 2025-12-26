"""
Trading signal publisher.

Publishes trading signals to RabbitMQ queue for order manager consumption
and persists them to PostgreSQL database for Grafana monitoring dashboard visibility.
"""

import json
from datetime import datetime
from typing import Optional

import aio_pika

from ..models.signal import TradingSignal
from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger, bind_context
from ..config.exceptions import MessageQueueError
from ..database.repositories.trading_signal_repo import TradingSignalRepository
from ..config.settings import settings
from ..services.signal_processing_metrics import signal_processing_metrics
from common.trading_events import trading_events_publisher

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

        # Calculate processing delay between signal creation and publication attempt
        try:
            signal_creation_time = signal.timestamp
            publication_time = datetime.utcnow()
            processing_delay_seconds = (publication_time - signal_creation_time).total_seconds()

            # Record metrics for monitoring
            signal_processing_metrics.record_delay(processing_delay_seconds)

            # Log structured delay information
            logger.info(
                "Signal processing delay measured",
                signal_id=signal.signal_id,
                strategy_id=signal.strategy_id,
                signal_creation_time=signal_creation_time.isoformat() + "Z",
                signal_publication_time=publication_time.isoformat() + "Z",
                processing_delay_seconds=processing_delay_seconds,
            )

            # Emit warning if delay exceeds configured alert threshold
            if processing_delay_seconds > settings.signal_processing_delay_alert_threshold_seconds:
                logger.warning(
                    "Signal processing delay exceeded alert threshold",
                    signal_id=signal.signal_id,
                    strategy_id=signal.strategy_id,
                    processing_delay_seconds=processing_delay_seconds,
                    alert_threshold_seconds=settings.signal_processing_delay_alert_threshold_seconds,
                )
        except Exception as e:
            # Delay measurement issues should not block publishing
            logger.warning(
                "Failed to measure signal processing delay (continuing)",
                signal_id=getattr(signal, "signal_id", None),
                error=str(e),
            )

        # Skip RabbitMQ publishing for rejected signals
        if signal.is_rejected:
            logger.info(
                "Skipping RabbitMQ publish for rejected signal",
                signal_id=signal.signal_id,
                rejection_reason=signal.rejection_reason,
                confidence=signal.confidence,
                threshold=signal.effective_threshold,
            )
        else:
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
            except Exception as e:
                logger.error(
                    "Failed to publish trading signal after retries",
                    signal_id=signal.signal_id,
                    error=str(e),
                    exc_info=True,
                )
                raise MessageQueueError(f"Failed to publish signal {signal.signal_id}: {e}") from e

        # Persist trading signal to database (for both valid and rejected signals)
        try:
            logger.info("Attempting to persist trading signal to database", signal_id=signal.signal_id)
            await self._persist_signal(signal)
            logger.info("Successfully persisted trading signal to database", signal_id=signal.signal_id)
            
            # Save prediction target AFTER signal is persisted (avoids foreign key race condition)
            # This ensures trading_signals record exists before prediction_targets references it
            if hasattr(signal, '_prediction_data'):
                try:
                    from ..services.intelligent_signal_generator import intelligent_signal_generator
                    prediction_target = await intelligent_signal_generator._save_prediction_target(
                        signal=signal,
                        prediction_result=signal._prediction_data['prediction_result'],
                        feature_vector=signal._prediction_data['feature_vector'],
                        model_version=signal._prediction_data['model_version'],
                        trace_id=signal._prediction_data.get('trace_id'),
                    )
                    
                    # Trigger immediate check if target timestamp has already passed
                    if prediction_target and prediction_target.get("target_timestamp"):
                        from datetime import datetime as _dt_utc, timezone
                        from ..tasks.target_evaluation_task import target_evaluation_task

                        target_ts = prediction_target["target_timestamp"]
                        
                        # Handle case where target_ts is a string (from _record_to_dict conversion)
                        if isinstance(target_ts, str):
                            try:
                                # Parse ISO format string, handling both with and without timezone
                                if target_ts.endswith("Z"):
                                    target_ts = datetime.fromisoformat(target_ts.replace("Z", "+00:00"))
                                else:
                                    target_ts = datetime.fromisoformat(target_ts)
                            except (ValueError, AttributeError) as e:
                                logger.warning(
                                    "Failed to parse target_timestamp string",
                                    signal_id=signal.signal_id,
                                    target_timestamp=target_ts,
                                    error=str(e),
                                )
                                # Skip further processing if parsing failed
                                target_ts = None
                        
                        # Normalize to UTC timezone-aware, then to naive for comparison
                        if target_ts and isinstance(target_ts, datetime):
                            if target_ts.tzinfo is None:
                                target_ts = target_ts.replace(tzinfo=timezone.utc)
                            else:
                                target_ts = target_ts.astimezone(timezone.utc)
                            target_ts = target_ts.replace(tzinfo=None)  # Convert to naive for comparison

                        if target_ts and target_ts <= _dt_utc.utcnow():
                            await target_evaluation_task.trigger_immediate_check(
                                prediction_target_id=str(prediction_target["id"])
                            )
                except Exception as e:
                    # Log error but don't fail signal publishing
                    logger.warning(
                        "Failed to save prediction target after signal persistence (continuing)",
                        signal_id=signal.signal_id,
                        error=str(e),
                        exc_info=True,
                    )
        except Exception as e:
            # Log error but don't raise - continue processing on persistence failures
            # This ensures signals are still published to RabbitMQ even if database persistence fails
            logger.warning(
                "Failed to persist trading signal to database (continuing)",
                signal_id=signal.signal_id,
                error=str(e),
                exc_info=True,
            )

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

            # Extract prediction horizon and target timestamp from metadata
            prediction_horizon_seconds = None
            target_timestamp = None
            if signal.metadata:
                prediction_horizon_seconds = signal.metadata.get("prediction_horizon_seconds")
                target_timestamp_str = signal.metadata.get("target_timestamp")
                if target_timestamp_str:
                    from datetime import datetime
                    try:
                        # Parse ISO format timestamp
                        target_timestamp = datetime.fromisoformat(target_timestamp_str.replace("Z", "+00:00"))
                    except Exception:
                        pass

            # Сформируем payload для торгового события (те же поля, что пишутся в БД)
            signal_payload = {
                "signal_id": signal.signal_id,
                "strategy_id": signal.strategy_id,
                "asset": signal.asset,
                "side": signal.signal_type,
                "price": price,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp,
                "model_version": signal.model_version,
                "is_warmup": signal.is_warmup,
                "market_data_snapshot": market_data_dict,
                "metadata": signal.metadata,
                "trace_id": signal.trace_id,
                "prediction_horizon_seconds": prediction_horizon_seconds,
                "target_timestamp": target_timestamp,
            }

            # Отправляем событие в trading_events exchange (не блокируя сохранение в БД)
            try:
                await trading_events_publisher.publish_trading_signal_event(
                    event_type="trading_signal_created",
                    service="model-service",
                    signal_payload=signal_payload,
                    trace_id=signal.trace_id,
                    level="info",
                    ts=signal.timestamp,
                )
            except Exception as pub_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to publish trading signal event (continuing)",
                    signal_id=signal.signal_id,
                    error=str(pub_exc),
                )

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
                prediction_horizon_seconds=prediction_horizon_seconds,
                target_timestamp=target_timestamp,
                is_rejected=signal.is_rejected,
                rejection_reason=signal.rejection_reason,
                effective_threshold=signal.effective_threshold,
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

