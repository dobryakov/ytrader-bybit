"""
Training orchestration service.

Coordinates dataset building, model training, quality evaluation, version management,
and handles training cancellation and restart on new triggers.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
from uuid import uuid4
import pandas as pd

from ..models.execution_event import OrderExecutionEvent
from ..models.signal import MarketDataSnapshot
from ..services.dataset_builder import dataset_builder
from ..services.model_trainer import model_trainer
from ..services.quality_evaluator import quality_evaluator
from ..services.model_version_manager import model_version_manager
from ..services.retraining_trigger import retraining_trigger
from ..services.buffer_persistence import buffer_persistence
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrates the model training pipeline."""

    def __init__(self):
        """Initialize training orchestrator."""
        self._current_training_task: Optional[asyncio.Task] = None
        self._current_strategy_id: Optional[str] = None
        self._training_cancelled = False
        self._execution_events_buffer: List[OrderExecutionEvent] = []
        self._signal_market_data: Dict[str, MarketDataSnapshot] = {}

        # Training queue: list of items waiting to be trained
        # Each item: {"strategy_id": str | None, "events": List[OrderExecutionEvent],
        #             "signal_market_data": Dict[str, MarketDataSnapshot],
        #             "priority": str, "enqueued_at": datetime}
        self._training_queue: List[Dict[str, Any]] = []

        # Metrics for observability
        self._metrics: Dict[str, Any] = {
            "buffer_recovery_count": 0,
            "cancelled_trainings_count": 0,
            "successful_trainings_count": 0,
            "total_trainings_count": 0,
            "last_training_duration_seconds": None,
        }
        self._last_buffer_persist_time: Optional[datetime] = None
        self._last_queue_metrics_update: Optional[datetime] = None

    async def add_execution_event(self, event: OrderExecutionEvent, signal_market_data: Optional[MarketDataSnapshot] = None) -> None:
        """
        Add an execution event to the training buffer.

        Args:
            event: Order execution event
            signal_market_data: Optional market data snapshot from the trading signal
        """
        self._execution_events_buffer.append(event)
        if signal_market_data:
            self._signal_market_data[event.signal_id] = signal_market_data

        logger.debug("Added execution event to buffer", event_id=event.event_id, buffer_size=len(self._execution_events_buffer))

    async def restore_buffer_from_database(self, strategy_id: Optional[str] = None) -> None:
        """
        Restore execution events buffer from database on startup.

        This uses the buffer_persistence service to load execution events that
        have not yet been used for training, so that the service can resume
        from where it left off before restart.
        """
        if not settings.buffer_persistence_enabled or not settings.buffer_recovery_on_startup:
            return

        max_events = settings.buffer_max_recovery_events
        restored_events = await buffer_persistence.restore_buffer(
            strategy_id=strategy_id,
            max_events=max_events,
        )

        if not restored_events:
            return

        # Append restored events to in-memory buffer
        self._execution_events_buffer.extend(restored_events)
        self._metrics["buffer_recovery_count"] = self._metrics.get("buffer_recovery_count", 0) + 1

        logger.info(
            "Restored training buffer from database",
            restored_events=len(restored_events),
            total_buffer_size=len(self._execution_events_buffer),
            strategy_id=strategy_id,
        )
        
        # After restoring buffer, check if training should be triggered immediately
        # (e.g., if we restored 100+ events)
        if restored_events:
            # Check if we have enough events to trigger training
            await self.check_and_trigger_training(strategy_id)

    def _normalize_strategy_id(self, strategy_id: Optional[str]) -> Optional[str]:
        """
        Normalize strategy_id to match configured strategies.

        This ensures that strategy_id from events matches the strategy_id
        used for signal generation, preventing mismatches like 'test-strategy' vs 'test_strategy'.

        Args:
            strategy_id: Strategy identifier from event or None

        Returns:
            Normalized strategy_id that matches configured strategies, or original if not found
        """
        if not strategy_id:
            return strategy_id

        configured_strategies = settings.trading_strategy_list
        if not configured_strategies:
            # No strategies configured, return as-is
            return strategy_id

        # Check exact match first
        if strategy_id in configured_strategies:
            return strategy_id

        # Check case-insensitive match
        strategy_id_lower = strategy_id.lower()
        for configured_strategy in configured_strategies:
            if configured_strategy.lower() == strategy_id_lower:
                logger.warning(
                    "Strategy ID case mismatch, using configured value",
                    event_strategy_id=strategy_id,
                    configured_strategy_id=configured_strategy,
                )
                return configured_strategy

        # Check if strategy_id differs only by separator (hyphen vs underscore)
        # Normalize separators and compare
        normalized_event = strategy_id.replace("-", "_").replace(" ", "_")
        for configured_strategy in configured_strategies:
            normalized_configured = configured_strategy.replace("-", "_").replace(" ", "_")
            if normalized_event.lower() == normalized_configured.lower():
                logger.warning(
                    "Strategy ID separator mismatch, using configured value",
                    event_strategy_id=strategy_id,
                    configured_strategy_id=configured_strategy,
                )
                return configured_strategy

        # Strategy not found in configured list - log warning but return original
        logger.warning(
            "Strategy ID not found in configured strategies, using event value",
            event_strategy_id=strategy_id,
            configured_strategies=configured_strategies,
        )
        return strategy_id

    async def check_and_trigger_training(self, strategy_id: Optional[str] = None) -> None:
        """
        Check if training should be triggered and start it if needed.

        Args:
            strategy_id: Trading strategy identifier
        """
        # Normalize strategy_id to match configured strategies
        strategy_id = self._normalize_strategy_id(strategy_id)

        # Check if retraining should occur
        should_retrain = await retraining_trigger.should_retrain(
            strategy_id=strategy_id,
            execution_event_count=len(self._execution_events_buffer),
            current_dataset_size=len(self._execution_events_buffer),
        )

        if not should_retrain:
            return

        # Check minimum dataset size before starting training
        # This prevents training from starting with insufficient data (e.g., scheduled retraining with only 1 event)
        min_dataset_size = settings.model_training_min_dataset_size
        buffer_size = len(self._execution_events_buffer)

        if buffer_size < min_dataset_size:
            logger.info(
                "Training triggered but insufficient data in buffer",
                strategy_id=strategy_id,
                buffer_size=buffer_size,
                min_dataset_size=min_dataset_size,
                reason="waiting_for_more_events",
            )
            return  # Wait for more events to accumulate

        # Check if training is already in progress
        if self._current_training_task and not self._current_training_task.done():
            # If queueing is enabled, enqueue a new training request instead of cancelling
            if settings.training_queue_enabled:
                if len(self._training_queue) >= settings.training_queue_max_size:
                    logger.warning(
                        "Training queue is full, dropping new training request",
                        strategy_id=strategy_id,
                        queue_size=len(self._training_queue),
                    )
                    return

                queue_item = {
                    "strategy_id": strategy_id,
                    "events": self._execution_events_buffer.copy(),
                    "signal_market_data": self._signal_market_data.copy(),
                    "priority": "NORMAL",
                    "enqueued_at": datetime.utcnow(),
                }
                self._training_queue.append(queue_item)
                self._execution_events_buffer.clear()
                self._signal_market_data.clear()

                logger.info(
                    "Enqueued training request while training in progress",
                    strategy_id=strategy_id,
                    queue_size=len(self._training_queue),
                    buffer_size=buffer_size,
                )
                return

            # Legacy behaviour: cancel current training and start a new one
            logger.info("Training already in progress, cancelling for new training", strategy_id=strategy_id)
            await self._cancel_current_training()

        # Start new training
        await self._start_training(strategy_id)

    async def _start_training(self, strategy_id: Optional[str] = None) -> None:
        """
        Start model training.

        Args:
            strategy_id: Trading strategy identifier
        """
        # Normalize strategy_id to match configured strategies
        strategy_id = self._normalize_strategy_id(strategy_id)

        if not self._execution_events_buffer:
            logger.warning("No execution events available for training", strategy_id=strategy_id)
            return

        logger.info("Starting model training", strategy_id=strategy_id, event_count=len(self._execution_events_buffer))

        # Reset cancellation flag
        self._training_cancelled = False
        self._current_strategy_id = strategy_id

        # Create training task
        self._current_training_task = asyncio.create_task(
            self._train_model_async(
                strategy_id,
                self._execution_events_buffer.copy(),
                self._signal_market_data.copy(),
            )
        )

        # Attach completion handler to process queued trainings
        self._current_training_task.add_done_callback(
            lambda _: asyncio.create_task(self._handle_training_completion())
        )

        # Clear buffer after starting training
        self._execution_events_buffer.clear()
        self._signal_market_data.clear()

    async def _train_model_async(
        self,
        strategy_id: Optional[str],
        execution_events: List[OrderExecutionEvent],
        signal_market_data: Dict[str, MarketDataSnapshot],
    ) -> None:
        """
        Train model asynchronously.

        Args:
            strategy_id: Trading strategy identifier
            execution_events: List of execution events to train on
            signal_market_data: Dictionary mapping signal_id to market data snapshot
        """
        training_start_time = datetime.utcnow()
        training_id = str(uuid4())

        try:
            logger.info(
                "Training model",
            training_id=training_id,
                strategy_id=strategy_id,
                event_count=len(execution_events),
            )

            # Build training dataset
            dataset = dataset_builder.build_dataset(
                execution_events=execution_events,
                signal_market_data=signal_market_data if signal_market_data else None,
                strategy_id=strategy_id,
                label_type="binary",
            )

            if not dataset:
                logger.error("Failed to build training dataset", training_id=training_id)
                return

            if self._training_cancelled:
                logger.info("Training cancelled during dataset building", training_id=training_id)
                return

            # Train model
            model = model_trainer.train_model(
                dataset=dataset,
                model_type="xgboost",  # Default to XGBoost, could be configurable
                task_type="classification",
            )

            if self._training_cancelled:
                logger.info("Training cancelled during model training", training_id=training_id)
                return

            # Evaluate model quality
            # For now, we'll use the training data for evaluation (in production, use validation set)
            y_pred = model.predict(dataset.features)
            y_pred_proba = model.predict_proba(dataset.features)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = quality_evaluator.evaluate(
                y_true=dataset.labels,
                y_pred=pd.Series(y_pred),
                y_pred_proba=pd.Series(y_pred_proba) if y_pred_proba is not None else None,
                task_type="classification",
            )

            # Calculate trading performance metrics
            trading_metrics = quality_evaluator.calculate_trading_metrics(execution_events, pd.Series(y_pred))
            metrics.update(trading_metrics)

            if self._training_cancelled:
                logger.info("Training cancelled during quality evaluation", training_id=training_id)
                return

            # Create model version
            training_duration = (datetime.utcnow() - training_start_time).total_seconds()
            version = f"v{int(datetime.utcnow().timestamp())}"  # Simple versioning based on timestamp

            # Determine file path
            file_path = f"v{version}/model.json" if model.__class__.__name__.startswith("XGB") else f"v{version}/model.pkl"

            model_version = await model_version_manager.create_version(
                version=version,
                model_type="xgboost",
                file_path=file_path,
                strategy_id=strategy_id,
                training_duration_seconds=int(training_duration),
                training_dataset_size=dataset.get_record_count(),
                training_config={
                    "model_type": "xgboost",
                    "task_type": "classification",
                    "feature_count": len(dataset.get_feature_names()),
                },
                is_active=False,  # Don't activate automatically, require manual activation or quality check
            )

            # Save model to file
            full_file_path = f"{settings.model_storage_path}/{file_path}"
            model_trainer.save_model(model, "xgboost", full_file_path)

            # Save quality metrics
            await model_version_manager.save_quality_metrics(
                model_version_id=model_version["id"],
                metrics=metrics,
                evaluation_dataset_size=dataset.get_record_count(),
            )

            # Mark events as used for training if buffer persistence is enabled
            if settings.buffer_persistence_enabled:
                try:
                    await buffer_persistence.mark_events_used_for_training(
                        [event.event_id for event in execution_events],
                        model_version["id"],
                    )
                except Exception:
                    # Errors are already logged in buffer_persistence; do not fail training
                    pass

            # Check if model quality meets threshold for activation
            accuracy = metrics.get("accuracy", 0.0)
            if accuracy >= settings.model_quality_threshold_accuracy:
                await model_version_manager.activate_version(model_version["id"], strategy_id)
                logger.info("Model activated automatically", version=version, accuracy=accuracy)
            else:
                logger.info("Model quality below threshold, not activated", version=version, accuracy=accuracy)

            # Record retraining
            retraining_trigger.record_retraining(strategy_id)

            training_duration = (datetime.utcnow() - training_start_time).total_seconds()
            self._metrics["last_training_duration_seconds"] = training_duration
            self._metrics["successful_trainings_count"] = self._metrics.get("successful_trainings_count", 0) + 1
            self._metrics["total_trainings_count"] = self._metrics.get("total_trainings_count", 0) + 1

            logger.info(
                "Model training completed",
                training_id=training_id,
                version=version,
                strategy_id=strategy_id,
                duration_seconds=training_duration,
                metrics=metrics,
            )

        except asyncio.CancelledError:
            self._metrics["cancelled_trainings_count"] = self._metrics.get("cancelled_trainings_count", 0) + 1
            self._metrics["total_trainings_count"] = self._metrics.get("total_trainings_count", 0) + 1
            logger.info("Training task cancelled", training_id=training_id)
            raise
        except Exception as e:
            logger.error("Model training failed", training_id=training_id, error=str(e), exc_info=True)
            raise

    async def _cancel_current_training(self) -> None:
        """Cancel current training task."""
        if self._current_training_task and not self._current_training_task.done():
            self._training_cancelled = True
            self._current_training_task.cancel()
            try:
                await self._current_training_task
            except asyncio.CancelledError:
                pass
            logger.info("Training cancelled")

    async def _handle_training_completion(self) -> None:
        """
        Handle completion of a training task.

        If the training queue has pending items, automatically start the next one.
        """
        # Clear current training reference
        self._current_training_task = None
        self._current_strategy_id = None

        if not settings.training_queue_enabled or not self._training_queue:
            return

        # Pop next item (FIFO queue)
        queue_item = self._training_queue.pop(0)
        self._execution_events_buffer = queue_item["events"]
        self._signal_market_data = queue_item["signal_market_data"]

        logger.info(
            "Starting queued training after previous completion",
            strategy_id=queue_item.get("strategy_id"),
            remaining_queue_size=len(self._training_queue),
        )

        await self._start_training(queue_item.get("strategy_id"))

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Gracefully shut down training orchestrator.

        - Wait for current training to complete (up to timeout)
        - Leave unused buffer events in database for future recovery
        - Log buffer and queue state for observability
        """
        logger.info(
            "Shutting down training orchestrator",
            buffer_size=len(self._execution_events_buffer),
            queue_size=len(self._training_queue),
        )

        if self._current_training_task and not self._current_training_task.done():
            try:
                await asyncio.wait_for(self._current_training_task, timeout=timeout)
                logger.info("Current training completed during shutdown")
            except asyncio.TimeoutError:
                logger.warning("Training did not complete before shutdown timeout, cancelling task")
                await self._cancel_current_training()
            except Exception as e:
                logger.error("Error while waiting for training to complete during shutdown", error=str(e), exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """
        Get current training orchestrator status for API responses.

        Returns:
            Dictionary with training status, buffer, and queue metrics.
        """
        is_training = self._current_training_task is not None and not self._current_training_task.done()
        now = datetime.utcnow()

        queue_size = len(self._training_queue)
        next_item = self._training_queue[0] if self._training_queue else None
        next_wait_seconds: Optional[float] = None
        next_strategy_id: Optional[str] = None

        if next_item:
            enqueued_at: datetime = next_item.get("enqueued_at", now)
            next_wait_seconds = max((now - enqueued_at).total_seconds(), 0.0)
            next_strategy_id = next_item.get("strategy_id")

        return {
            "is_training": is_training,
            "buffered_events_count": len(self._execution_events_buffer),
            "queue_size": queue_size,
            "queue_next_wait_time_seconds": next_wait_seconds,
            "next_queued_training_strategy_id": next_strategy_id,
            "metrics": self._metrics.copy(),
        }


# Global training orchestrator instance
training_orchestrator = TrainingOrchestrator()

