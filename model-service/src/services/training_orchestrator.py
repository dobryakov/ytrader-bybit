"""
Training orchestration service.

Coordinates dataset building, model training, quality evaluation, version management,
and handles training cancellation and restart on new triggers.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
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
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrates the model training pipeline."""

    def __init__(self):
        """Initialize training orchestrator."""
        self._current_training_task: Optional[asyncio.Task] = None
        self._training_cancelled = False
        self._execution_events_buffer: List[OrderExecutionEvent] = []
        self._signal_market_data: Dict[str, MarketDataSnapshot] = {}

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

    async def check_and_trigger_training(self, strategy_id: Optional[str] = None) -> None:
        """
        Check if training should be triggered and start it if needed.

        Args:
            strategy_id: Trading strategy identifier
        """
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
        if not self._execution_events_buffer:
            logger.warning("No execution events available for training", strategy_id=strategy_id)
            return

        logger.info("Starting model training", strategy_id=strategy_id, event_count=len(self._execution_events_buffer))

        # Reset cancellation flag
        self._training_cancelled = False

        # Create training task
        self._current_training_task = asyncio.create_task(
            self._train_model_async(strategy_id, self._execution_events_buffer.copy(), self._signal_market_data.copy())
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
                training_config={"model_type": "xgboost", "task_type": "classification"},
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
            logger.info(
                "Model training completed",
                training_id=training_id,
                version=version,
                strategy_id=strategy_id,
                duration_seconds=training_duration,
                metrics=metrics,
            )

        except asyncio.CancelledError:
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


# Global training orchestrator instance
training_orchestrator = TrainingOrchestrator()

