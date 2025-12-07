"""
Training orchestration service.

Coordinates dataset building via Feature Service, model training, quality evaluation, version management,
and handles training cancellation and restart on new triggers.

Training pipeline uses only market data from Feature Service (not execution_events).
Model learns from market movements (price predictions), not from own trading results.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import asyncio
from uuid import uuid4
import pandas as pd
from pathlib import Path

from ..models.training_dataset import TrainingDataset
from ..models.dataset import DatasetBuildRequest, TargetConfig, SplitStrategy, DatasetStatus
from ..services.model_trainer import model_trainer
from ..services.quality_evaluator import quality_evaluator
from ..services.model_version_manager import model_version_manager
from ..services.retraining_trigger import retraining_trigger
from ..services.feature_service_client import feature_service_client
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrates the model training pipeline using Feature Service datasets."""

    def __init__(self):
        """Initialize training orchestrator."""
        self._current_training_task: Optional[asyncio.Task] = None
        self._current_strategy_id: Optional[str] = None
        self._training_cancelled = False

        # Training queue: list of items waiting to be trained
        # Each item: {"strategy_id": str | None, "dataset_id": UUID, "symbol": str,
        #             "priority": str, "enqueued_at": datetime}
        self._training_queue: List[Dict[str, Any]] = []

        # Pending dataset builds: tracks dataset_id -> strategy_id mapping
        # Used to match dataset.ready notifications with training requests
        self._pending_dataset_builds: Dict[UUID, Dict[str, Any]] = {}

        # Metrics for observability
        self._metrics: Dict[str, Any] = {
            "cancelled_trainings_count": 0,
            "successful_trainings_count": 0,
            "total_trainings_count": 0,
            "last_training_duration_seconds": None,
            "dataset_build_requests": 0,
            "dataset_build_failures": 0,
        }
        self._last_queue_metrics_update: Optional[datetime] = None

    async def request_dataset_build(self, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> Optional[UUID]:
        """
        Request dataset build from Feature Service.

        Args:
            strategy_id: Trading strategy identifier
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, uses first symbol from configured strategies.

        Returns:
            Dataset ID (UUID) if request successful, None otherwise
        """
        # Normalize strategy_id to match configured strategies
        strategy_id = self._normalize_strategy_id(strategy_id)

        # Determine symbol if not provided
        if not symbol:
            # Use first configured trading pair or default to BTCUSDT
            trading_pairs = getattr(settings, "trading_pairs", None) or ["BTCUSDT"]
            symbol = trading_pairs[0] if trading_pairs else "BTCUSDT"
            logger.debug("Using default symbol for dataset build", symbol=symbol, strategy_id=strategy_id)

        # Calculate dataset periods
        periods = self._calculate_dataset_periods()

        # Build dataset request
        request = {
            "symbol": symbol,
            "split_strategy": SplitStrategy.TIME_BASED.value,
            "train_period_start": periods["train_period_start"].isoformat() + "Z",
            "train_period_end": periods["train_period_end"].isoformat() + "Z",
            "validation_period_start": periods["validation_period_start"].isoformat() + "Z",
            "validation_period_end": periods["validation_period_end"].isoformat() + "Z",
            "test_period_start": periods["test_period_start"].isoformat() + "Z",
            "test_period_end": periods["test_period_end"].isoformat() + "Z",
            "target_config": {
                "type": "classification",
                "horizon": "1h",  # Default horizon, could be configurable
                "threshold": 0.001,
            },
            "feature_registry_version": "latest",  # Use latest feature registry version
            "output_format": "parquet",
        }

        # Request dataset build from Feature Service
        trace_id = str(uuid4())
        dataset_id = await feature_service_client.build_dataset(request, trace_id=trace_id)

        if not dataset_id:
            self._metrics["dataset_build_failures"] = self._metrics.get("dataset_build_failures", 0) + 1
            logger.error("Failed to request dataset build from Feature Service", strategy_id=strategy_id, symbol=symbol, trace_id=trace_id)
            return None

        # Track pending dataset build
        self._pending_dataset_builds[dataset_id] = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "requested_at": datetime.utcnow(),
            "trace_id": trace_id,
        }
        self._metrics["dataset_build_requests"] = self._metrics.get("dataset_build_requests", 0) + 1

        logger.info(
            "Dataset build requested from Feature Service",
            dataset_id=str(dataset_id),
            strategy_id=strategy_id,
            symbol=symbol,
            trace_id=trace_id,
        )

        return dataset_id

    async def handle_dataset_ready(self, dataset_id: UUID, symbol: Optional[str] = None, trace_id: Optional[str] = None) -> None:
        """
        Handle dataset ready notification from Feature Service.

        This method is called by dataset_ready_consumer when a dataset.ready notification is received.
        It triggers model training using the ready dataset.

        Args:
            dataset_id: Dataset UUID identifier
            symbol: Trading pair symbol (optional, for logging)
            trace_id: Optional trace ID for request flow tracking
        """
        # Check if this dataset was requested by us
        pending_build = self._pending_dataset_builds.get(dataset_id)
        if not pending_build:
            logger.warning(
                "Dataset ready notification received for unknown dataset",
                dataset_id=str(dataset_id),
                symbol=symbol,
                trace_id=trace_id,
            )
            return

        strategy_id = pending_build["strategy_id"]
        symbol = symbol or pending_build.get("symbol")

        logger.info(
            "Dataset ready notification received, starting training",
            dataset_id=str(dataset_id),
            strategy_id=strategy_id,
            symbol=symbol,
            trace_id=trace_id,
        )

        # Remove from pending builds
        del self._pending_dataset_builds[dataset_id]

        # Check if training is already in progress
        if self._current_training_task and not self._current_training_task.done():
            # If queueing is enabled, enqueue a new training request instead of cancelling
            if settings.training_queue_enabled:
                if len(self._training_queue) >= settings.training_queue_max_size:
                    logger.warning(
                        "Training queue is full, dropping dataset ready notification",
                        dataset_id=str(dataset_id),
                        strategy_id=strategy_id,
                        queue_size=len(self._training_queue),
                    )
                    return

                queue_item = {
                    "strategy_id": strategy_id,
                    "dataset_id": dataset_id,
                    "symbol": symbol,
                    "priority": "NORMAL",
                    "enqueued_at": datetime.utcnow(),
                }
                self._training_queue.append(queue_item)

                logger.info(
                    "Enqueued training request while training in progress",
                    dataset_id=str(dataset_id),
                    strategy_id=strategy_id,
                    queue_size=len(self._training_queue),
                )
                return

            # Legacy behaviour: cancel current training and start a new one
            logger.info("Training already in progress, cancelling for new training", dataset_id=str(dataset_id), strategy_id=strategy_id)
            await self._cancel_current_training()

        # Start training with ready dataset
        await self._start_training_with_dataset(strategy_id, dataset_id, symbol, trace_id)

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

    def _calculate_dataset_periods(self, reference_time: Optional[datetime] = None) -> Dict[str, datetime]:
        """
        Calculate train/validation/test periods for dataset building based on current time and configuration.

        Uses rolling window approach:
        - Test period: most recent data (e.g., last 1 day)
        - Validation period: before test period (e.g., 7 days before test)
        - Train period: before validation period (e.g., 30 days before validation)

        Args:
            reference_time: Reference time for calculation (defaults to UTC now).
                          Typically set to 1 day ago to exclude most recent data.

        Returns:
            Dictionary with keys:
                - train_period_start: datetime
                - train_period_end: datetime
                - validation_period_start: datetime
                - validation_period_end: datetime
                - test_period_start: datetime
                - test_period_end: datetime
        """
        if reference_time is None:
            # Use 1 day ago as reference to exclude most recent data (which may be incomplete)
            reference_time = datetime.utcnow() - timedelta(days=1)

        # Calculate periods backwards from reference time
        test_period_end = reference_time
        test_period_start = test_period_end - timedelta(days=settings.model_retraining_test_period_days)

        validation_period_end = test_period_start
        validation_period_start = validation_period_end - timedelta(days=settings.model_retraining_validation_period_days)

        train_period_end = validation_period_start
        train_period_start = train_period_end - timedelta(days=settings.model_retraining_train_period_days)

        periods = {
            "train_period_start": train_period_start,
            "train_period_end": train_period_end,
            "validation_period_start": validation_period_start,
            "validation_period_end": validation_period_end,
            "test_period_start": test_period_start,
            "test_period_end": test_period_end,
        }

        logger.debug(
            "Calculated dataset periods",
            train_period_days=settings.model_retraining_train_period_days,
            validation_period_days=settings.model_retraining_validation_period_days,
            test_period_days=settings.model_retraining_test_period_days,
            periods=periods,
        )

        return periods

    async def check_and_trigger_training(self, strategy_id: Optional[str] = None, symbol: Optional[str] = None) -> None:
        """
        Check if training should be triggered and request dataset build if needed.

        Args:
            strategy_id: Trading strategy identifier
            symbol: Trading pair symbol (optional, will use default if not provided)
        """
        # Normalize strategy_id to match configured strategies
        strategy_id = self._normalize_strategy_id(strategy_id)

        # Check if retraining should occur (time-based or scheduled, no execution_events dependency)
        should_retrain = await retraining_trigger.should_retrain(strategy_id=strategy_id)

        if not should_retrain:
            return

        # Request dataset build from Feature Service
        # Training will start automatically when dataset.ready notification is received
        dataset_id = await self.request_dataset_build(strategy_id=strategy_id, symbol=symbol)

        if not dataset_id:
            logger.error("Failed to request dataset build, training not triggered", strategy_id=strategy_id, symbol=symbol)
            return

        logger.info(
            "Dataset build requested for training",
            dataset_id=str(dataset_id),
            strategy_id=strategy_id,
            symbol=symbol,
        )

    async def _start_training_with_dataset(
        self, strategy_id: Optional[str], dataset_id: UUID, symbol: Optional[str] = None, trace_id: Optional[str] = None
    ) -> None:
        """
        Start model training with ready dataset from Feature Service.

        Args:
            strategy_id: Trading strategy identifier
            dataset_id: Dataset UUID identifier
            symbol: Trading pair symbol (for logging)
            trace_id: Optional trace ID for request flow tracking
        """
        # Normalize strategy_id to match configured strategies
        strategy_id = self._normalize_strategy_id(strategy_id)

        logger.info(
            "Starting model training with Feature Service dataset",
            strategy_id=strategy_id,
            dataset_id=str(dataset_id),
            symbol=symbol,
            trace_id=trace_id,
        )

        # Reset cancellation flag
        self._training_cancelled = False
        self._current_strategy_id = strategy_id

        # Create training task
        self._current_training_task = asyncio.create_task(
            self._train_model_async(strategy_id, dataset_id, symbol, trace_id)
        )

        # Attach completion handler to process queued trainings
        self._current_training_task.add_done_callback(
            lambda _: asyncio.create_task(self._handle_training_completion())
        )

    async def _train_model_async(
        self,
        strategy_id: Optional[str],
        dataset_id: UUID,
        symbol: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Train model asynchronously using dataset from Feature Service.

        Args:
            strategy_id: Trading strategy identifier
            dataset_id: Dataset UUID identifier
            symbol: Trading pair symbol (for logging)
            trace_id: Optional trace ID for request flow tracking
        """
        training_start_time = datetime.utcnow()
        training_id = str(uuid4())

        try:
            logger.info(
                "Training model with Feature Service dataset",
                training_id=training_id,
                strategy_id=strategy_id,
                dataset_id=str(dataset_id),
                symbol=symbol,
                trace_id=trace_id,
            )

            # Verify dataset is ready
            dataset_meta = await feature_service_client.get_dataset(dataset_id, trace_id)
            if not dataset_meta:
                logger.error("Dataset not found", training_id=training_id, dataset_id=str(dataset_id), trace_id=trace_id)
                return

            if dataset_meta.status != DatasetStatus.READY:
                logger.error(
                    "Dataset is not ready",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    status=dataset_meta.status.value,
                    trace_id=trace_id,
                )
                return

            if self._training_cancelled:
                logger.info("Training cancelled before dataset download", training_id=training_id)
                return

            # Download train split from Feature Service
            train_file_path = await feature_service_client.download_dataset(dataset_id, split="train", trace_id=trace_id)
            if not train_file_path:
                logger.error("Failed to download train dataset", training_id=training_id, dataset_id=str(dataset_id), trace_id=trace_id)
                return

            if self._training_cancelled:
                logger.info("Training cancelled during dataset download", training_id=training_id)
                return

            # Load dataset from Parquet file
            logger.info("Loading dataset from Parquet file", training_id=training_id, file_path=str(train_file_path), trace_id=trace_id)
            df = pd.read_parquet(train_file_path)

            # Extract features and labels
            # Feature Service dataset format: features are all columns except 'target'
            # Target column name may vary (target, label, y, etc.)
            target_column = None
            for col in ["target", "label", "y", "target_value"]:
                if col in df.columns:
                    target_column = col
                    break

            if not target_column:
                logger.error(
                    "Target column not found in dataset",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    available_columns=list(df.columns),
                    trace_id=trace_id,
                )
                return

            # Separate features and labels
            features_df = df.drop(columns=[target_column])
            labels_series = df[target_column]

            if features_df.empty or labels_series.empty:
                logger.error("Empty dataset after loading", training_id=training_id, dataset_id=str(dataset_id), trace_id=trace_id)
                return

            # Create TrainingDataset object
            dataset = TrainingDataset(
                dataset_id=str(dataset_id),
                strategy_id=strategy_id or "default",
                features=features_df,
                labels=labels_series,
                metadata={
                    "source": "feature_service",
                    "dataset_id": str(dataset_id),
                    "symbol": symbol,
                    "record_count": len(features_df),
                    "feature_count": len(features_df.columns),
                    "feature_names": list(features_df.columns),
                },
            )

            logger.info(
                "Dataset loaded successfully",
                training_id=training_id,
                dataset_id=str(dataset_id),
                record_count=dataset.get_record_count(),
                feature_count=len(dataset.get_feature_names()),
                trace_id=trace_id,
            )

            if self._training_cancelled:
                logger.info("Training cancelled during dataset loading", training_id=training_id)
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

            # Download validation split for evaluation
            validation_file_path = await feature_service_client.download_dataset(
                dataset_id, split="validation", trace_id=trace_id
            )
            validation_features = None
            validation_labels = None

            if validation_file_path:
                try:
                    val_df = pd.read_parquet(validation_file_path)
                    if target_column in val_df.columns:
                        validation_features = val_df.drop(columns=[target_column])
                        validation_labels = val_df[target_column]
                        logger.info(
                            "Validation dataset loaded",
                            training_id=training_id,
                            record_count=len(validation_features),
                            trace_id=trace_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to load validation dataset, using train set for evaluation",
                        training_id=training_id,
                        error=str(e),
                        trace_id=trace_id,
                    )

            # Evaluate model quality on validation set
            # Use validation set if available, otherwise use training set
            eval_features = validation_features if validation_features is not None else dataset.features
            eval_labels = validation_labels if validation_labels is not None else dataset.labels

            y_pred = model.predict(eval_features)
            y_pred_proba = model.predict_proba(eval_features)[:, 1] if hasattr(model, "predict_proba") else None

            validation_metrics = quality_evaluator.evaluate(
                y_true=eval_labels,
                y_pred=pd.Series(y_pred),
                y_pred_proba=pd.Series(y_pred_proba) if y_pred_proba is not None else None,
                task_type="classification",
            )

            # Note: Trading metrics are not calculated here since we don't have execution_events
            # Model learns from market movements (price predictions), not from own trading results

            if self._training_cancelled:
                logger.info("Training cancelled during quality evaluation", training_id=training_id)
                return

            # Download test split for final out-of-sample evaluation
            test_file_path = None
            test_features = None
            test_labels = None
            test_metrics = None

            try:
                logger.info(
                    "Downloading test split for final evaluation",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    trace_id=trace_id,
                )
                test_file_path = await feature_service_client.download_dataset(
                    dataset_id, split="test", trace_id=trace_id
                )

                if test_file_path:
                    try:
                        test_df = pd.read_parquet(test_file_path)
                        if target_column in test_df.columns:
                            test_features = test_df.drop(columns=[target_column])
                            test_labels = test_df[target_column]

                            if test_features.empty or test_labels.empty:
                                logger.warning(
                                    "Test split is empty",
                                    training_id=training_id,
                                    dataset_id=str(dataset_id),
                                    reason="empty_split",
                                    trace_id=trace_id,
                                )
                            else:
                                logger.info(
                                    "Test dataset loaded",
                                    training_id=training_id,
                                    record_count=len(test_features),
                                    trace_id=trace_id,
                                )

                                # Evaluate model on test set
                                test_y_pred = model.predict(test_features)
                                test_y_pred_proba = (
                                    model.predict_proba(test_features)[:, 1] if hasattr(model, "predict_proba") else None
                                )

                                test_metrics = quality_evaluator.evaluate(
                                    y_true=test_labels,
                                    y_pred=pd.Series(test_y_pred),
                                    y_pred_proba=pd.Series(test_y_pred_proba) if test_y_pred_proba is not None else None,
                                    task_type="classification",
                                )

                                logger.info(
                                    "Test set evaluation completed",
                                    training_id=training_id,
                                    metrics=test_metrics,
                                    trace_id=trace_id,
                                )
                        else:
                            logger.warning(
                                "Target column not found in test split",
                                training_id=training_id,
                                dataset_id=str(dataset_id),
                                reason="file_not_found",
                                trace_id=trace_id,
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to load test dataset",
                            training_id=training_id,
                            dataset_id=str(dataset_id),
                            error=str(e),
                            reason="download_failed",
                            trace_id=trace_id,
                        )
                else:
                    logger.warning(
                        "Test split download failed or unavailable",
                        training_id=training_id,
                        dataset_id=str(dataset_id),
                        reason="download_failed",
                        trace_id=trace_id,
                    )
            except Exception as e:
                logger.warning(
                    "Error downloading test split, will use validation metrics",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    error=str(e),
                    reason="download_failed",
                    trace_id=trace_id,
                )

            if self._training_cancelled:
                logger.info("Training cancelled during test split evaluation", training_id=training_id)
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
                    "dataset_source": "feature_service",
                    "dataset_id": str(dataset_id),
                },
                is_active=False,  # Don't activate automatically, require manual activation or quality check
            )

            # Save model to file
            full_file_path = f"{settings.model_storage_path}/{file_path}"
            model_trainer.save_model(model, "xgboost", full_file_path)

            # Save validation metrics with dataset_split metadata
            await model_version_manager.save_quality_metrics(
                model_version_id=model_version["id"],
                metrics=validation_metrics,
                evaluation_dataset_size=len(eval_features),
                dataset_split="validation",
            )

            # Save test metrics with dataset_split metadata if available
            if test_metrics:
                await model_version_manager.save_quality_metrics(
                    model_version_id=model_version["id"],
                    metrics=test_metrics,
                    evaluation_dataset_size=len(test_features),
                    dataset_split="test",
                )

            # Check if model quality meets threshold for activation
            # Use test set metrics if available, otherwise fallback to validation metrics
            final_metrics = test_metrics if test_metrics else validation_metrics
            metrics_source = "test" if test_metrics else "validation"

            accuracy = final_metrics.get("accuracy", 0.0)
            if accuracy >= settings.model_quality_threshold_accuracy:
                await model_version_manager.activate_version(model_version["id"], strategy_id)
                logger.info(
                    "Model activated automatically",
                    version=version,
                    accuracy=accuracy,
                    metrics_source=metrics_source,
                    trace_id=trace_id,
                )
            else:
                logger.info(
                    "Model quality below threshold, not activated",
                    version=version,
                    accuracy=accuracy,
                    metrics_source=metrics_source,
                    threshold=settings.model_quality_threshold_accuracy,
                    trace_id=trace_id,
                )

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
                dataset_id=str(dataset_id),
                duration_seconds=training_duration,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                final_metrics_source=metrics_source,
                trace_id=trace_id,
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
        dataset_id = queue_item.get("dataset_id")
        strategy_id = queue_item.get("strategy_id")
        symbol = queue_item.get("symbol")
        trace_id = queue_item.get("trace_id")

        if not dataset_id:
            logger.warning("Queue item missing dataset_id, skipping", queue_item=queue_item)
            return

        logger.info(
            "Starting queued training after previous completion",
            strategy_id=strategy_id,
            dataset_id=str(dataset_id),
            remaining_queue_size=len(self._training_queue),
        )

        await self._start_training_with_dataset(strategy_id, dataset_id, symbol, trace_id)

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Gracefully shut down training orchestrator.

        - Wait for current training to complete (up to timeout)
        - Log queue state for observability
        """
        logger.info(
            "Shutting down training orchestrator",
            queue_size=len(self._training_queue),
            pending_dataset_builds=len(self._pending_dataset_builds),
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
            Dictionary with training status, queue metrics, and pending dataset builds.
        """
        is_training = self._current_training_task is not None and not self._current_training_task.done()
        now = datetime.utcnow()

        queue_size = len(self._training_queue)
        next_item = self._training_queue[0] if self._training_queue else None
        next_wait_seconds: Optional[float] = None
        next_strategy_id: Optional[str] = None
        next_dataset_id: Optional[str] = None

        if next_item:
            enqueued_at: datetime = next_item.get("enqueued_at", now)
            next_wait_seconds = max((now - enqueued_at).total_seconds(), 0.0)
            next_strategy_id = next_item.get("strategy_id")
            next_dataset_id = str(next_item.get("dataset_id")) if next_item.get("dataset_id") else None

        return {
            "is_training": is_training,
            "queue_size": queue_size,
            "queue_next_wait_time_seconds": next_wait_seconds,
            "next_queued_training_strategy_id": next_strategy_id,
            "next_queued_training_dataset_id": next_dataset_id,
            "pending_dataset_builds": len(self._pending_dataset_builds),
            "metrics": self._metrics.copy(),
        }


# Global training orchestrator instance
training_orchestrator = TrainingOrchestrator()

