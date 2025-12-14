"""
Training orchestration service.

Coordinates dataset building via Feature Service, model training, quality evaluation, version management,
and handles training cancellation and restart on new triggers.

Training pipeline uses only market data from Feature Service (not execution_events).
Model learns from market movements (price predictions), not from own trading results.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone, time as dt_time
from uuid import UUID
import asyncio
from uuid import uuid4
import pandas as pd
import numpy as np
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

    async def request_dataset_build(
        self, 
        strategy_id: Optional[str] = None, 
        symbol: Optional[str] = None,
        target_type: str = "classification"
    ) -> Optional[UUID]:
        """
        Request dataset build from Feature Service.

        Args:
            strategy_id: Trading strategy identifier
            symbol: Trading pair symbol (e.g., 'BTCUSDT'). If None, uses first symbol from configured strategies.
            target_type: Target type - 'classification' or 'regression'. Default: 'classification'

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
        # Format datetime as ISO string with Z suffix for UTC
        def format_dt(dt: Optional[datetime]) -> Optional[str]:
            """Format datetime as ISO string with Z suffix."""
            if dt is None:
                return None
            if dt.tzinfo is None:
                # If naive, assume UTC
                return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            # If timezone-aware, convert to UTC and format
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        
        request = {
            "symbol": symbol,
            "split_strategy": SplitStrategy.TIME_BASED.value,
            "train_period_start": format_dt(periods["train_period_start"]),
            "train_period_end": format_dt(periods["train_period_end"]),
            "validation_period_start": format_dt(periods["validation_period_start"]),
            "validation_period_end": format_dt(periods["validation_period_end"]),
            "test_period_start": format_dt(periods["test_period_start"]),
            "test_period_end": format_dt(periods["test_period_end"]),
            "target_config": {
                "type": target_type,
                "horizon": settings.model_prediction_horizon_seconds,  # Configurable horizon in seconds
                "threshold": settings.model_classification_threshold if target_type == "classification" else None,  # Threshold only for classification
            },
            "feature_registry_version": "latest",  # Use latest feature registry version
            "output_format": "parquet",
        }

        # Log request to debug validation periods
        logger.info(
            "Dataset build request prepared",
            symbol=symbol,
            strategy_id=strategy_id,
            train_period_start=request["train_period_start"],
            train_period_end=request["train_period_end"],
            validation_period_start=request["validation_period_start"],
            validation_period_end=request["validation_period_end"],
            test_period_start=request["test_period_start"],
            test_period_end=request["test_period_end"],
            periods_raw=periods,
        )

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
        if pending_build:
            # Dataset was requested by us - use stored strategy_id
            strategy_id = pending_build["strategy_id"]
            symbol = symbol or pending_build.get("symbol")
            # Remove from pending builds
            del self._pending_dataset_builds[dataset_id]
        else:
            # Dataset was not requested by us, but we can still train on it
            # Use default strategy_id (first configured strategy or None)
            strategy_id = self._normalize_strategy_id(None)
            logger.info(
                "Dataset ready notification received for unrequested dataset, will train with default strategy",
                dataset_id=str(dataset_id),
                strategy_id=strategy_id,
                symbol=symbol,
                trace_id=trace_id,
            )

        logger.info(
            "Dataset ready notification received, starting training",
            dataset_id=str(dataset_id),
            strategy_id=strategy_id,
            symbol=symbol,
            trace_id=trace_id,
        )

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

        Periods are rounded to start/end of day to ensure full day coverage and avoid
        issues with data that starts at 00:01:00 instead of exact period boundaries.

        Args:
            reference_time: Reference time for calculation (defaults to UTC now).
                          Typically set to 1 day ago to exclude most recent data.

        Returns:
            Dictionary with keys:
                - train_period_start: datetime (start of day)
                - train_period_end: datetime (end of day)
                - validation_period_start: datetime (start of day)
                - validation_period_end: datetime (end of day)
                - test_period_start: datetime (start of day)
                - test_period_end: datetime (end of day)
        """
        if reference_time is None:
            # Use 1 day ago as reference to exclude most recent data (which may be incomplete)
            reference_time = datetime.now(timezone.utc) - timedelta(days=1)

        # Calculate periods backwards from reference time
        # Round reference_time to end of day (23:59:59) to include all data for that day
        reference_date = reference_time.date()
        test_period_end = datetime.combine(reference_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
        # For N days period, we need N days: from (reference_date - N + 1) to reference_date
        test_period_start_date = reference_date - timedelta(days=settings.model_retraining_test_period_days - 1)
        test_period_start = datetime.combine(test_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)

        # Validation period ends 1 day before test period starts
        validation_period_end_date = test_period_start_date - timedelta(days=1)
        validation_period_end = datetime.combine(validation_period_end_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
        # For N days period, we need N days: from (validation_period_end_date - N + 1) to validation_period_end_date
        validation_period_start_date = validation_period_end_date - timedelta(days=settings.model_retraining_validation_period_days - 1)
        validation_period_start = datetime.combine(validation_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)

        # Train period ends 1 day before validation period starts
        train_period_end_date = validation_period_start_date - timedelta(days=1)
        train_period_end = datetime.combine(train_period_end_date, dt_time(23, 59, 59)).replace(tzinfo=timezone.utc)
        # For N days period, we need N days: from (train_period_end_date - N + 1) to train_period_end_date
        train_period_start_date = train_period_end_date - timedelta(days=settings.model_retraining_train_period_days - 1)
        train_period_start = datetime.combine(train_period_start_date, dt_time(0, 0, 0)).replace(tzinfo=timezone.utc)

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

            # Extract task_type from target_config
            task_type = dataset_meta.target_config.type
            logger.info(
                "Dataset task type determined",
                training_id=training_id,
                dataset_id=str(dataset_id),
                task_type=task_type,
                trace_id=trace_id,
            )

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
            # Exclude non-numeric columns (symbol, timestamp) and target column from features
            exclude_cols = [target_column, "symbol", "timestamp"]
            features_df = df.drop(columns=[col for col in exclude_cols if col in df.columns])
            
            # Keep only numeric columns for features
            features_df = features_df.select_dtypes(include=[np.number])
            
            labels_series = df[target_column]

            if features_df.empty or labels_series.empty:
                logger.error("Empty dataset after loading", training_id=training_id, dataset_id=str(dataset_id), trace_id=trace_id)
                return
            
            # Log distribution for train split (class distribution for classification, statistics for regression)
            if task_type == "classification":
                train_class_dist = labels_series.value_counts().to_dict()
                train_class_dist_pct = {k: (v / len(labels_series) * 100) for k, v in train_class_dist.items()}
                logger.info(
                    "Train dataset loaded with class distribution",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    record_count=len(features_df),
                    class_distribution=train_class_dist,
                    class_distribution_percentage={k: round(v, 2) for k, v in train_class_dist_pct.items()},
                    unique_labels=sorted(labels_series.unique().tolist()),
                    trace_id=trace_id,
                )
            else:  # regression
                logger.info(
                    "Train dataset loaded with target statistics",
                    training_id=training_id,
                    dataset_id=str(dataset_id),
                    record_count=len(features_df),
                    target_mean=float(labels_series.mean()),
                    target_std=float(labels_series.std()),
                    target_min=float(labels_series.min()),
                    target_max=float(labels_series.max()),
                    target_median=float(labels_series.median()),
                    trace_id=trace_id,
                )

            # Perform data quality validation if enabled
            if settings.model_training_quality_checks_enabled:
                quality_issues = self._validate_data_quality(
                    features_df=features_df,
                    labels_series=labels_series,
                    training_id=training_id,
                    trace_id=trace_id,
                )
                if quality_issues.get("critical", False):
                    logger.error(
                        "Data quality validation failed with critical issues",
                        training_id=training_id,
                        dataset_id=str(dataset_id),
                        issues=quality_issues,
                        trace_id=trace_id,
                    )
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
                task_type=task_type,  # Use task_type from dataset target_config
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
                        # Exclude non-numeric columns (symbol, timestamp) and target column from features
                        exclude_cols = [target_column, "symbol", "timestamp"]
                        validation_features = val_df.drop(columns=[col for col in exclude_cols if col in val_df.columns])
                        # Keep only numeric columns for features
                        validation_features = validation_features.select_dtypes(include=[np.number])
                        validation_labels = val_df[target_column]
                        
                        # Log distribution for validation split (class distribution for classification, statistics for regression)
                        if not validation_labels.empty:
                            if task_type == "classification":
                                val_class_dist = validation_labels.value_counts().to_dict()
                                val_class_dist_pct = {k: (v / len(validation_labels) * 100) for k, v in val_class_dist.items()}
                                logger.info(
                                    "Validation dataset loaded",
                                    training_id=training_id,
                                    record_count=len(validation_features),
                                    class_distribution=val_class_dist,
                                    class_distribution_percentage={k: round(v, 2) for k, v in val_class_dist_pct.items()},
                                    unique_labels=sorted(validation_labels.unique().tolist()),
                                    trace_id=trace_id,
                                )
                            else:  # regression
                                logger.info(
                                    "Validation dataset loaded",
                                    training_id=training_id,
                                    record_count=len(validation_features),
                                    target_mean=float(validation_labels.mean()),
                                    target_std=float(validation_labels.std()),
                                    target_min=float(validation_labels.min()),
                                    target_max=float(validation_labels.max()),
                                    target_median=float(validation_labels.median()),
                                    trace_id=trace_id,
                                )
                        else:
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
            # For multi-class classification, pass all class probabilities (2D array)
            # For binary classification, this will still work correctly
            y_pred_proba = model.predict_proba(eval_features) if hasattr(model, "predict_proba") else None

            # Log prediction statistics before evaluation (only for classification)
            if task_type == "classification" and y_pred_proba is not None:
                if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                    # Log average probabilities per class
                    avg_probs = np.mean(y_pred_proba, axis=0)
                    logger.info(
                        "model_predictions_before_evaluation",
                        split="validation",
                        training_id=training_id,
                        avg_class_probabilities={f"class_{i}": float(avg_probs[i]) for i in range(len(avg_probs))},
                        prediction_threshold="argmax (default)",
                    )
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 1:
                    logger.info(
                        "model_predictions_before_evaluation",
                        split="validation",
                        training_id=training_id,
                        avg_class_probability=float(np.mean(y_pred_proba)),
                        prediction_threshold="argmax (default)",
                    )
            elif task_type == "regression":
                # Log regression prediction statistics
                logger.info(
                    "model_predictions_before_evaluation",
                    split="validation",
                    training_id=training_id,
                    avg_predicted_return=float(np.mean(y_pred)),
                    min_predicted_return=float(np.min(y_pred)),
                    max_predicted_return=float(np.max(y_pred)),
                    std_predicted_return=float(np.std(y_pred)),
                )

            validation_metrics = quality_evaluator.evaluate(
                y_true=eval_labels,
                y_pred=pd.Series(y_pred),
                y_pred_proba=y_pred_proba,  # Pass 2D array directly for multi-class (None for regression)
                task_type=task_type,  # Use task_type from dataset target_config
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
                # Check if test split has records before attempting download
                if dataset_meta.test_records == 0:
                    logger.info(
                        "Test split is empty, skipping download",
                        training_id=training_id,
                        dataset_id=str(dataset_id),
                        test_records=dataset_meta.test_records,
                        trace_id=trace_id,
                    )
                    test_file_path = None
                else:
                    logger.info(
                        "Downloading test split for final evaluation",
                        training_id=training_id,
                        dataset_id=str(dataset_id),
                        test_records=dataset_meta.test_records,
                        trace_id=trace_id,
                    )
                    test_file_path = await feature_service_client.download_dataset(
                        dataset_id, split="test", trace_id=trace_id
                    )

                if test_file_path:
                    try:
                        test_df = pd.read_parquet(test_file_path)
                        if target_column in test_df.columns:
                            # Exclude non-numeric columns (symbol, timestamp) and target column from features
                            exclude_cols = [target_column, "symbol", "timestamp"]
                            test_features = test_df.drop(columns=[col for col in exclude_cols if col in test_df.columns])
                            # Keep only numeric columns for features
                            test_features = test_features.select_dtypes(include=[np.number])
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
                                # Log distribution for test split (class distribution for classification, statistics for regression)
                                if task_type == "classification":
                                    test_class_dist = test_labels.value_counts().to_dict()
                                    test_class_dist_pct = {k: (v / len(test_labels) * 100) for k, v in test_class_dist.items()}
                                    logger.info(
                                        "Test dataset loaded",
                                        training_id=training_id,
                                        record_count=len(test_features),
                                        class_distribution=test_class_dist,
                                        class_distribution_percentage={k: round(v, 2) for k, v in test_class_dist_pct.items()},
                                        unique_labels=sorted(test_labels.unique().tolist()),
                                        trace_id=trace_id,
                                    )
                                else:  # regression
                                    logger.info(
                                        "Test dataset loaded",
                                        training_id=training_id,
                                        record_count=len(test_features),
                                        target_mean=float(test_labels.mean()),
                                        target_std=float(test_labels.std()),
                                        target_min=float(test_labels.min()),
                                        target_max=float(test_labels.max()),
                                        target_median=float(test_labels.median()),
                                        trace_id=trace_id,
                                    )
                                    logger.info(
                                        "Test dataset loaded",
                                        training_id=training_id,
                                        record_count=len(test_features),
                                        target_mean=float(test_labels.mean()),
                                        target_std=float(test_labels.std()),
                                        target_min=float(test_labels.min()),
                                        target_max=float(test_labels.max()),
                                        target_median=float(test_labels.median()),
                                        trace_id=trace_id,
                                    )

                                # Evaluate model on test set
                                test_y_pred = model.predict(test_features)
                                # For multi-class classification, pass all class probabilities (2D array)
                                test_y_pred_proba = (
                                    model.predict_proba(test_features) if hasattr(model, "predict_proba") else None
                                )

                                # Log prediction statistics before evaluation (only for classification)
                                if task_type == "classification" and test_y_pred_proba is not None:
                                    if isinstance(test_y_pred_proba, np.ndarray) and test_y_pred_proba.ndim == 2:
                                        # Log average probabilities per class
                                        avg_probs = np.mean(test_y_pred_proba, axis=0)
                                        logger.info(
                                            "model_predictions_before_evaluation",
                                            split="test",
                                            training_id=training_id,
                                            avg_class_probabilities={f"class_{i}": float(avg_probs[i]) for i in range(len(avg_probs))},
                                            prediction_threshold="argmax (default)",
                                        )
                                    elif isinstance(test_y_pred_proba, np.ndarray) and test_y_pred_proba.ndim == 1:
                                        logger.info(
                                            "model_predictions_before_evaluation",
                                            split="test",
                                            training_id=training_id,
                                            avg_class_probability=float(np.mean(test_y_pred_proba)),
                                            prediction_threshold="argmax (default)",
                                        )
                                elif task_type == "regression":
                                    # Log regression prediction statistics
                                    logger.info(
                                        "model_predictions_before_evaluation",
                                        split="test",
                                        training_id=training_id,
                                        avg_predicted_return=float(np.mean(test_y_pred)),
                                        min_predicted_return=float(np.min(test_y_pred)),
                                        max_predicted_return=float(np.max(test_y_pred)),
                                        std_predicted_return=float(np.std(test_y_pred)),
                                    )

                                test_metrics = quality_evaluator.evaluate(
                                    y_true=test_labels,
                                    y_pred=pd.Series(test_y_pred),
                                    y_pred_proba=test_y_pred_proba,  # Pass 2D array directly for multi-class (None for regression)
                                    task_type=task_type,  # Use task_type from dataset target_config
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
                    "task_type": task_type,  # Use task_type from dataset target_config
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

            # Determine activation based on task type
            should_activate = False
            quality_metric = None
            quality_value = None
            threshold_value = None

            if task_type == "classification":
                # For classification: use accuracy
                quality_value = final_metrics.get("accuracy", 0.0)
                threshold_value = settings.model_quality_threshold_accuracy
                quality_metric = "accuracy"
                should_activate = quality_value >= threshold_value
            elif task_type == "regression":
                # For regression: use R² score (primary) and optionally RMSE (secondary)
                r2_score = final_metrics.get("r2_score", -float('inf'))
                rmse = final_metrics.get("rmse", float('inf'))
                
                # Primary check: R² score should meet threshold
                r2_meets_threshold = r2_score >= settings.model_quality_threshold_r2
                
                # Secondary check: RMSE (if threshold is configured)
                rmse_meets_threshold = True
                if settings.model_quality_threshold_rmse is not None:
                    rmse_meets_threshold = rmse <= settings.model_quality_threshold_rmse
                
                quality_metric = "r2_score"
                quality_value = r2_score
                threshold_value = settings.model_quality_threshold_r2
                should_activate = r2_meets_threshold and rmse_meets_threshold
                
                # Log additional RMSE info if threshold is configured
                if settings.model_quality_threshold_rmse is not None:
                    logger.debug(
                        "Regression activation check",
                        version=version,
                        r2_score=r2_score,
                        r2_threshold=settings.model_quality_threshold_r2,
                        rmse=rmse,
                        rmse_threshold=settings.model_quality_threshold_rmse,
                        r2_ok=r2_meets_threshold,
                        rmse_ok=rmse_meets_threshold,
                        metrics_source=metrics_source,
                        trace_id=trace_id,
                    )
            else:
                logger.warning(
                    "Unknown task type, skipping auto-activation",
                    version=version,
                    task_type=task_type,
                    trace_id=trace_id,
                )

            if should_activate:
                await model_version_manager.activate_version(model_version["id"], strategy_id)
                logger.info(
                    "Model activated automatically",
                    version=version,
                    task_type=task_type,
                    quality_metric=quality_metric,
                    quality_value=quality_value,
                    threshold=threshold_value,
                    metrics_source=metrics_source,
                    trace_id=trace_id,
                )
            else:
                logger.info(
                    "Model quality below threshold, not activated",
                    version=version,
                    task_type=task_type,
                    quality_metric=quality_metric,
                    quality_value=quality_value,
                    threshold=threshold_value,
                    metrics_source=metrics_source,
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

    def _validate_data_quality(
        self,
        features_df: pd.DataFrame,
        labels_series: pd.Series,
        training_id: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate data quality before training.

        Checks for:
        - Missing values in features
        - Infinite values in features
        - Constant features (zero variance)
        - Duplicate samples
        - Data leakage (future information in features)

        Args:
            features_df: Feature DataFrame
            labels_series: Target labels Series
            training_id: Training ID for logging
            trace_id: Optional trace ID for request flow tracking

        Returns:
            Dictionary with quality check results and issues
        """
        issues = {
            "critical": False,
            "warnings": [],
            "missing_values": {},
            "infinite_values": {},
            "constant_features": [],
            "duplicate_samples": 0,
            "data_leakage": False,
        }

        # Check for missing values in features
        missing_counts = features_df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        if len(missing_features) > 0:
            issues["missing_values"] = {col: int(count) for col, count in missing_features.items()}
            issues["warnings"].append(f"Found missing values in {len(missing_features)} features")
            logger.warning(
                "Data quality check: missing values found",
                training_id=training_id,
                missing_features=issues["missing_values"],
                trace_id=trace_id,
            )

        # Check for infinite values in features
        infinite_counts = {}
        for col in features_df.columns:
            infinite_count = np.isinf(features_df[col]).sum()
            if infinite_count > 0:
                infinite_counts[col] = int(infinite_count)
        if infinite_counts:
            issues["infinite_values"] = infinite_counts
            issues["warnings"].append(f"Found infinite values in {len(infinite_counts)} features")
            logger.warning(
                "Data quality check: infinite values found",
                training_id=training_id,
                infinite_features=issues["infinite_values"],
                trace_id=trace_id,
            )

        # Check for constant features (zero variance)
        constant_features = []
        for col in features_df.columns:
            if features_df[col].nunique() <= 1:
                constant_features.append(col)
        if constant_features:
            issues["constant_features"] = constant_features
            issues["warnings"].append(f"Found {len(constant_features)} constant features (zero variance)")
            logger.warning(
                "Data quality check: constant features found",
                training_id=training_id,
                constant_features=constant_features,
                trace_id=trace_id,
            )

        # Check for duplicate samples
        # Create a combined DataFrame to check for exact duplicates
        combined_df = features_df.copy()
        combined_df["_label"] = labels_series.values
        duplicate_count = combined_df.duplicated().sum()
        if duplicate_count > 0:
            issues["duplicate_samples"] = int(duplicate_count)
            duplicate_pct = (duplicate_count / len(combined_df)) * 100
            issues["warnings"].append(f"Found {duplicate_count} duplicate samples ({duplicate_pct:.2f}%)")
            logger.warning(
                "Data quality check: duplicate samples found",
                training_id=training_id,
                duplicate_count=duplicate_count,
                duplicate_percentage=round(duplicate_pct, 2),
                trace_id=trace_id,
            )

        # Check for data leakage (future information in features)
        # This is a heuristic check: look for features that might contain future information
        # Common patterns: features with "future", "next", "ahead" in name, or features that correlate perfectly with target
        leakage_features = []
        for col in features_df.columns:
            col_lower = col.lower()
            # Check for suspicious feature names
            if any(keyword in col_lower for keyword in ["future", "next", "ahead", "forward", "predicted"]):
                leakage_features.append(col)
            # Check for perfect correlation with target (might indicate leakage)
            elif len(features_df[col].unique()) == len(labels_series.unique()):
                # If feature has same number of unique values as target, check correlation
                correlation = abs(features_df[col].corr(pd.Series(labels_series.values, index=features_df.index)))
                if correlation > 0.99:  # Very high correlation might indicate leakage
                    leakage_features.append(col)

        if leakage_features:
            issues["data_leakage"] = True
            issues["critical"] = True  # Data leakage is a critical issue
            issues["warnings"].append(f"Potential data leakage detected in {len(leakage_features)} features")
            logger.error(
                "Data quality check: potential data leakage detected",
                training_id=training_id,
                leakage_features=leakage_features,
                trace_id=trace_id,
            )

        # Log quality check summary
        logger.info(
            "Data quality validation completed",
            training_id=training_id,
            total_features=len(features_df.columns),
            total_samples=len(features_df),
            missing_features_count=len(issues["missing_values"]),
            infinite_features_count=len(issues["infinite_values"]),
            constant_features_count=len(issues["constant_features"]),
            duplicate_samples_count=issues["duplicate_samples"],
            data_leakage_detected=issues["data_leakage"],
            warnings_count=len(issues["warnings"]),
            critical_issues=issues["critical"],
            trace_id=trace_id,
        )

        return issues

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

