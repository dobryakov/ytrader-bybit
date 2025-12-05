"""
Quality monitoring service.

Periodically evaluates model quality, detects degradation,
and triggers alerts.
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime, timedelta
import asyncio

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..database.connection import db_pool
from ..services.model_version_manager import model_version_manager
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class QualityMonitor:
    """Monitors model quality and detects degradation."""

    def __init__(self):
        """Initialize quality monitor."""
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._evaluation_interval_seconds = getattr(settings, "model_quality_evaluation_interval_seconds", 3600)  # Default 1 hour

    async def start(self) -> None:
        """Start periodic quality monitoring."""
        if self._running:
            logger.warning("Quality monitor already running")
            return

        self._running = True
        logger.info("Starting quality monitor", evaluation_interval_seconds=self._evaluation_interval_seconds)

        try:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Quality monitor started")
        except Exception as e:
            logger.error("Failed to start quality monitor", error=str(e), exc_info=True)
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop quality monitoring."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping quality monitor")

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Quality monitor stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._evaluate_all_active_models()
                await asyncio.sleep(self._evaluation_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Quality monitor cancelled")
                break
            except Exception as e:
                logger.error("Error in quality monitoring loop", error=str(e), exc_info=True)
                # Continue monitoring even if one evaluation fails
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _evaluate_all_active_models(self) -> None:
        """Evaluate quality for all active models."""
        try:
            repo = ModelVersionRepository()
            active_models = await repo._fetch("SELECT * FROM model_versions WHERE is_active = true")

            logger.info("Evaluating quality for active models", count=len(active_models))

            for record in active_models:
                model_version = repo._record_to_dict(record)
                await self._evaluate_model_quality(UUID(model_version["id"]), model_version)

        except Exception as e:
            logger.error("Failed to evaluate active models", error=str(e), exc_info=True)

    async def _evaluate_model_quality(self, model_version_id: UUID, model_version: Dict[str, Any]) -> None:
        """
        Evaluate quality for a specific model using ML metrics from validation/test sets.

        **NOTE**: Quality monitoring now uses only ML metrics (accuracy, precision, recall, F1, MSE, MAE, R2)
        from validation/test sets stored during training. Trading metrics from execution_events are no longer
        used for quality monitoring, as training pipeline uses market-data-only approach.

        Args:
            model_version_id: Model version UUID
            model_version: Model version record
        """
        try:
            strategy_id = model_version.get("strategy_id")

            # Get latest ML metrics from quality_metrics table (saved during training)
            metrics_repo = ModelQualityMetricsRepository()

            # Get latest metrics for this model version (get all, then filter by latest evaluation)
            all_metrics = await metrics_repo.get_by_model_version(model_version_id)

            if not all_metrics:
                logger.debug(
                    "No quality metrics found for model version",
                    model_version_id=str(model_version_id),
                    strategy_id=strategy_id,
                )
                return

            # Group metrics by evaluation time and get the latest evaluation
            # Extract ML metrics (accuracy, precision, recall, F1, MSE, MAE, R2) from latest evaluation
            ml_metrics = {}
            latest_evaluation_time = None
            
            for metric_record in all_metrics:
                evaluated_at = metric_record.get("evaluated_at")
                if evaluated_at and (latest_evaluation_time is None or evaluated_at > latest_evaluation_time):
                    latest_evaluation_time = evaluated_at
            
            # Extract metrics from latest evaluation
            for metric_record in all_metrics:
                if metric_record.get("evaluated_at") == latest_evaluation_time:
                    metric_name = metric_record.get("metric_name")
                    metric_value = metric_record.get("metric_value")
                    if metric_name in ["accuracy", "precision", "recall", "f1_score", "mse", "mae", "r2_score", "roc_auc"]:
                        ml_metrics[metric_name] = float(metric_value) if metric_value is not None else 0.0

            if not ml_metrics:
                logger.debug(
                    "No ML metrics found in quality metrics",
                    model_version_id=str(model_version_id),
                    strategy_id=strategy_id,
                )
                return

            logger.info(
                "Evaluated model quality using ML metrics",
                model_version_id=str(model_version_id),
                strategy_id=strategy_id,
                ml_metrics=ml_metrics,
            )

            # Check for quality degradation using ML metrics
            await self._check_quality_degradation(model_version_id, ml_metrics)

        except Exception as e:
            logger.error("Failed to evaluate model quality", model_version_id=str(model_version_id), error=str(e), exc_info=True)

    async def _check_quality_degradation(self, model_version_id: UUID, current_metrics: Dict[str, float]) -> None:
        """
        Check if model quality has degraded and trigger alerts if needed.

        Args:
            model_version_id: Model version UUID
            current_metrics: Current quality metrics
        """
        try:
            # Get previous metrics for comparison
            metrics_repo = ModelQualityMetricsRepository()

            # Get latest win_rate and profit_factor from previous evaluation
            previous_win_rate = await metrics_repo.get_latest_by_model_version(model_version_id, metric_name="win_rate")
            previous_profit_factor = await metrics_repo.get_latest_by_model_version(model_version_id, metric_name="profit_factor")

            # Check thresholds using ML metrics (configurable)
            accuracy_threshold = getattr(settings, "model_quality_threshold_accuracy", 0.75)
            f1_threshold = getattr(settings, "model_quality_threshold_f1", 0.7)

            degradation_detected = False
            degradation_reasons = []

            # Check accuracy threshold
            if current_metrics.get("accuracy", 0) < accuracy_threshold:
                degradation_detected = True
                degradation_reasons.append(f"accuracy {current_metrics.get('accuracy', 0):.2f} below threshold {accuracy_threshold}")

            # Check F1 score threshold (for classification)
            if current_metrics.get("f1_score", 0) < f1_threshold:
                degradation_detected = True
                degradation_reasons.append(f"f1_score {current_metrics.get('f1_score', 0):.2f} below threshold {f1_threshold}")

            # Compare with previous metrics if available
            previous_accuracy = await metrics_repo.get_latest_by_model_version(model_version_id, metric_name="accuracy")
            if previous_accuracy and previous_accuracy.get("metric_value"):
                prev_accuracy = float(previous_accuracy["metric_value"])
                current_accuracy = current_metrics.get("accuracy", 0)
                if current_accuracy < prev_accuracy * 0.8:  # 20% degradation
                    degradation_detected = True
                    degradation_reasons.append(f"accuracy dropped from {prev_accuracy:.2f} to {current_accuracy:.2f}")

            previous_f1 = await metrics_repo.get_latest_by_model_version(model_version_id, metric_name="f1_score")
            if previous_f1 and previous_f1.get("metric_value"):
                prev_f1 = float(previous_f1["metric_value"])
                current_f1 = current_metrics.get("f1_score", 0)
                if current_f1 < prev_f1 * 0.8:  # 20% degradation
                    degradation_detected = True
                    degradation_reasons.append(f"f1_score dropped from {prev_f1:.2f} to {current_f1:.2f}")

            if degradation_detected:
                logger.warning(
                    "Model quality degradation detected",
                    model_version_id=str(model_version_id),
                    reasons=degradation_reasons,
                    current_metrics=current_metrics,
                )
                # TODO: Trigger alert/notification mechanism

        except Exception as e:
            logger.error("Failed to check quality degradation", model_version_id=str(model_version_id), error=str(e), exc_info=True)


# Global quality monitor instance
quality_monitor = QualityMonitor()

