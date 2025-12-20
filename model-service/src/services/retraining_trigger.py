"""
Retraining trigger service.

Detects quality degradation to trigger model retraining.
Automatic time-based retraining is handled by retraining_task.
"""

from typing import Optional

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..config.settings import settings
from ..config.logging import get_logger

logger = get_logger(__name__)


class RetrainingTrigger:
    """Detects conditions that trigger model retraining."""

    def __init__(self):
        """Initialize retraining trigger."""
        self.model_version_repo = ModelVersionRepository()
        self.quality_metrics_repo = ModelQualityMetricsRepository()

    async def should_retrain(
        self,
        strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Check if model should be retrained based on quality degradation.

        Time-based retraining is handled automatically by retraining_task.
        This method only checks for quality degradation.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            True if retraining should be triggered due to quality degradation
        """
        # Check quality degradation
        if await self._check_quality_degradation(strategy_id):
            logger.info("Quality degradation detected", strategy_id=strategy_id)
            return True

        return False

    async def _check_quality_degradation(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if model quality has degraded below threshold.

        Prefers test set metrics over validation metrics for quality degradation detection.
        Falls back to validation metrics if test set metrics are not available.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            True if quality degradation is detected
        """
        # Get active model version
        active_version = await self.model_version_repo.get_active_by_strategy(strategy_id)
        if not active_version:
            # No active model, don't trigger retraining based on quality
            return False

        model_version_id = active_version["id"]

        # Try to get test set metrics first (preferred for quality assessment)
        latest_metrics = await self.quality_metrics_repo.get_latest_by_model_version(
            model_version_id, metric_name="accuracy", dataset_split="test"
        )
        metrics_source = "test"

        # Fallback to validation metrics if test set not available
        if not latest_metrics:
            logger.debug(
                "Test set metrics not available, using validation metrics for quality degradation check",
                strategy_id=strategy_id,
                model_version_id=str(model_version_id),
            )
            latest_metrics = await self.quality_metrics_repo.get_latest_by_model_version(
                model_version_id, metric_name="accuracy", dataset_split="validation"
            )
            metrics_source = "validation"

        # Final fallback: try without dataset_split filter (backward compatibility)
        if not latest_metrics:
            logger.debug(
                "Validation metrics not available, using any available metrics (backward compatibility)",
                strategy_id=strategy_id,
                model_version_id=str(model_version_id),
            )
            latest_metrics = await self.quality_metrics_repo.get_latest_by_model_version(
                model_version_id, metric_name="accuracy"
            )
            metrics_source = "any"

        if not latest_metrics:
            # No quality metrics available, can't check degradation
            logger.debug(
                "No quality metrics available for quality degradation check",
                strategy_id=strategy_id,
                model_version_id=str(model_version_id),
            )
            return False

        accuracy = float(latest_metrics["metric_value"])
        threshold = settings.model_activation_threshold

        if accuracy < threshold:
            logger.warning(
                "Model quality below threshold",
                strategy_id=strategy_id,
                accuracy=accuracy,
                threshold=threshold,
                metrics_source=metrics_source,
                model_version_id=str(model_version_id),
            )
            return True

        return False

    def record_retraining(self, strategy_id: Optional[str] = None) -> None:
        """
        Record that retraining has occurred.

        Note: This method is kept for backward compatibility but no longer stores
        retraining time in memory. Retraining time is now tracked in the database
        via model_versions.trained_at field.

        Args:
            strategy_id: Trading strategy identifier
        """
        logger.debug("Retraining recorded (time tracked in database)", strategy_id=strategy_id)


# Global retraining trigger instance
retraining_trigger = RetrainingTrigger()

