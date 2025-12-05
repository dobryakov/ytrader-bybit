"""
Retraining trigger service.

Detects scheduled periodic retraining, data accumulation thresholds,
and quality degradation detection to trigger model retraining.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import re

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
        self._last_retraining_time: Dict[str, datetime] = {}  # strategy_id -> last retraining time

    async def should_retrain(
        self,
        strategy_id: Optional[str] = None,
    ) -> bool:
        """
        Check if model should be retrained.

        Uses time-based or scheduled retraining triggers. No longer depends on execution_events accumulation.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            True if retraining should be triggered
        """
        # Check scheduled periodic retraining (if configured)
        if await self._check_scheduled_retraining(strategy_id):
            logger.info("Scheduled retraining triggered", strategy_id=strategy_id)
            return True

        # Check time-based retraining (interval-based)
        if await self._check_time_based_retraining(strategy_id):
            logger.info("Time-based retraining triggered", strategy_id=strategy_id)
            return True

        # Check quality degradation
        if await self._check_quality_degradation(strategy_id):
            logger.info("Quality degradation detected", strategy_id=strategy_id)
            return True

        return False

    async def _check_scheduled_retraining(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if scheduled periodic retraining should occur.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            True if scheduled retraining should occur
        """
        if not settings.model_retraining_schedule:
            return False

        # Parse schedule (format: "daily", "weekly", "hourly", or cron-like "0 0 * * *")
        schedule = settings.model_retraining_schedule.lower()

        # Get last retraining time for this strategy
        last_time = self._last_retraining_time.get(strategy_id or "default")

        if not last_time:
            # First time, check if we should retrain now
            return True

        now = datetime.utcnow()
        time_since_last = now - last_time

        if schedule == "hourly":
            return time_since_last >= timedelta(hours=1)
        elif schedule == "daily":
            return time_since_last >= timedelta(days=1)
        elif schedule == "weekly":
            return time_since_last >= timedelta(weeks=1)
        else:
            # Try to parse as cron-like schedule (simplified)
            # For now, just check if it's a valid schedule string
            logger.warning("Complex cron schedule not fully supported", schedule=schedule)
            return False

    async def _check_time_based_retraining(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if time-based retraining should occur.

        Triggers training when configured interval (days) has passed since last training.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            True if time-based retraining should occur
        """
        interval_days = settings.model_retraining_interval_days

        # Get last retraining time for this strategy
        last_time = self._last_retraining_time.get(strategy_id or "default")

        if not last_time:
            # First time, check if we should retrain now
            logger.debug("No previous retraining time found, triggering initial training", strategy_id=strategy_id)
            return True

        now = datetime.utcnow()
        time_since_last = now - last_time
        interval_timedelta = timedelta(days=interval_days)

        if time_since_last >= interval_timedelta:
            logger.debug(
                "Time-based retraining interval reached",
                strategy_id=strategy_id,
                time_since_last_days=time_since_last.days,
                interval_days=interval_days,
            )
            return True

        return False

    async def _check_quality_degradation(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if model quality has degraded below threshold.

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

        # Get latest quality metrics
        latest_metrics = await self.quality_metrics_repo.get_latest_by_model_version(
            model_version_id, metric_name="accuracy"
        )

        if not latest_metrics:
            # No quality metrics available, can't check degradation
            return False

        accuracy = float(latest_metrics["metric_value"])
        threshold = settings.model_quality_threshold_accuracy

        if accuracy < threshold:
            logger.warning(
                "Model quality below threshold",
                strategy_id=strategy_id,
                accuracy=accuracy,
                threshold=threshold,
            )
            return True

        return False

    def record_retraining(self, strategy_id: Optional[str] = None) -> None:
        """
        Record that retraining has occurred.

        Args:
            strategy_id: Trading strategy identifier
        """
        self._last_retraining_time[strategy_id or "default"] = datetime.utcnow()
        logger.debug("Recorded retraining time", strategy_id=strategy_id)


# Global retraining trigger instance
retraining_trigger = RetrainingTrigger()

