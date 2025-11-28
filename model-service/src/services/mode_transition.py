"""
Mode transition service.

Automatically transitions from warm-up mode to model-based generation
when model quality reaches configured threshold.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..services.model_version_manager import model_version_manager
from ..config.settings import settings
from ..config.logging import get_logger, bind_context

logger = get_logger(__name__)


class ModeTransition:
    """Manages transitions between warm-up and model-based modes."""

    def __init__(self):
        """Initialize mode transition service."""
        self.model_version_repo = ModelVersionRepository()
        self.quality_metrics_repo = ModelQualityMetricsRepository()
        self.quality_threshold = settings.model_quality_threshold_accuracy

    async def check_and_transition(
        self,
        strategy_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Check if mode transition should occur and perform it if needed.

        Args:
            strategy_id: Trading strategy identifier (None for default strategy)
            trace_id: Trace ID for request flow tracking

        Returns:
            True if transition occurred, False otherwise
        """
        bind_context(strategy_id=strategy_id, trace_id=trace_id)

        logger.debug("Checking mode transition", strategy_id=strategy_id)

        # Check if we're currently in warm-up mode
        active_model = await self.model_version_repo.get_active_by_strategy(strategy_id)
        if active_model:
            # Already have an active model, not in warm-up mode
            logger.debug("Active model exists, no transition needed", strategy_id=strategy_id, model_version=active_model["version"])
            return False

        # Find the best model version for this strategy that meets quality threshold
        best_model = await self._find_best_model(strategy_id)
        if not best_model:
            logger.debug("No model meeting quality threshold found", strategy_id=strategy_id, threshold=self.quality_threshold)
            return False

        # Check if model quality meets threshold
        model_quality = await self._get_model_quality(best_model["id"])
        if not model_quality or model_quality < self.quality_threshold:
            logger.debug(
                "Model quality below threshold",
                strategy_id=strategy_id,
                model_version=best_model["version"],
                quality=model_quality,
                threshold=self.quality_threshold,
            )
            return False

        # Transition to model-based mode
        try:
            # Activate the model
            activated_model = await model_version_manager.activate_version(best_model["id"], strategy_id)
            if activated_model:
                logger.info(
                    "Mode transition: warm-up -> model-based",
                    strategy_id=strategy_id,
                    model_version=best_model["version"],
                    quality=model_quality,
                    threshold=self.quality_threshold,
                    trace_id=trace_id,
                )
                return True
            else:
                logger.error("Failed to activate model during transition", strategy_id=strategy_id, model_version=best_model["version"])
                return False

        except Exception as e:
            logger.error(
                "Error during mode transition",
                strategy_id=strategy_id,
                model_version=best_model["version"],
                error=str(e),
                exc_info=True,
            )
            return False

    async def _find_best_model(self, strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best model version for a strategy.

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            Best model version record or None if not found
        """
        # Get all model versions for this strategy, ordered by training date (newest first)
        model_versions = await self.model_version_repo.list_by_strategy(
            strategy_id=strategy_id,
            limit=10,  # Check last 10 versions
            order_by="trained_at DESC",
        )

        if not model_versions:
            return None

        # Find the model with highest quality that meets threshold
        best_model = None
        best_quality = 0.0

        for model_version in model_versions:
            # Skip if already active
            if model_version.get("is_active"):
                continue

            # Get model quality
            quality = await self._get_model_quality(model_version["id"])
            if quality and quality >= self.quality_threshold and quality > best_quality:
                best_quality = quality
                best_model = model_version

        return best_model

    async def _get_model_quality(self, model_version_id: str) -> Optional[float]:
        """
        Get model quality score (accuracy or primary metric).

        Args:
            model_version_id: Model version UUID

        Returns:
            Quality score (0-1) or None if not available
        """
        try:
            from uuid import UUID
            model_version_uuid = UUID(model_version_id) if isinstance(model_version_id, str) else model_version_id

            # Try to get latest quality metrics for this model version
            # Try accuracy first
            metric = await self.quality_metrics_repo.get_latest_by_model_version(
                model_version_id=model_version_uuid,
                metric_name="accuracy",
            )
            if metric:
                quality = float(metric["metric_value"])
                # Normalize to 0-1 range if needed
                if quality > 1.0:
                    quality = quality / 100.0  # Assume percentage
                return quality

            # Try f1_score
            metric = await self.quality_metrics_repo.get_latest_by_model_version(
                model_version_id=model_version_uuid,
                metric_name="f1_score",
            )
            if metric:
                quality = float(metric["metric_value"])
                if quality > 1.0:
                    quality = quality / 100.0
                return quality

            # Try win_rate
            metric = await self.quality_metrics_repo.get_latest_by_model_version(
                model_version_id=model_version_uuid,
                metric_name="win_rate",
            )
            if metric:
                quality = float(metric["metric_value"])
                if quality > 1.0:
                    quality = quality / 100.0
                return quality

            # Get any metric as fallback
            metrics = await self.quality_metrics_repo.get_by_model_version(model_version_id=model_version_uuid)
            if metrics:
                quality = float(metrics[0]["metric_value"])
                if quality > 1.0:
                    quality = quality / 100.0
                return quality

            return None

        except Exception as e:
            logger.warning("Failed to get model quality", model_version_id=model_version_id, error=str(e))
            return None

    async def force_warmup_mode(
        self,
        strategy_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Force transition back to warm-up mode (deactivate all models).

        Args:
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            True if transition occurred, False otherwise
        """
        bind_context(strategy_id=strategy_id, trace_id=trace_id)

        logger.info("Forcing warm-up mode", strategy_id=strategy_id, trace_id=trace_id)

        try:
            # Deactivate all models for this strategy
            count = await self.model_version_repo.deactivate_all_for_strategy(strategy_id)
            if count > 0:
                logger.info("Deactivated models, entered warm-up mode", strategy_id=strategy_id, deactivated_count=count)
                return True
            else:
                logger.debug("No active models to deactivate", strategy_id=strategy_id)
                return False

        except Exception as e:
            logger.error("Error forcing warm-up mode", strategy_id=strategy_id, error=str(e), exc_info=True)
            return False

    async def get_current_mode(self, strategy_id: Optional[str] = None) -> str:
        """
        Get current mode (warm-up or model-based).

        Args:
            strategy_id: Trading strategy identifier

        Returns:
            'warmup' or 'model-based'
        """
        active_model = await self.model_version_repo.get_active_by_strategy(strategy_id)
        if active_model:
            return "model-based"
        else:
            return "warmup"


# Global mode transition instance
mode_transition = ModeTransition()

