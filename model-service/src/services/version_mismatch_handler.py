"""
Version mismatch handler service.

Automatically triggers dataset build and model retraining when version mismatches
are detected between model training config and current feature/target registry versions.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import asyncio

from ..services.training_orchestrator import training_orchestrator
from ..config.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)


class VersionMismatchHandler:
    """Handles version mismatches by triggering automatic retraining."""

    def __init__(self):
        """Initialize version mismatch handler."""
        # Track last retraining trigger per strategy to prevent spam
        # Format: {strategy_id: {"last_trigger": datetime, "feature_registry_version": str, "target_registry_version": str}}
        self._last_retraining_triggers: Dict[str, Dict[str, Any]] = {}
        
        # Minimum time between retraining triggers for the same strategy (hours)
        self._min_retraining_interval_hours = settings.version_mismatch_retraining_interval_hours

    async def handle_version_mismatch(
        self,
        strategy_id: str,
        asset: str,
        model_feature_registry_version: Optional[str],
        model_target_registry_version: Optional[str],
        current_feature_registry_version: str,
        current_target_registry_version: Optional[str],
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Handle version mismatch by triggering retraining if needed.

        Args:
            strategy_id: Trading strategy identifier
            asset: Trading pair symbol
            model_feature_registry_version: Feature registry version used during model training
            model_target_registry_version: Target registry version used during model training
            current_feature_registry_version: Current feature registry version from feature vector
            current_target_registry_version: Current target registry version (active or from feature vector)
            trace_id: Optional trace ID

        Returns:
            True if retraining was triggered, False otherwise
        """
        # Check if retraining should be triggered
        should_trigger = False
        mismatch_reasons = []

        # Check feature registry version mismatch
        if model_feature_registry_version and model_feature_registry_version != current_feature_registry_version:
            should_trigger = True
            mismatch_reasons.append(
                f"feature_registry: {model_feature_registry_version} -> {current_feature_registry_version}"
            )

        # Check target registry version mismatch
        if (
            current_target_registry_version
            and model_target_registry_version
            and model_target_registry_version != current_target_registry_version
        ):
            should_trigger = True
            mismatch_reasons.append(
                f"target_registry: {model_target_registry_version} -> {current_target_registry_version}"
            )

        if not should_trigger:
            logger.debug(
                "No version mismatch detected, skipping retraining trigger",
                strategy_id=strategy_id,
                asset=asset,
                model_feature_registry_version=model_feature_registry_version,
                model_target_registry_version=model_target_registry_version,
                current_feature_registry_version=current_feature_registry_version,
                current_target_registry_version=current_target_registry_version,
                trace_id=trace_id,
            )
            return False

        # Check if retraining was recently triggered for this strategy
        last_trigger_info = self._last_retraining_triggers.get(strategy_id)
        if last_trigger_info:
            last_trigger_time = last_trigger_info.get("last_trigger")
            if last_trigger_time:
                time_since_last_trigger = datetime.now(timezone.utc) - last_trigger_time
                if time_since_last_trigger < timedelta(hours=self._min_retraining_interval_hours):
                    logger.debug(
                        "Retraining recently triggered for this strategy, skipping",
                        strategy_id=strategy_id,
                        asset=asset,
                        time_since_last_trigger_hours=time_since_last_trigger.total_seconds() / 3600,
                        min_interval_hours=self._min_retraining_interval_hours,
                        mismatch_reasons=mismatch_reasons,
                        trace_id=trace_id,
                    )
                    return False

                # Check if versions haven't changed since last trigger
                if (
                    last_trigger_info.get("feature_registry_version") == current_feature_registry_version
                    and last_trigger_info.get("target_registry_version") == current_target_registry_version
                ):
                    logger.debug(
                        "Versions unchanged since last retraining trigger, skipping",
                        strategy_id=strategy_id,
                        asset=asset,
                        feature_registry_version=current_feature_registry_version,
                        target_registry_version=current_target_registry_version,
                        trace_id=trace_id,
                    )
                    return False

        # Check if training is already in progress
        training_status = training_orchestrator.get_status()
        if training_status.get("is_training"):
            logger.info(
                "Training already in progress, skipping version mismatch retraining trigger",
                strategy_id=strategy_id,
                asset=asset,
                mismatch_reasons=mismatch_reasons,
                trace_id=trace_id,
            )
            return False

        # Trigger retraining with current versions
        logger.info(
            "Version mismatch detected, triggering automatic retraining",
            strategy_id=strategy_id,
            asset=asset,
            mismatch_reasons=mismatch_reasons,
            model_feature_registry_version=model_feature_registry_version,
            model_target_registry_version=model_target_registry_version,
            current_feature_registry_version=current_feature_registry_version,
            current_target_registry_version=current_target_registry_version,
            trace_id=trace_id,
        )

        try:
            # Request dataset build with current target registry version
            # Feature registry version will be "latest" (handled by training_orchestrator)
            # Target registry version will be current_active_version to match current feature vector
            dataset_id = await training_orchestrator.request_dataset_build(
                strategy_id=strategy_id,
                symbol=asset,
                target_registry_version=current_target_registry_version or "latest",
            )

            if dataset_id:
                # Update last trigger info
                self._last_retraining_triggers[strategy_id] = {
                    "last_trigger": datetime.now(timezone.utc),
                    "feature_registry_version": current_feature_registry_version,
                    "target_registry_version": current_target_registry_version,
                    "dataset_id": str(dataset_id),
                }

                logger.info(
                    "Automatic retraining triggered successfully due to version mismatch",
                    strategy_id=strategy_id,
                    asset=asset,
                    dataset_id=str(dataset_id),
                    mismatch_reasons=mismatch_reasons,
                    trace_id=trace_id,
                )
                return True
            else:
                logger.error(
                    "Failed to trigger automatic retraining - dataset build request failed",
                    strategy_id=strategy_id,
                    asset=asset,
                    mismatch_reasons=mismatch_reasons,
                    trace_id=trace_id,
                )
                return False

        except Exception as e:
            logger.error(
                "Failed to trigger automatic retraining due to version mismatch",
                strategy_id=strategy_id,
                asset=asset,
                mismatch_reasons=mismatch_reasons,
                error=str(e),
                exc_info=True,
                trace_id=trace_id,
            )
            return False


# Global instance
version_mismatch_handler = VersionMismatchHandler()

