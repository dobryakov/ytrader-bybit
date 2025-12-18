"""
Periodic task for automatic model retraining.

Проверяет необходимость переобучения модели на основе времени с последнего обучения
и автоматически запускает переобучение при необходимости.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from datetime import datetime, timedelta

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.training_orchestrator import training_orchestrator
from ..database.repositories.model_version_repo import ModelVersionRepository

logger = get_logger(__name__)


class RetrainingTask:
    """Background task that periodically checks and triggers model retraining."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        self._model_version_repo = ModelVersionRepository()

    async def start(self) -> None:
        """Start background retraining check loop."""
        if self._task and not self._task.done():
            return

        self._stopped.clear()
        self._task = asyncio.create_task(self._retraining_check_loop(), name="retraining_task")
        
        check_interval_hours = settings.model_retraining_check_interval_hours
        retraining_interval_days = settings.model_retraining_interval_days
        
        logger.info(
            "RetrainingTask started",
            check_interval_hours=check_interval_hours,
            retraining_interval_days=retraining_interval_days,
        )

    async def stop(self) -> None:
        """Stop background retraining check loop."""
        if not self._task:
            return

        self._stopped.set()
        try:
            # Ждем завершения с таймаутом
            max_wait = settings.model_retraining_check_interval_hours * 3600
            await asyncio.wait_for(self._task, timeout=max_wait)
        except asyncio.TimeoutError:
            logger.warning("RetrainingTask stop timed out")
        except Exception as e:
            logger.error("Error stopping RetrainingTask", error=str(e), exc_info=True)
        finally:
            self._task = None
            logger.info("RetrainingTask stopped")

    async def _get_last_training_time(self, strategy_id: Optional[str] = None) -> Optional[datetime]:
        """
        Get last training time for a strategy from database.
        
        Args:
            strategy_id: Trading strategy identifier
            
        Returns:
            Last training time or None if no models found
        """
        try:
            # Get the most recent model version for this strategy
            versions = await self._model_version_repo.list_by_strategy(
                strategy_id=strategy_id,
                limit=1,
                order_by="trained_at DESC"
            )
            
            if versions and len(versions) > 0:
                last_version = versions[0]
                trained_at = last_version.get("trained_at")
                
                # Convert to datetime if it's a string
                if isinstance(trained_at, str):
                    trained_at = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
                    # Convert to timezone-naive UTC for comparison
                    if trained_at.tzinfo is not None:
                        trained_at = trained_at.replace(tzinfo=None)
                elif isinstance(trained_at, datetime):
                    # Ensure timezone-naive for comparison
                    if trained_at.tzinfo is not None:
                        trained_at = trained_at.replace(tzinfo=None)
                
                return trained_at
            
            return None
        except Exception as e:
            logger.error(
                "Failed to get last training time",
                strategy_id=strategy_id,
                error=str(e),
                exc_info=True,
            )
            return None

    async def _should_retrain(self, strategy_id: Optional[str] = None) -> bool:
        """
        Check if retraining should be triggered based on time interval.
        
        Args:
            strategy_id: Trading strategy identifier
            
        Returns:
            True if retraining should be triggered
        """
        interval_days = settings.model_retraining_interval_days
        
        # Get last training time from database
        last_training_time = await self._get_last_training_time(strategy_id=strategy_id)
        
        if last_training_time is None:
            # No previous training found - trigger initial training
            logger.debug(
                "No previous training found, triggering initial training",
                strategy_id=strategy_id,
            )
            return True
        
        # Normalize timezone for comparison
        if last_training_time.tzinfo is None:
            last_training_time = last_training_time.replace(tzinfo=None)
        
        now = datetime.utcnow()
        time_since_last = now - last_training_time
        interval_timedelta = timedelta(days=interval_days)
        
        if time_since_last >= interval_timedelta:
            logger.info(
                "Retraining interval reached",
                strategy_id=strategy_id,
                time_since_last_days=time_since_last.days,
                interval_days=interval_days,
                last_training_time=last_training_time.isoformat() if last_training_time else None,
            )
            return True
        
        logger.debug(
            "Retraining interval not reached yet",
            strategy_id=strategy_id,
            time_since_last_days=time_since_last.days,
            interval_days=interval_days,
        )
        return False

    async def _retraining_check_loop(self) -> None:
        """Main retraining check loop with periodic interval."""
        check_interval_hours = settings.model_retraining_check_interval_hours
        check_interval_seconds = check_interval_hours * 3600
        
        logger.info(
            "RetrainingTask loop started",
            check_interval_hours=check_interval_hours,
            check_interval_seconds=check_interval_seconds,
        )

        while not self._stopped.is_set():
            try:
                # Get configured strategies
                if settings.trading_strategies:
                    strategies = [s.strip() for s in settings.trading_strategies.split(",") if s.strip()]
                else:
                    strategies = [None]
                
                for strategy_id_str in strategies:
                    strategy_id = strategy_id_str.strip() if strategy_id_str else None
                    
                    # Check if retraining should occur
                    if await self._should_retrain(strategy_id=strategy_id):
                        # Check if training is already in progress
                        status = training_orchestrator.get_status()
                        if status.get("is_training", False):
                            logger.info(
                                "Training already in progress, skipping retraining trigger",
                                strategy_id=strategy_id,
                            )
                            continue
                        
                        # Trigger retraining by requesting dataset build
                        try:
                            logger.info(
                                "Triggering automatic retraining",
                                strategy_id=strategy_id,
                            )
                            await training_orchestrator.request_dataset_build(strategy_id=strategy_id)
                        except Exception as e:
                            logger.error(
                                "Failed to trigger retraining",
                                strategy_id=strategy_id,
                                error=str(e),
                                exc_info=True,
                            )
                
                # Wait for next check interval
                await asyncio.wait_for(self._stopped.wait(), timeout=check_interval_seconds)
            except asyncio.TimeoutError:
                # Normal case - timeout reached, continue loop
                continue
            except Exception as e:
                logger.error("Error in RetrainingTask loop", error=str(e), exc_info=True)
                # On error, wait before retrying
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=check_interval_seconds)
                except asyncio.TimeoutError:
                    continue


# Global retraining task instance
retraining_task = RetrainingTask()

