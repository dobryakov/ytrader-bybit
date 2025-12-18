"""
Periodic task for target evaluation.

Запускает TargetEvaluator с адаптивным интервалом в фоне.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.target_evaluator import target_evaluator

logger = get_logger(__name__)


class TargetEvaluationTask:
    """Background task that periodically evaluates pending prediction targets."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        """Start background evaluation loop."""
        if self._task and not self._task.done():
            return

        self._stopped.clear()
        self._task = asyncio.create_task(self._evaluation_loop(), name="target_evaluation_task")
        logger.info("TargetEvaluationTask started")

    async def stop(self) -> None:
        """Stop background evaluation loop."""
        if not self._task:
            return

        self._stopped.set()
        try:
            await asyncio.wait_for(self._task, timeout=settings.target_evaluation_max_interval_seconds)
        except asyncio.TimeoutError:
            logger.warning("TargetEvaluationTask stop timed out")
        except Exception as e:
            logger.error("Error stopping TargetEvaluationTask", error=str(e), exc_info=True)
        finally:
            self._task = None
            logger.info("TargetEvaluationTask stopped")

    async def trigger_immediate_check(self, prediction_target_id: str) -> None:
        """
        Trigger immediate evaluation for single prediction target.

        Вызывается event-driven логикой сразу после сохранения prediction_target.
        """
        try:
            await target_evaluator.check_and_evaluate_immediate(prediction_target_id)
        except Exception as e:
            logger.error(
                "Immediate target evaluation trigger failed",
                prediction_target_id=prediction_target_id,
                error=str(e),
                exc_info=True,
            )

    async def _evaluation_loop(self) -> None:
        """Main evaluation loop with adaptive interval."""
        base_interval = settings.target_evaluation_base_interval_seconds
        min_interval = settings.target_evaluation_min_interval_seconds
        max_interval = settings.target_evaluation_max_interval_seconds

        logger.info(
            "TargetEvaluationTask loop started",
            base_interval=base_interval,
            min_interval=min_interval,
            max_interval=max_interval,
        )

        while not self._stopped.is_set():
            try:
                evaluated = await target_evaluator.evaluate_pending_targets()
                # Простейшая адаптация: если что‑то было обработано – спим меньше,
                # если нет pending – спим дольше.
                if evaluated > 0:
                    interval = min_interval
                else:
                    interval = max_interval

                await asyncio.wait_for(self._stopped.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # Нормальный случай – таймаут ожидания, просто продолжаем цикл
                continue
            except Exception as e:
                logger.error("Error in TargetEvaluationTask loop", error=str(e), exc_info=True)
                # На ошибке делаем небольшую паузу, чтобы не крутиться слишком часто
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=base_interval)
                except asyncio.TimeoutError:
                    continue


# Глобальный инстанс по аналогии с другими задачами
target_evaluation_task = TargetEvaluationTask()


