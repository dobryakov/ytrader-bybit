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
        Evaluate quality for a specific model based on recent execution events.

        Args:
            model_version_id: Model version UUID
            model_version: Model version record
        """
        try:
            strategy_id = model_version.get("strategy_id")

            # Check if execution_events table exists
            async with db_pool.get_connection() as conn:
                table_exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'execution_events'
                    )
                """
                )

                if not table_exists:
                    logger.debug("execution_events table does not exist, skipping quality evaluation", model_version_id=str(model_version_id))
                    return

                # Get recent execution events for this strategy (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)

                query = """
                    SELECT 
                        COUNT(*) as total_orders,
                        COUNT(*) FILTER (WHERE (performance->>'realized_pnl')::numeric > 0) as successful_orders,
                        AVG((performance->>'realized_pnl')::numeric) as avg_pnl,
                        SUM((performance->>'realized_pnl')::numeric) as total_pnl,
                        COUNT(DISTINCT signal_id) as unique_signals
                    FROM execution_events
                    WHERE strategy_id = $1
                        AND executed_at >= $2
                """

                result = await conn.fetchrow(query, strategy_id, cutoff_time)

                if not result or result["total_orders"] == 0:
                    logger.debug("No recent execution events for quality evaluation", model_version_id=str(model_version_id), strategy_id=strategy_id)
                    return

                total_orders = result["total_orders"]
                successful_orders = result["successful_orders"]
                win_rate = (successful_orders / total_orders) if total_orders > 0 else 0.0
                total_pnl = float(result["total_pnl"] or 0)
                avg_pnl = float(result["avg_pnl"] or 0)

                # Calculate additional metrics
                # Sharpe ratio approximation (simplified)
                # Profit factor
                positive_pnl = await conn.fetchval(
                    """
                    SELECT SUM((performance->>'realized_pnl')::numeric)
                    FROM execution_events
                    WHERE strategy_id = $1
                        AND executed_at >= $2
                        AND (performance->>'realized_pnl')::numeric > 0
                """,
                    strategy_id,
                    cutoff_time,
                )
                negative_pnl = await conn.fetchval(
                    """
                    SELECT ABS(SUM((performance->>'realized_pnl')::numeric))
                    FROM execution_events
                    WHERE strategy_id = $1
                        AND executed_at >= $2
                        AND (performance->>'realized_pnl')::numeric < 0
                """,
                    strategy_id,
                    cutoff_time,
                )

                profit_factor = (float(positive_pnl or 0) / float(negative_pnl or 1)) if negative_pnl else (float(positive_pnl or 0) if positive_pnl else 0.0)

                # Store metrics
                metrics = {
                    "win_rate": win_rate,
                    "total_pnl": total_pnl,
                    "avg_pnl": avg_pnl,
                    "profit_factor": profit_factor,
                }

                await model_version_manager.save_quality_metrics(
                    model_version_id=model_version_id,
                    metrics=metrics,
                    evaluation_dataset_size=total_orders,
                    metadata={
                        "evaluation_period_hours": 24,
                        "cutoff_time": cutoff_time.isoformat(),
                    },
                )

                logger.info(
                    "Evaluated model quality",
                    model_version_id=str(model_version_id),
                    strategy_id=strategy_id,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    profit_factor=profit_factor,
                )

                # Check for quality degradation
                await self._check_quality_degradation(model_version_id, metrics)

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

            # Check thresholds (configurable)
            win_rate_threshold = getattr(settings, "model_quality_threshold_win_rate", 0.5)
            profit_factor_threshold = getattr(settings, "model_quality_threshold_profit_factor", 1.0)

            degradation_detected = False
            degradation_reasons = []

            if current_metrics.get("win_rate", 0) < win_rate_threshold:
                degradation_detected = True
                degradation_reasons.append(f"win_rate {current_metrics['win_rate']:.2f} below threshold {win_rate_threshold}")

            if current_metrics.get("profit_factor", 0) < profit_factor_threshold:
                degradation_detected = True
                degradation_reasons.append(f"profit_factor {current_metrics['profit_factor']:.2f} below threshold {profit_factor_threshold}")

            # Compare with previous metrics if available
            if previous_win_rate and previous_win_rate.get("metric_value"):
                prev_win_rate = float(previous_win_rate["metric_value"])
                if current_metrics.get("win_rate", 0) < prev_win_rate * 0.8:  # 20% degradation
                    degradation_detected = True
                    degradation_reasons.append(f"win_rate dropped from {prev_win_rate:.2f} to {current_metrics['win_rate']:.2f}")

            if previous_profit_factor and previous_profit_factor.get("metric_value"):
                prev_profit_factor = float(previous_profit_factor["metric_value"])
                if current_metrics.get("profit_factor", 0) < prev_profit_factor * 0.8:  # 20% degradation
                    degradation_detected = True
                    degradation_reasons.append(f"profit_factor dropped from {prev_profit_factor:.2f} to {current_metrics['profit_factor']:.2f}")

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

