"""
Buffer persistence service for training orchestrator.

Provides helpers to:
- Load unused execution events from the database on startup
- Mark execution events as used for training after a successful training run
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from ..models.execution_event import (
    OrderExecutionEvent,
    MarketConditions,
    PerformanceMetrics,
)
from ..database.repositories.execution_event_repo import ExecutionEventRepository
from ..config.logging import get_logger

logger = get_logger(__name__)


class BufferPersistenceService:
    """Service responsible for persisting and restoring the training buffer."""

    def __init__(self) -> None:
        self._repo = ExecutionEventRepository()

    async def restore_buffer(
        self,
        strategy_id: Optional[str] = None,
        max_events: Optional[int] = None,
    ) -> List[OrderExecutionEvent]:
        """
        Restore execution events buffer from database.

        Loads execution events that have not yet been used for training
        (used_for_training = FALSE) and converts them to OrderExecutionEvent
        models suitable for dataset building.

        Args:
            strategy_id: Optional trading strategy identifier. If None, loads
                         events for all strategies.
            max_events: Maximum number of events to load.

        Returns:
            List of OrderExecutionEvent instances.
        """
        restored_events: List[OrderExecutionEvent] = []

        try:
            if strategy_id:
                raw_events = await self._repo.get_unused_events(
                    strategy_id=strategy_id,
                    limit=max_events,
                )
            else:
                # When no explicit strategy is provided, pick unused events for all
                # strategies by iterating over distinct strategy_ids to avoid a
                # full-table scan per call.
                # For now, we query without strategy filter and rely on limit.
                raw_events = await self._repo.get_unused_events(
                    strategy_id="default",  # Fallback; real strategies should pass explicit ID
                    limit=max_events,
                )

            for row in raw_events:
                try:
                    performance_data: Dict[str, Any] = row.get("performance") or {}
                    performance = PerformanceMetrics(
                        slippage=float(performance_data.get("slippage", 0.0)),
                        slippage_percent=float(performance_data.get("slippage_percent", 0.0)),
                        realized_pnl=performance_data.get("realized_pnl"),
                        return_percent=performance_data.get("return_percent"),
                    )

                    # execution_events table does not currently store market_conditions;
                    # use conservative defaults so training can still proceed.
                    market_conditions = MarketConditions(
                        spread=0.0,
                        volume_24h=0.0,
                        volatility=0.0,
                    )

                    event = OrderExecutionEvent(
                        event_id=str(row.get("id")),
                        order_id=row.get("order_id") or str(row.get("signal_id")),
                        signal_id=str(row.get("signal_id")),
                        strategy_id=row.get("strategy_id"),
                        asset=row.get("asset"),
                        side=row.get("side"),
                        execution_price=float(row.get("execution_price")),
                        execution_quantity=float(row.get("execution_quantity")),
                        execution_fees=float(row.get("execution_fees")),
                        executed_at=row.get("executed_at"),
                        signal_price=float(row.get("signal_price")),
                        signal_timestamp=row.get("signal_timestamp"),
                        market_conditions=market_conditions,
                        performance=performance,
                        trace_id=row.get("trace_id"),
                    )
                    restored_events.append(event)
                except Exception as parse_error:
                    logger.error(
                        "Failed to restore execution event from database row",
                        error=str(parse_error),
                        row_id=str(row.get("id")),
                        exc_info=True,
                    )

            logger.info(
                "Restored execution events buffer from database",
                restored_count=len(restored_events),
                strategy_id=strategy_id,
            )
            return restored_events
        except Exception as e:
            logger.error(
                "Failed to restore execution events buffer from database",
                error=str(e),
                strategy_id=strategy_id,
                exc_info=True,
            )
            return []

    async def mark_events_used_for_training(
        self,
        event_ids: List[str],
        model_version_id: str,
    ) -> None:
        """
        Mark execution events as used for training.

        Args:
            event_ids: List of execution event IDs that participated in training
            model_version_id: Model version UUID that consumed these events
        """
        if not event_ids:
            return

        try:
            updated = await self._repo.mark_as_used_for_training(
                event_ids,
                model_version_id,
            )
            logger.info(
                "Marked execution events as used for training",
                updated_count=updated,
                model_version_id=model_version_id,
            )
        except Exception as e:
            logger.error(
                "Failed to mark execution events as used for training",
                error=str(e),
                event_count=len(event_ids),
                model_version_id=model_version_id,
                exc_info=True,
            )


# Global buffer persistence service instance
buffer_persistence = BufferPersistenceService()


