"""
Buffer persistence service for training orchestrator.

Provides helpers to:
- Load unused execution events from the database on startup
- Mark execution events as used for training after a successful training run
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import json
from uuid import UUID

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
                # When no explicit strategy is provided, try to load for configured strategies first,
                # then fall back to loading all events
                from ..config.settings import settings
                
                strategies = settings.trading_strategy_list if settings.trading_strategy_list else []
                
                if strategies:
                    # Load events for each configured strategy and combine them
                    # Also try normalized versions (hyphen vs underscore)
                    all_events = []
                    seen_event_ids = set()  # Avoid duplicates
                    
                    for strat_id in strategies:
                        # Try exact match
                        events = await self._repo.get_unused_events(
                            strategy_id=strat_id,
                            limit=max_events,
                        )
                        for event in events:
                            event_id = str(event.get("id"))
                            if event_id not in seen_event_ids:
                                all_events.append(event)
                                seen_event_ids.add(event_id)
                        
                        # Try normalized version (hyphen <-> underscore)
                        normalized_id = strat_id.replace("-", "_")
                        if normalized_id != strat_id:
                            events = await self._repo.get_unused_events(
                                strategy_id=normalized_id,
                                limit=max_events,
                            )
                            for event in events:
                                event_id = str(event.get("id"))
                                if event_id not in seen_event_ids:
                                    all_events.append(event)
                                    seen_event_ids.add(event_id)
                        
                        normalized_id = strat_id.replace("_", "-")
                        if normalized_id != strat_id:
                            events = await self._repo.get_unused_events(
                                strategy_id=normalized_id,
                                limit=max_events,
                            )
                            for event in events:
                                event_id = str(event.get("id"))
                                if event_id not in seen_event_ids:
                                    all_events.append(event)
                                    seen_event_ids.add(event_id)
                    
                    # Sort by executed_at and limit to max_events total
                    from datetime import datetime as dt
                    all_events_sorted = sorted(all_events, key=lambda x: x.get("executed_at") or dt.min)
                    raw_events = all_events_sorted[:max_events] if max_events else all_events_sorted
                else:
                    # No strategies configured, load all unused events without strategy filter
                    logger.info(
                        "No strategies configured, loading all unused events without strategy filter",
                    )
                    raw_events = await self._repo.get_all_unused_events(limit=max_events)

            for row in raw_events:
                try:
                    # Handle performance field - it can be a dict, JSON string, or None
                    performance_raw = row.get("performance")
                    if isinstance(performance_raw, str):
                        # Parse JSON string
                        try:
                            performance_data: Dict[str, Any] = json.loads(performance_raw)
                        except (json.JSONDecodeError, TypeError):
                            logger.warning("Failed to parse performance JSON", performance_raw=performance_raw)
                            performance_data = {}
                    elif isinstance(performance_raw, dict):
                        performance_data = performance_raw
                    else:
                        performance_data = {}
                    
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

                    # Normalize datetime fields - ensure they have timezone
                    executed_at = row.get("executed_at")
                    if executed_at and executed_at.tzinfo is None:
                        executed_at = executed_at.replace(tzinfo=timezone.utc)
                    
                    signal_timestamp = row.get("signal_timestamp")
                    if signal_timestamp and signal_timestamp.tzinfo is None:
                        signal_timestamp = signal_timestamp.replace(tzinfo=timezone.utc)
                    
                    # Normalize UUID fields - convert to string
                    event_id = row.get("id")
                    if isinstance(event_id, UUID):
                        event_id = str(event_id)
                    elif event_id:
                        event_id = str(event_id)
                    
                    signal_id = row.get("signal_id")
                    if isinstance(signal_id, UUID):
                        signal_id = str(signal_id)
                    elif signal_id:
                        signal_id = str(signal_id)
                    
                    event = OrderExecutionEvent(
                        event_id=event_id or str(UUID()),
                        order_id=row.get("order_id") or str(signal_id),
                        signal_id=str(signal_id) if signal_id else "",
                        strategy_id=str(row.get("strategy_id")) if row.get("strategy_id") else "unknown",
                        asset=row.get("asset"),
                        side=row.get("side"),
                        execution_price=float(row.get("execution_price")),
                        execution_quantity=float(row.get("execution_quantity")),
                        execution_fees=float(row.get("execution_fees")),
                        executed_at=executed_at,
                        signal_price=float(row.get("signal_price")),
                        signal_timestamp=signal_timestamp,
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


