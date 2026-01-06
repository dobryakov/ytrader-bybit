"""Periodic order state synchronization task.

This background task periodically synchronizes order state with Bybit exchange
to ensure orders that were filled/cancelled but missed WebSocket events are
properly updated in the database.
"""

import asyncio
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.order_state_sync import OrderStateSync
from ..utils.tracing import generate_trace_id, set_trace_id

logger = get_logger(__name__)


class OrderStateSyncTask:
    """Background task that periodically synchronizes order state with Bybit."""

    def __init__(self) -> None:
        """Initialize order state sync task."""
        self._sync_service = OrderStateSync()
        self._should_run = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start periodic sync loop."""
        if self._should_run:
            return

        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

        logger.info(
            "order_state_sync_task_started",
            interval=settings.order_manager_order_sync_interval,
        )

    async def stop(self) -> None:
        """Stop periodic sync loop."""
        if not self._should_run:
            return

        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("order_state_sync_task_stopped")

    async def _run(self) -> None:
        """Internal loop that triggers sync at configured interval."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.order_manager_order_sync_interval)
                if not self._should_run:
                    break

                logger.info(
                    "order_state_sync_periodic_started",
                    trace_id=trace_id,
                )

                sync_result = await self._sync_service.sync_active_orders(trace_id=trace_id)

                logger.info(
                    "order_state_sync_periodic_completed",
                    synced_count=sync_result.get("synced_count", 0),
                    discrepancies_count=len(sync_result.get("discrepancies", [])),
                    errors_count=len(sync_result.get("errors", [])),
                    trace_id=trace_id,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "order_state_sync_periodic_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs - wait before retrying
                await asyncio.sleep(60)  # Wait 1 minute before retrying on error

