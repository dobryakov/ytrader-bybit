"""Background tasks for periodic position snapshots and cleanup."""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.position_manager import PositionManager
from ..utils.tracing import generate_trace_id, set_trace_id


logger = get_logger(__name__)


class PositionSnapshotTask:
    """Background task that periodically snapshots all positions.

    This is the analogue of the Order Manager's `PositionSnapshotTask`, adapted
    to the Position Manager architecture and configuration.
    """

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager = PositionManager()

    async def start(self) -> None:
        """Start the snapshot loop."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._snapshot_loop())
        logger.info(
            "position_snapshot_task_started",
            interval=settings.position_manager_snapshot_interval,
        )

    async def stop(self) -> None:
        """Stop the snapshot loop."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("position_snapshot_task_stopped")

    async def _snapshot_loop(self) -> None:
        """Loop that periodically snapshots all positions."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.position_manager_snapshot_interval)

                if not self._should_run:
                    break

                positions = await self._position_manager.get_all_positions()
                snapshot_count = 0

                for position in positions:
                    try:
                        await self._position_manager.create_position_snapshot(
                            position,
                            trace_id=trace_id,
                        )
                        snapshot_count += 1
                    except Exception as e:  # pragma: no cover - defensive
                        logger.error(
                            "position_snapshot_failed",
                            position_id=str(position.id),
                            asset=position.asset,
                            mode=position.mode,
                            error=str(e),
                            trace_id=trace_id,
                        )

                if snapshot_count > 0:
                    logger.info(
                        "position_snapshots_created",
                        count=snapshot_count,
                        trace_id=trace_id,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "position_snapshot_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Small delay before retrying the loop to avoid hot failure loop
                await asyncio.sleep(60)


class PositionSnapshotCleanupTask:
    """One-off cleanup task for pruning old snapshots on startup."""

    def __init__(self) -> None:
        self._position_manager = PositionManager()

    async def run_once(self) -> None:
        """Run a single cleanup pass based on retention settings."""
        try:
            deleted = await self._position_manager.cleanup_old_snapshots()
            logger.info(
                "position_snapshot_cleanup_run_once_completed",
                deleted=deleted,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.error(
                "position_snapshot_cleanup_run_once_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )


