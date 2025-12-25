"""Background task for periodic position synchronization with Bybit."""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.position_manager import PositionManager
from ..services.ws_gateway_client import WSGatewayClient
from ..utils.tracing import generate_trace_id, set_trace_id


logger = get_logger(__name__)


class PositionBybitSyncTask:
    """Background task that periodically synchronizes positions with Bybit API.

    This task:
    - Fetches positions from Bybit API
    - Compares them with local positions
    - If discrepancies are found, performs forced sync
    - Logs all discrepancies and sync actions
    """

    def __init__(self) -> None:
        """Initialize sync task."""
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager = PositionManager()
        self._ws_gateway_client = WSGatewayClient()

    async def start(self) -> None:
        """Start the sync loop."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            "position_bybit_sync_task_started",
            interval=settings.position_manager_bybit_sync_interval,
        )

    async def stop(self) -> None:
        """Stop the sync loop."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("position_bybit_sync_task_stopped")

    async def _sync_loop(self) -> None:
        """Continuously sync positions with Bybit at regular intervals."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.position_manager_bybit_sync_interval)

                if not self._should_run:
                    break

                # Get sync report (without force first)
                logger.info("position_bybit_sync_check_started", trace_id=trace_id)
                report = await self._position_manager.sync_positions_with_bybit(
                    force=False,
                    trace_id=trace_id,
                )

                # Check for discrepancies
                discrepancies = [
                    comp
                    for comp in report.get("comparisons", [])
                    if not comp.get("size_match", True)
                    or (
                        comp.get("avg_price_diff_pct") is not None
                        and comp.get("avg_price_diff_pct", 0) > 1.0
                    )
                ]

                if discrepancies:
                    logger.warning(
                        "position_bybit_sync_discrepancies_found",
                        count=len(discrepancies),
                        bybit_count=report.get("bybit_positions_count", 0),
                        local_count=report.get("local_positions_count", 0),
                        trace_id=trace_id,
                    )

                    # Log each discrepancy
                    for comp in discrepancies:
                        logger.warning(
                            "position_bybit_sync_discrepancy",
                            asset=comp.get("asset"),
                            bybit_size=comp.get("bybit_size"),
                            local_size=comp.get("local_size"),
                            size_match=comp.get("size_match"),
                            size_diff=comp.get("size_diff"),
                            bybit_avg_price=comp.get("bybit_avg_price"),
                            local_avg_price=comp.get("local_avg_price"),
                            avg_price_diff_pct=comp.get("avg_price_diff_pct"),
                            bybit_exists=comp.get("bybit_exists"),
                            local_exists=comp.get("local_exists"),
                            trace_id=trace_id,
                        )

                    # Perform forced sync
                    logger.info(
                        "position_bybit_sync_force_sync_started",
                        discrepancies_count=len(discrepancies),
                        trace_id=trace_id,
                    )

                    force_report = await self._position_manager.sync_positions_with_bybit(
                        force=True,
                        trace_id=trace_id,
                    )

                    # Log sync results
                    logger.info(
                        "position_bybit_sync_force_sync_completed",
                        updated_count=len(force_report.get("updated", [])),
                        created_count=len(force_report.get("created", [])),
                        errors_count=len(force_report.get("errors", [])),
                        trace_id=trace_id,
                    )

                    # Log each updated/created position
                    for item in force_report.get("updated", []):
                        logger.info(
                            "position_bybit_sync_updated",
                            asset=item.get("asset"),
                            action=item.get("action"),
                            reason=item.get("reason"),
                            position_id=item.get("position_id"),
                            trace_id=trace_id,
                        )

                    for item in force_report.get("created", []):
                        logger.info(
                            "position_bybit_sync_created",
                            asset=item.get("asset"),
                            action=item.get("action"),
                            position_id=item.get("position_id"),
                            trace_id=trace_id,
                        )

                    # Log errors if any
                    for error in force_report.get("errors", []):
                        logger.error(
                            "position_bybit_sync_error",
                            asset=error.get("asset"),
                            error=error.get("error"),
                            trace_id=trace_id,
                        )

                else:
                    logger.info(
                        "position_bybit_sync_no_discrepancies",
                        bybit_count=report.get("bybit_positions_count", 0),
                        local_count=report.get("local_positions_count", 0),
                        trace_id=trace_id,
                    )

                # Refresh position subscription to prevent deactivation
                # This indicates that position-manager is actively syncing with Bybit
                # and confirms that the system is working even if no WebSocket events are received
                try:
                    await self._ws_gateway_client.refresh_position_subscription()
                except Exception as e:
                    # Non-fatal: log but continue
                    logger.warning(
                        "position_subscription_refresh_failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        trace_id=trace_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "position_bybit_sync_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs
                await asyncio.sleep(60)  # Wait before retrying

