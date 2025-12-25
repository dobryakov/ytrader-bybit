"""Background task for periodic position validation."""

from __future__ import annotations

import asyncio
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..services.position_manager import PositionManager
from ..services.ws_gateway_client import WSGatewayClient
from ..utils.tracing import generate_trace_id, set_trace_id


logger = get_logger(__name__)


class PositionValidationTask:
    """Background task that periodically validates all positions.

    This is extracted from Order Manager's PositionValidationTask, adapted
    to the Position Manager architecture and configuration.
    """

    def __init__(self) -> None:
        """Initialize validation task."""
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager = PositionManager()
        self._ws_gateway_client = WSGatewayClient()

    async def start(self) -> None:
        """Start the validation loop."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._validation_loop())
        logger.info(
            "position_validation_task_started",
            interval=settings.position_manager_validation_interval,
        )

    async def stop(self) -> None:
        """Stop the validation loop."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("position_validation_task_stopped")

    async def _validation_loop(self) -> None:
        """Continuously validate positions at regular intervals."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.position_manager_validation_interval)

                if not self._should_run:
                    break

                # Get all positions
                positions = await self._position_manager.get_all_positions()

                # Validate all positions
                validated_count = 0
                fixed_count = 0
                error_count = 0

                for position in positions:
                    try:
                        is_valid, error_msg, updated_position = (
                            await self._position_manager.validate_position(
                                position.asset,
                                position.mode,
                                fix_discrepancies=True,
                                trace_id=trace_id,
                            )
                        )

                        if is_valid:
                            validated_count += 1
                            if updated_position is not None:
                                fixed_count += 1
                                logger.info(
                                    "position_discrepancy_fixed",
                                    asset=position.asset,
                                    mode=position.mode,
                                    trace_id=trace_id,
                                )
                        else:
                            error_count += 1
                            logger.warning(
                                "position_validation_failed",
                                asset=position.asset,
                                mode=position.mode,
                                error=error_msg,
                                trace_id=trace_id,
                            )

                    except Exception as e:
                        error_count += 1
                        logger.error(
                            "position_validation_error",
                            asset=position.asset,
                            mode=position.mode,
                            error=str(e),
                            trace_id=trace_id,
                        )

                logger.info(
                    "position_validation_completed",
                    validated_count=validated_count,
                    fixed_count=fixed_count,
                    error_count=error_count,
                    total_positions=len(positions),
                    trace_id=trace_id,
                )

                # Update validation statistics
                await self._position_manager.record_validation_statistics(
                    validated_count=validated_count,
                    fixed_count=fixed_count,
                    error_count=error_count,
                    total_positions=len(positions),
                )

                # Refresh position subscription to prevent deactivation
                # This indicates that position-manager is actively processing positions
                # even if no new WebSocket events are received
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
                    "position_validation_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs
                await asyncio.sleep(60)  # Wait before retrying

