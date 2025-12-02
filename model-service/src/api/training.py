"""
Training API endpoints.

Provides REST API endpoints for training management:
- Get training status
- Manually trigger training
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..services.training_orchestrator import training_orchestrator
from ..config.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["training"])


class TrainingStatusResponse(BaseModel):
    """Training status response model."""

    is_training: bool
    current_training: Optional[dict] = None
    last_training: Optional[dict] = None
    next_scheduled_training: Optional[str] = None
    buffered_events_count: int = 0
    queue_size: int = 0
    queue_wait_time_seconds: Optional[float] = None
    next_queued_training_strategy_id: Optional[str] = None


class TrainingTriggerRequest(BaseModel):
    """Training trigger request model."""

    strategy_id: Optional[str] = Field(None, description="Trading strategy identifier")


class TrainingTriggerResponse(BaseModel):
    """Training trigger response model."""

    triggered: bool
    message: str
    strategy_id: Optional[str] = None


@router.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status() -> TrainingStatusResponse:
    """
    Get current training status.

    Returns:
        Training status with current, last, and next scheduled training info
    """
    try:
        status = training_orchestrator.get_status()

        logger.info(
            "Retrieved training status",
            is_training=status["is_training"],
            buffered_events_count=status["buffered_events_count"],
            queue_size=status["queue_size"],
        )

        return TrainingStatusResponse(
            is_training=status["is_training"],
            current_training=None,
            last_training=None,
            next_scheduled_training=None,
            buffered_events_count=status["buffered_events_count"],
            queue_size=status["queue_size"],
            queue_wait_time_seconds=status["queue_next_wait_time_seconds"],
            next_queued_training_strategy_id=status["next_queued_training_strategy_id"],
        )
    except Exception as e:
        logger.error("Failed to get training status", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/training/trigger", response_model=TrainingTriggerResponse)
async def trigger_training(request: Optional[TrainingTriggerRequest] = None) -> TrainingTriggerResponse:
    """
    Manually trigger training for a strategy.

    Args:
        request: Optional training trigger request with strategy_id

    Returns:
        Training trigger response

    Raises:
        HTTPException: If training trigger fails
    """
    try:
        strategy_id = request.strategy_id if request else None

        # Check if there are enough events to train
        if len(training_orchestrator._execution_events_buffer) == 0:
            logger.warning("No execution events available for training", strategy_id=strategy_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No execution events available for training. Wait for more events to accumulate.",
            )

        # Trigger training
        await training_orchestrator.check_and_trigger_training(strategy_id)

        logger.info("Training triggered manually", strategy_id=strategy_id)

        return TrainingTriggerResponse(
            triggered=True,
            message="Training triggered successfully",
            strategy_id=strategy_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to trigger training", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

