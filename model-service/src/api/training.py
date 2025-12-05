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
    queue_size: int = 0
    queue_wait_time_seconds: Optional[float] = None
    next_queued_training_strategy_id: Optional[str] = None
    next_queued_training_dataset_id: Optional[str] = None
    pending_dataset_builds: int = 0


class TrainingTriggerRequest(BaseModel):
    """Training trigger request model."""

    strategy_id: Optional[str] = Field(None, description="Trading strategy identifier")
    symbol: Optional[str] = Field(None, description="Trading pair symbol (e.g., 'BTCUSDT'). If not provided, uses default.")


class DatasetBuildRequest(BaseModel):
    """Dataset build request model."""

    strategy_id: Optional[str] = Field(None, description="Trading strategy identifier")
    symbol: Optional[str] = Field(None, description="Trading pair symbol (e.g., 'BTCUSDT'). If not provided, uses default.")


class DatasetBuildResponse(BaseModel):
    """Dataset build response model."""

    dataset_id: str
    message: str
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None


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
            queue_size=status["queue_size"],
            pending_dataset_builds=status["pending_dataset_builds"],
        )

        return TrainingStatusResponse(
            is_training=status["is_training"],
            current_training=None,
            last_training=None,
            next_scheduled_training=None,
            queue_size=status["queue_size"],
            queue_wait_time_seconds=status["queue_next_wait_time_seconds"],
            next_queued_training_strategy_id=status["next_queued_training_strategy_id"],
            next_queued_training_dataset_id=status.get("next_queued_training_dataset_id"),
            pending_dataset_builds=status["pending_dataset_builds"],
        )
    except Exception as e:
        logger.error("Failed to get training status", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/training/trigger", response_model=TrainingTriggerResponse)
async def trigger_training(request: Optional[TrainingTriggerRequest] = None) -> TrainingTriggerResponse:
    """
    Manually trigger training for a strategy.

    This will request dataset build from Feature Service and start training
    when dataset is ready. Training uses market data from Feature Service,
    not execution events.

    Args:
        request: Optional training trigger request with strategy_id and symbol

    Returns:
        Training trigger response

    Raises:
        HTTPException: If training trigger fails
    """
    try:
        strategy_id = request.strategy_id if request else None
        symbol = request.symbol if request else None

        # Trigger training (will request dataset build from Feature Service)
        await training_orchestrator.check_and_trigger_training(strategy_id=strategy_id, symbol=symbol)

        logger.info("Training triggered manually", strategy_id=strategy_id, symbol=symbol)

        return TrainingTriggerResponse(
            triggered=True,
            message="Training triggered successfully. Dataset build requested from Feature Service. Training will start when dataset is ready.",
            strategy_id=strategy_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to trigger training", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/training/dataset/build", response_model=DatasetBuildResponse)
async def request_dataset_build(request: Optional[DatasetBuildRequest] = None) -> DatasetBuildResponse:
    """
    Manually request dataset build from Feature Service.

    This endpoint allows you to explicitly request a dataset build without
    triggering training immediately. Training will start automatically when
    dataset.ready notification is received.

    Args:
        request: Optional dataset build request with strategy_id and symbol

    Returns:
        Dataset build response with dataset_id

    Raises:
        HTTPException: If dataset build request fails
    """
    try:
        strategy_id = request.strategy_id if request else None
        symbol = request.symbol if request else None

        # Request dataset build from Feature Service
        dataset_id = await training_orchestrator.request_dataset_build(strategy_id=strategy_id, symbol=symbol)

        if not dataset_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to request dataset build from Feature Service. Check Feature Service availability and logs.",
            )

        logger.info("Dataset build requested manually", dataset_id=str(dataset_id), strategy_id=strategy_id, symbol=symbol)

        return DatasetBuildResponse(
            dataset_id=str(dataset_id),
            message=f"Dataset build requested successfully. Dataset ID: {dataset_id}. Training will start automatically when dataset is ready.",
            strategy_id=strategy_id,
            symbol=symbol,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to request dataset build", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

