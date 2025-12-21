"""
Model versions API endpoints.

Provides REST API endpoints for model version management:
- List model versions with filtering and pagination
- Get model version details with quality metrics
- Activate/deactivate model versions
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..database.repositories.quality_metrics_repo import ModelQualityMetricsRepository
from ..services.model_version_manager import model_version_manager
from ..config.logging import get_logger
from .middleware.security import validate_version_string

logger = get_logger(__name__)

router = APIRouter(tags=["models"])


class ModelVersionResponse(BaseModel):
    """Model version response model."""

    id: str
    version: str
    file_path: str
    model_type: str
    strategy_id: Optional[str] = None
    trained_at: str
    training_duration_seconds: Optional[int] = None
    training_dataset_size: Optional[int] = None
    training_config: Optional[dict] = None
    is_active: bool
    is_warmup_mode: bool
    created_at: str
    updated_at: str


class ModelVersionListResponse(BaseModel):
    """Model version list response with pagination."""

    items: List[ModelVersionResponse]
    total: int
    limit: Optional[int] = None
    offset: int


class ModelVersionDetailResponse(ModelVersionResponse):
    """Model version detail response with quality metrics."""

    quality_metrics: List[dict] = Field(default_factory=list)


class ModelActivationRequest(BaseModel):
    """Model activation request model."""

    strategy_id: Optional[str] = None


@router.get("/models", response_model=ModelVersionListResponse)
async def list_model_versions(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> ModelVersionListResponse:
    """
    List model versions with filtering and pagination.

    Args:
        strategy_id: Filter by trading strategy identifier
        is_active: Filter by active status
        limit: Maximum number of results (1-1000)
        offset: Number of results to skip

    Returns:
        Model version list with pagination metadata
    """
    try:
        repo = ModelVersionRepository()

        # Build query conditions
        conditions = []
        params = []
        param_index = 1

        if strategy_id is not None:
            conditions.append(f"strategy_id = ${param_index}")
            params.append(strategy_id)
            param_index += 1

        if is_active is not None:
            conditions.append(f"is_active = ${param_index}")
            params.append(is_active)
            param_index += 1

        # Get total count
        count_query = f"SELECT COUNT(*) FROM model_versions"
        if conditions:
            count_query += f" WHERE {' AND '.join(conditions)}"
        count_record = await repo._fetchrow(count_query, *params)
        total = count_record[0] if count_record else 0

        # Get paginated results
        query = "SELECT * FROM model_versions"
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        query += " ORDER BY trained_at DESC"
        query += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
        params.extend([limit, offset])

        records = await repo._fetch(query, *params)
        items = [ModelVersionResponse(**repo._record_to_dict(record)) for record in records]

        logger.info(
            "Listed model versions",
            strategy_id=strategy_id,
            is_active=is_active,
            count=len(items),
            total=total,
        )

        return ModelVersionListResponse(items=items, total=total, limit=limit, offset=offset)
    except Exception as e:
        logger.error("Failed to list model versions", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/models/{version}", response_model=ModelVersionDetailResponse)
async def get_model_version_details(version: str) -> ModelVersionDetailResponse:
    """
    Get model version details with quality metrics.

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')

    Returns:
        Model version details with quality metrics

    Raises:
        HTTPException: If model version not found
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        repo = ModelVersionRepository()
        model_version = await repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        # Get quality metrics
        metrics_repo = ModelQualityMetricsRepository()
        quality_metrics = await metrics_repo.get_by_model_version(UUID(model_version["id"]))

        logger.info("Retrieved model version details", version=version, metrics_count=len(quality_metrics))

        return ModelVersionDetailResponse(
            **model_version,
            quality_metrics=quality_metrics,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model version details", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/models/{version}/activate", response_model=ModelVersionResponse)
async def activate_model_version(
    version: str,
    request: Optional[ModelActivationRequest] = None,
) -> ModelVersionResponse:
    """
    Activate a model version (deactivates previous active model for the strategy).

    Args:
        version: Model version identifier (e.g., 'v1', 'v2.1')
        request: Optional activation request with strategy_id

    Returns:
        Activated model version record

    Raises:
        HTTPException: If model version not found or activation fails
    """
    # Validate version string to prevent path traversal
    if not validate_version_string(version):
        logger.warning("Invalid version string detected", version=version)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid version format")
    
    try:
        repo = ModelVersionRepository()
        model_version = await repo.get_by_version(version)

        if not model_version:
            logger.warning("Model version not found for activation", version=version)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model version {version} not found")

        strategy_id = request.strategy_id if request else model_version.get("strategy_id")
        activated_version = await model_version_manager.activate_version(UUID(model_version["id"]), strategy_id)

        if not activated_version:
            logger.error("Failed to activate model version", version=version)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to activate model version {version}",
            )

        logger.info("Model version activated", version=version, strategy_id=strategy_id)

        return ModelVersionResponse(**activated_version)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to activate model version", version=version, error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

