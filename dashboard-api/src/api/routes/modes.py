"""Mode management endpoints."""

from typing import Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Query, Path, HTTPException, Body
from fastapi.responses import JSONResponse
import httpx

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...config.settings import settings
from ...utils.tracing import get_or_create_trace_id
from ...models.mode import (
    ModeCreate,
    ModeUpdate,
    ModeResponse,
    RebuildDatasetRequest,
    RebuildDatasetResponse,
)
from ...utils.period_calculator import calculate_periods_from_durations

logger = get_logger(__name__)
router = APIRouter()


def get_feature_service_url(path: str) -> str:
    """Get full URL for Feature Service endpoint."""
    return f"http://{settings.feature_service_host}:{settings.feature_service_port}{path}"


def get_feature_service_headers() -> dict:
    """Get headers for Feature Service requests."""
    headers = {}
    api_key = settings.feature_service_api_key
    if api_key and api_key.strip():
        headers["X-API-Key"] = str(api_key).strip()
    return headers


@router.get("/modes", response_model=list[ModeResponse])
async def list_modes(
    asset: Optional[str] = Query(None, description="Filter by asset"),
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results"),
):
    """List modes with optional filters."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_list_request", asset=asset, strategy_id=strategy_id, is_active=is_active, limit=limit, trace_id=trace_id)

    try:
        query = """
            SELECT 
                id, name, asset, strategy_id,
                feature_registry_version, target_registry_version,
                train_duration_days, validation_duration_days, test_duration_days,
                description, is_active,
                created_at, updated_at, created_by
            FROM modes
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        if strategy_id:
            query += f" AND strategy_id = ${param_idx}"
            params.append(strategy_id)
            param_idx += 1

        if is_active is not None:
            query += f" AND is_active = ${param_idx}"
            params.append(is_active)
            param_idx += 1

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        rows = await DatabaseConnection.fetch(query, *params)

        modes = []
        for row in rows:
            modes.append(ModeResponse(
                id=row["id"],
                name=row["name"],
                asset=row["asset"],
                strategy_id=row["strategy_id"],
                feature_registry_version=row["feature_registry_version"],
                target_registry_version=row["target_registry_version"],
                train_duration_days=row["train_duration_days"],
                validation_duration_days=row["validation_duration_days"],
                test_duration_days=row["test_duration_days"],
                description=row["description"],
                is_active=row["is_active"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                created_by=row["created_by"],
            ))

        return modes
    except Exception as e:
        logger.error("mode_list_error", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list modes: {str(e)}")


@router.get("/modes/{mode_id}", response_model=ModeResponse)
async def get_mode(mode_id: UUID = Path(..., description="Mode ID")):
    """Get mode by ID."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_get_request", mode_id=str(mode_id), trace_id=trace_id)

    try:
        query = """
            SELECT 
                id, name, asset, strategy_id,
                feature_registry_version, target_registry_version,
                train_duration_days, validation_duration_days, test_duration_days,
                description, is_active,
                created_at, updated_at, created_by
            FROM modes
            WHERE id = $1
        """
        row = await DatabaseConnection.fetchrow(query, mode_id)

        if not row:
            raise HTTPException(status_code=404, detail="Mode not found")

        return ModeResponse(
            id=row["id"],
            name=row["name"],
            asset=row["asset"],
            strategy_id=row["strategy_id"],
            feature_registry_version=row["feature_registry_version"],
            target_registry_version=row["target_registry_version"],
            train_duration_days=row["train_duration_days"],
            validation_duration_days=row["validation_duration_days"],
            test_duration_days=row["test_duration_days"],
            description=row["description"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mode_get_error", mode_id=str(mode_id), error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get mode: {str(e)}")


@router.post("/modes", response_model=ModeResponse, status_code=201)
async def create_mode(mode: ModeCreate):
    """Create a new mode."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_create_request", mode_name=mode.name, asset=mode.asset, trace_id=trace_id)

    try:
        query = """
            INSERT INTO modes (
                name, asset, strategy_id,
                feature_registry_version, target_registry_version,
                train_duration_days, validation_duration_days, test_duration_days,
                description
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING 
                id, name, asset, strategy_id,
                feature_registry_version, target_registry_version,
                train_duration_days, validation_duration_days, test_duration_days,
                description, is_active,
                created_at, updated_at, created_by
        """
        row = await DatabaseConnection.fetchrow(
            query,
            mode.name,
            mode.asset,
            mode.strategy_id,
            mode.feature_registry_version,
            mode.target_registry_version,
            mode.train_duration_days,
            mode.validation_duration_days,
            mode.test_duration_days,
            mode.description,
        )

        return ModeResponse(
            id=row["id"],
            name=row["name"],
            asset=row["asset"],
            strategy_id=row["strategy_id"],
            feature_registry_version=row["feature_registry_version"],
            target_registry_version=row["target_registry_version"],
            train_duration_days=row["train_duration_days"],
            validation_duration_days=row["validation_duration_days"],
            test_duration_days=row["test_duration_days"],
            description=row["description"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )
    except Exception as e:
        error_msg = str(e)
        if "unique constraint" in error_msg.lower() or "duplicate key" in error_msg.lower():
            logger.warning("mode_create_duplicate", mode_name=mode.name, trace_id=trace_id)
            raise HTTPException(status_code=409, detail=f"Mode with name '{mode.name}' already exists")
        logger.error("mode_create_error", error=error_msg, trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create mode: {error_msg}")


@router.put("/modes/{mode_id}", response_model=ModeResponse)
async def update_mode(mode_id: UUID = Path(..., description="Mode ID"), mode_update: ModeUpdate = Body(...)):
    """Update a mode."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_update_request", mode_id=str(mode_id), trace_id=trace_id)

    try:
        # Build update query dynamically based on provided fields
        update_fields = []
        params = []
        param_idx = 1

        if mode_update.name is not None:
            update_fields.append(f"name = ${param_idx}")
            params.append(mode_update.name)
            param_idx += 1

        if mode_update.asset is not None:
            update_fields.append(f"asset = ${param_idx}")
            params.append(mode_update.asset)
            param_idx += 1

        if mode_update.strategy_id is not None:
            update_fields.append(f"strategy_id = ${param_idx}")
            params.append(mode_update.strategy_id)
            param_idx += 1

        if mode_update.feature_registry_version is not None:
            update_fields.append(f"feature_registry_version = ${param_idx}")
            params.append(mode_update.feature_registry_version)
            param_idx += 1

        if mode_update.target_registry_version is not None:
            update_fields.append(f"target_registry_version = ${param_idx}")
            params.append(mode_update.target_registry_version)
            param_idx += 1

        if mode_update.train_duration_days is not None:
            update_fields.append(f"train_duration_days = ${param_idx}")
            params.append(mode_update.train_duration_days)
            param_idx += 1

        if mode_update.validation_duration_days is not None:
            update_fields.append(f"validation_duration_days = ${param_idx}")
            params.append(mode_update.validation_duration_days)
            param_idx += 1

        if mode_update.test_duration_days is not None:
            update_fields.append(f"test_duration_days = ${param_idx}")
            params.append(mode_update.test_duration_days)
            param_idx += 1

        if mode_update.description is not None:
            update_fields.append(f"description = ${param_idx}")
            params.append(mode_update.description)
            param_idx += 1

        if mode_update.is_active is not None:
            update_fields.append(f"is_active = ${param_idx}")
            params.append(mode_update.is_active)
            param_idx += 1

        if not update_fields:
            # No fields to update, return existing mode
            return await get_mode(mode_id)

        # Add updated_at
        update_fields.append(f"updated_at = NOW()")

        # Add mode_id as last parameter
        params.append(mode_id)

        query = f"""
            UPDATE modes
            SET {', '.join(update_fields)}
            WHERE id = ${param_idx}
            RETURNING 
                id, name, asset, strategy_id,
                feature_registry_version, target_registry_version,
                train_duration_days, validation_duration_days, test_duration_days,
                description, is_active,
                created_at, updated_at, created_by
        """

        row = await DatabaseConnection.fetchrow(query, *params)

        if not row:
            raise HTTPException(status_code=404, detail="Mode not found")

        return ModeResponse(
            id=row["id"],
            name=row["name"],
            asset=row["asset"],
            strategy_id=row["strategy_id"],
            feature_registry_version=row["feature_registry_version"],
            target_registry_version=row["target_registry_version"],
            train_duration_days=row["train_duration_days"],
            validation_duration_days=row["validation_duration_days"],
            test_duration_days=row["test_duration_days"],
            description=row["description"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "unique constraint" in error_msg.lower() or "duplicate key" in error_msg.lower():
            logger.warning("mode_update_duplicate", mode_id=str(mode_id), trace_id=trace_id)
            raise HTTPException(status_code=409, detail="Mode with this name already exists")
        logger.error("mode_update_error", mode_id=str(mode_id), error=error_msg, trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update mode: {error_msg}")


@router.delete("/modes/{mode_id}", status_code=204)
async def delete_mode(mode_id: UUID = Path(..., description="Mode ID")):
    """Delete a mode (soft delete by setting is_active=false)."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_delete_request", mode_id=str(mode_id), trace_id=trace_id)

    try:
        query = """
            UPDATE modes
            SET is_active = false, updated_at = NOW()
            WHERE id = $1
            RETURNING id
        """
        row = await DatabaseConnection.fetchrow(query, mode_id)

        if not row:
            raise HTTPException(status_code=404, detail="Mode not found")

        return JSONResponse(status_code=204, content=None)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mode_delete_error", mode_id=str(mode_id), error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete mode: {str(e)}")


@router.post("/modes/{mode_id}/rebuild-dataset", response_model=RebuildDatasetResponse)
async def rebuild_dataset(
    mode_id: UUID = Path(..., description="Mode ID"),
    request: Optional[RebuildDatasetRequest] = Body(None),
):
    """Rebuild dataset for a mode by calculating periods from durations."""
    trace_id = get_or_create_trace_id()
    logger.info("mode_rebuild_dataset_request", mode_id=str(mode_id), trace_id=trace_id)

    try:
        # Get mode
        mode = await get_mode(mode_id)

        # Calculate periods from durations
        reference_time = request.reference_time if request and request.reference_time else None
        periods = calculate_periods_from_durations(
            train_duration_days=mode.train_duration_days,
            validation_duration_days=mode.validation_duration_days,
            test_duration_days=mode.test_duration_days,
            reference_time=reference_time,
        )

        logger.info(
            "mode_rebuild_dataset_periods_calculated",
            mode_id=str(mode_id),
            periods=periods,
            trace_id=trace_id,
        )

        # Build dataset request for Feature Service
        dataset_request = {
            "symbol": mode.asset,
            "strategy_id": mode.strategy_id,
            "split_strategy": "time_based",
            "train_period_start": periods["train_period_start"].isoformat(),
            "train_period_end": periods["train_period_end"].isoformat(),
            "validation_period_start": periods["validation_period_start"].isoformat(),
            "validation_period_end": periods["validation_period_end"].isoformat(),
            "test_period_start": periods["test_period_start"].isoformat(),
            "test_period_end": periods["test_period_end"].isoformat(),
            "feature_registry_version": mode.feature_registry_version,
            "target_registry_version": mode.target_registry_version,
            "output_format": "parquet",
        }

        # Call Feature Service to build dataset
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                get_feature_service_url("/dataset/build"),
                headers=get_feature_service_headers(),
                json=dataset_request,
            )
            response.raise_for_status()
            dataset_response = response.json()

        dataset_id = dataset_response.get("dataset_id")
        if not dataset_id:
            raise HTTPException(status_code=500, detail="Feature Service did not return dataset_id")

        # Format periods for response (convert to ISO strings)
        computed_periods = {
            "train_period_start": periods["train_period_start"].isoformat(),
            "train_period_end": periods["train_period_end"].isoformat(),
            "validation_period_start": periods["validation_period_start"].isoformat(),
            "validation_period_end": periods["validation_period_end"].isoformat(),
            "test_period_start": periods["test_period_start"].isoformat(),
            "test_period_end": periods["test_period_end"].isoformat(),
        }

        return RebuildDatasetResponse(
            mode_id=mode_id,
            dataset_id=UUID(dataset_id),
            computed_periods=computed_periods,
            message=f"Dataset build requested successfully. Dataset ID: {dataset_id}",
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        logger.error(
            "mode_rebuild_dataset_feature_service_error",
            mode_id=str(mode_id),
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=f"Feature Service error: {e.response.text}")
    except Exception as e:
        logger.error("mode_rebuild_dataset_error", mode_id=str(mode_id), error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to rebuild dataset: {str(e)}")

