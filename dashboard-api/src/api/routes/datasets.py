"""Dataset query endpoints - read from PostgreSQL database."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, Path, HTTPException, Request
from fastapi.responses import JSONResponse
import json

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/datasets")
async def list_datasets(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[str] = Query(None, description="Filter by status (building, ready, failed)"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results"),
):
    """List datasets with optional filters."""
    trace_id = get_or_create_trace_id()
    logger.info("dataset_list_request", symbol=symbol, status=status, limit=limit, trace_id=trace_id)

    try:
        query = """
            SELECT 
                id, symbol, status, split_strategy, strategy_id,
                train_period_start, train_period_end,
                validation_period_start, validation_period_end,
                test_period_start, test_period_end,
                walk_forward_config, target_config, split_statistics,
                feature_registry_version, target_registry_version,
                train_records, validation_records, test_records,
                output_format, storage_path,
                created_at, completed_at, estimated_completion,
                error_message
            FROM datasets
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if symbol:
            query += f" AND symbol = ${param_idx}"
            params.append(symbol)
            param_idx += 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        rows = await DatabaseConnection.fetch(query, *params)

        datasets_data = []
        for row in rows:
            # Parse JSONB fields if they are strings
            walk_forward_config = row["walk_forward_config"]
            if isinstance(walk_forward_config, str):
                try:
                    walk_forward_config = json.loads(walk_forward_config)
                except (json.JSONDecodeError, TypeError):
                    walk_forward_config = None

            target_config = row["target_config"]
            if target_config is None:
                target_config = None
            elif isinstance(target_config, str):
                try:
                    target_config = json.loads(target_config)
                except (json.JSONDecodeError, TypeError):
                    target_config = None
            # If it's already a dict/list, keep it as is

            split_statistics = row.get("split_statistics")
            if isinstance(split_statistics, str):
                try:
                    split_statistics = json.loads(split_statistics)
                except (json.JSONDecodeError, TypeError):
                    split_statistics = None
            # If it's already a dict, keep it as is

            dataset_dict = {
                "id": str(row["id"]),
                "symbol": row["symbol"],
                "status": row["status"],
                "split_strategy": row["split_strategy"],
                "strategy_id": row.get("strategy_id"),
                "train_period_start": row["train_period_start"].isoformat() + "Z" if row["train_period_start"] else None,
                "train_period_end": row["train_period_end"].isoformat() + "Z" if row["train_period_end"] else None,
                "validation_period_start": row["validation_period_start"].isoformat() + "Z" if row["validation_period_start"] else None,
                "validation_period_end": row["validation_period_end"].isoformat() + "Z" if row["validation_period_end"] else None,
                "test_period_start": row["test_period_start"].isoformat() + "Z" if row["test_period_start"] else None,
                "test_period_end": row["test_period_end"].isoformat() + "Z" if row["test_period_end"] else None,
                "walk_forward_config": walk_forward_config,
                "target_config": target_config,
                "split_statistics": split_statistics,
                "feature_registry_version": row["feature_registry_version"],
                "target_registry_version": row.get("target_registry_version"),
                "train_records": row["train_records"] or 0,
                "validation_records": row["validation_records"] or 0,
                "test_records": row["test_records"] or 0,
                "output_format": row["output_format"],
                "storage_path": row["storage_path"],
                "created_at": row["created_at"].isoformat() + "Z",
                "completed_at": row["completed_at"].isoformat() + "Z" if row["completed_at"] else None,
                "estimated_completion": row["estimated_completion"].isoformat() + "Z" if row["estimated_completion"] else None,
                "error_message": row["error_message"],
            }
            datasets_data.append(dataset_dict)

        logger.info("dataset_list_completed", count=len(datasets_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content=datasets_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("dataset_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve datasets: {str(e)}")


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: UUID = Path(..., description="Dataset ID"),
):
    """Get dataset by ID."""
    trace_id = get_or_create_trace_id()
    logger.info("dataset_get_request", dataset_id=str(dataset_id), trace_id=trace_id)

    try:
        query = """
            SELECT 
                id, symbol, status, split_strategy, strategy_id,
                train_period_start, train_period_end,
                validation_period_start, validation_period_end,
                test_period_start, test_period_end,
                walk_forward_config, target_config, split_statistics,
                feature_registry_version, target_registry_version,
                train_records, validation_records, test_records,
                output_format, storage_path,
                created_at, completed_at, estimated_completion,
                error_message
            FROM datasets
            WHERE id = $1
        """
        rows = await DatabaseConnection.fetch(query, dataset_id)

        if not rows:
            raise HTTPException(status_code=404, detail="Dataset not found")

        row = rows[0]

        # Parse JSONB fields if they are strings
        walk_forward_config = row["walk_forward_config"]
        if isinstance(walk_forward_config, str):
            try:
                walk_forward_config = json.loads(walk_forward_config)
            except (json.JSONDecodeError, TypeError):
                walk_forward_config = None

        target_config = row["target_config"]
        if target_config is None:
            target_config = None
        elif isinstance(target_config, str):
            try:
                target_config = json.loads(target_config)
            except (json.JSONDecodeError, TypeError):
                target_config = None
        # If it's already a dict/list, keep it as is

        split_statistics = row.get("split_statistics")
        if isinstance(split_statistics, str):
            try:
                split_statistics = json.loads(split_statistics)
            except (json.JSONDecodeError, TypeError):
                split_statistics = None
        # If it's already a dict, keep it as is

        dataset_dict = {
            "id": str(row["id"]),
            "symbol": row["symbol"],
            "status": row["status"],
            "split_strategy": row["split_strategy"],
            "strategy_id": row.get("strategy_id"),
            "train_period_start": row["train_period_start"].isoformat() + "Z" if row["train_period_start"] else None,
            "train_period_end": row["train_period_end"].isoformat() + "Z" if row["train_period_end"] else None,
            "validation_period_start": row["validation_period_start"].isoformat() + "Z" if row["validation_period_start"] else None,
            "validation_period_end": row["validation_period_end"].isoformat() + "Z" if row["validation_period_end"] else None,
            "test_period_start": row["test_period_start"].isoformat() + "Z" if row["test_period_start"] else None,
            "test_period_end": row["test_period_end"].isoformat() + "Z" if row["test_period_end"] else None,
            "walk_forward_config": walk_forward_config,
            "target_config": target_config,
            "split_statistics": split_statistics,
            "feature_registry_version": row["feature_registry_version"],
            "target_registry_version": row.get("target_registry_version"),
            "train_records": row["train_records"] or 0,
            "validation_records": row["validation_records"] or 0,
            "test_records": row["test_records"] or 0,
            "output_format": row["output_format"],
            "storage_path": row["storage_path"],
            "created_at": row["created_at"].isoformat() + "Z",
            "completed_at": row["completed_at"].isoformat() + "Z" if row["completed_at"] else None,
            "estimated_completion": row["estimated_completion"].isoformat() + "Z" if row["estimated_completion"] else None,
            "error_message": row["error_message"],
        }

        logger.info("dataset_get_completed", dataset_id=str(dataset_id), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content=dataset_dict,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("dataset_get_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {str(e)}")

