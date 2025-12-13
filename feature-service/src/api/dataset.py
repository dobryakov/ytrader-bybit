"""
Dataset API endpoints.
"""
from fastapi import APIRouter, HTTPException, Query, Path, Depends, Body
from fastapi.responses import FileResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from pathlib import Path as PathLib
import structlog
import shutil
import asyncio

from src.models.dataset import (
    Dataset,
    DatasetStatus,
    SplitStrategy,
    TargetConfig,
    WalkForwardConfig,
)
from src.api.middleware.auth import verify_api_key
from src.storage.metadata_storage import MetadataStorage
from src.services.dataset_builder import DatasetBuilder
from src.storage.parquet_storage import ParquetStorage
from src.config import config

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dataset", tags=["Datasets"])

# Global instances (will be set by main.py)
_metadata_storage: Optional[MetadataStorage] = None
_dataset_builder: Optional[DatasetBuilder] = None


def set_metadata_storage(storage: MetadataStorage):
    """Set metadata storage instance."""
    global _metadata_storage
    _metadata_storage = storage


def set_dataset_builder(builder: DatasetBuilder):
    """Set dataset builder instance."""
    global _dataset_builder
    _dataset_builder = builder


from pydantic import BaseModel


class DatasetBuildRequest(BaseModel):
    """Request model for dataset build."""
    symbol: str
    split_strategy: SplitStrategy
    train_period_start: Optional[datetime] = None
    train_period_end: Optional[datetime] = None
    validation_period_start: Optional[datetime] = None
    validation_period_end: Optional[datetime] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None
    walk_forward_config: Optional[Dict[str, Any]] = None
    target_config: TargetConfig
    feature_registry_version: str
    output_format: str = "parquet"


class DatasetResplitRequest(BaseModel):
    """Request model for dataset resplitting."""
    train_period_start: Optional[datetime] = None
    train_period_end: Optional[datetime] = None
    validation_period_start: Optional[datetime] = None
    validation_period_end: Optional[datetime] = None
    test_period_start: Optional[datetime] = None
    test_period_end: Optional[datetime] = None


@router.post("/build", status_code=202)
async def build_dataset(
    request: DatasetBuildRequest,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Build a dataset from historical data.
    
    Returns:
        Dataset ID and estimated completion time
    """
    if _dataset_builder is None:
        raise HTTPException(status_code=503, detail="Dataset builder not initialized")
    
    try:
        # Log received request to debug validation periods
        logger.info(
            "Dataset build request received",
            symbol=request.symbol,
            split_strategy=request.split_strategy.value,
            train_period_start=request.train_period_start.isoformat() if request.train_period_start else None,
            train_period_end=request.train_period_end.isoformat() if request.train_period_end else None,
            validation_period_start=request.validation_period_start.isoformat() if request.validation_period_start else None,
            validation_period_end=request.validation_period_end.isoformat() if request.validation_period_end else None,
            test_period_start=request.test_period_start.isoformat() if request.test_period_start else None,
            test_period_end=request.test_period_end.isoformat() if request.test_period_end else None,
        )
        
        dataset_id = await _dataset_builder.build_dataset(
            symbol=request.symbol,
            split_strategy=request.split_strategy,
            target_config=request.target_config,
            train_period_start=request.train_period_start,
            train_period_end=request.train_period_end,
            validation_period_start=request.validation_period_start,
            validation_period_end=request.validation_period_end,
            test_period_start=request.test_period_start,
            test_period_end=request.test_period_end,
            walk_forward_config=request.walk_forward_config.model_dump() if request.walk_forward_config else None,
            output_format=request.output_format,
            feature_registry_version=request.feature_registry_version,
        )
        
        # Get estimated completion
        progress = await _dataset_builder.get_build_progress(dataset_id)
        estimated_completion = progress.get("estimated_completion") if progress else None
        
        logger.info(
            "dataset_build_requested",
            dataset_id=dataset_id,
            symbol=request.symbol,
            split_strategy=request.split_strategy.value,
        )
        
        return {
            "dataset_id": dataset_id,
            "status": "building",
            "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("dataset_build_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/list")
async def list_datasets(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    status: Optional[DatasetStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    api_key: str = Depends(verify_api_key),
) -> List[dict]:
    """
    List datasets with optional filters.
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    datasets = await _metadata_storage.list_datasets(
        symbol=symbol,
        status=status.value if status else None,
        limit=limit,
    )
    
    return datasets


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: UUID = Path(..., description="Dataset ID"),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Get dataset by ID.
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    dataset_id_str = str(dataset_id)
    
    # Get dataset with timeout to prevent hanging
    try:
        dataset = await asyncio.wait_for(
            _metadata_storage.get_dataset(dataset_id_str),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.error(
            "get_dataset_timeout",
            dataset_id=dataset_id_str,
            message="Timeout getting dataset from database",
        )
        raise HTTPException(status_code=504, detail="Timeout getting dataset from database")
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Ensure target_config is a dict, not a JSON string
    if isinstance(dataset.get("target_config"), str):
        import json
        try:
            dataset["target_config"] = json.loads(dataset["target_config"])
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, keep original value
            pass
    
    # Ensure walk_forward_config is a dict, not a JSON string
    if isinstance(dataset.get("walk_forward_config"), str):
        import json
        try:
            dataset["walk_forward_config"] = json.loads(dataset["walk_forward_config"])
        except (json.JSONDecodeError, TypeError):
            # If parsing fails, keep original value
            pass
    
    return dataset


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: UUID = Path(..., description="Dataset ID"),
    split: str = Query("train", description="Split to download (train, validation, test)"),
    api_key: str = Depends(verify_api_key),
) -> FileResponse:
    """
    Download dataset split file.
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    dataset = await _metadata_storage.get_dataset(str(dataset_id))
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset["status"] != DatasetStatus.READY.value:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset is not ready (status: {dataset['status']})"
        )
    
    if split not in ["train", "validation", "test"]:
        raise HTTPException(status_code=400, detail="Invalid split name")
    
    storage_path = dataset.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    
    output_format = dataset.get("output_format", "parquet")
    file_path = PathLib(storage_path) / f"{split}.{output_format}"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Split file not found: {split}")
    
    media_type = {
        "parquet": "application/octet-stream",
        "csv": "text/csv",
        "hdf5": "application/octet-stream",
    }.get(output_format, "application/octet-stream")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=f"{dataset_id}_{split}.{output_format}",
    )


@router.post("/{dataset_id}/resplit", status_code=200)
async def resplit_dataset(
    dataset_id: UUID = Path(..., description="Dataset ID"),
    request: DatasetResplitRequest = Body(...),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Resplit an existing ready dataset with new time periods.
    
    Loads all existing splits (train, validation, test), merges them,
    and applies new time-based splitting. Only works for time_based split strategy.
    
    Returns:
        Updated dataset information with new split counts
    """
    if _dataset_builder is None:
        raise HTTPException(status_code=503, detail="Dataset builder not initialized")
    
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    try:
        await _dataset_builder.resplit_dataset(
            dataset_id=str(dataset_id),
            train_period_start=request.train_period_start,
            train_period_end=request.train_period_end,
            validation_period_start=request.validation_period_start,
            validation_period_end=request.validation_period_end,
            test_period_start=request.test_period_start,
            test_period_end=request.test_period_end,
        )
        
        # Get updated dataset
        dataset = await _metadata_storage.get_dataset(str(dataset_id))
        
        logger.info(
            "dataset_resplit_requested",
            dataset_id=str(dataset_id),
            train_records=dataset.get("train_records", 0),
            validation_records=dataset.get("validation_records", 0),
            test_records=dataset.get("test_records", 0),
        )
        
        return {
            "dataset_id": str(dataset_id),
            "status": "ready",
            "train_records": dataset.get("train_records", 0),
            "validation_records": dataset.get("validation_records", 0),
            "test_records": dataset.get("test_records", 0),
            "message": "Dataset resplit completed successfully",
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("dataset_resplit_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: UUID = Path(..., description="Dataset ID"),
    api_key: str = Depends(verify_api_key),
) -> None:
    """
    Delete dataset by ID.
    
    This will:
    1. Cancel active build task if exists
    2. Delete the dataset record from the database
    3. Delete all dataset files from storage (if they exist)
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    dataset_id_str = str(dataset_id)
    
    logger.info("delete_dataset_requested", dataset_id=dataset_id_str)
    
    # Cancel active build task if exists (non-blocking)
    if _dataset_builder is not None:
        try:
            active_builds = getattr(_dataset_builder, '_active_builds', {})
            if dataset_id_str in active_builds:
                build_task = active_builds[dataset_id_str]
                logger.info(
                    "cancelling_active_build",
                    dataset_id=dataset_id_str,
                )
                build_task.cancel()
                # Don't wait for cancellation - just cancel and continue
                logger.info(
                    "active_build_cancellation_initiated",
                    dataset_id=dataset_id_str,
                )
        except Exception as e:
            logger.warning(
                "failed_to_cancel_build",
                dataset_id=dataset_id_str,
                error=str(e),
            )
    
    # Get dataset metadata first to get storage_path (with timeout)
    try:
        dataset = await asyncio.wait_for(
            _metadata_storage.get_dataset(dataset_id_str),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.error(
            "get_dataset_timeout",
            dataset_id=dataset_id_str,
            message="Timeout getting dataset metadata",
        )
        raise HTTPException(status_code=504, detail="Timeout getting dataset metadata")
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    storage_path = dataset.get("storage_path")
    
    # Delete dataset record from database (with timeout)
    try:
        deleted = await asyncio.wait_for(
            _metadata_storage.delete_dataset(dataset_id_str),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        logger.error(
            "delete_dataset_db_timeout",
            dataset_id=dataset_id_str,
            message="Timeout deleting dataset from database",
        )
        raise HTTPException(status_code=504, detail="Timeout deleting dataset from database")
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    logger.info("dataset_deleted_from_db", dataset_id=dataset_id_str)
    
    # Delete dataset files asynchronously (fire-and-forget) to avoid blocking
    if storage_path:
        async def delete_files_async():
            """Delete files asynchronously without blocking the response."""
            try:
                dataset_dir = PathLib(storage_path)
                if dataset_dir.exists() and dataset_dir.is_dir():
                    # Use asyncio.to_thread with timeout
                    await asyncio.wait_for(
                        asyncio.to_thread(shutil.rmtree, dataset_dir),
                        timeout=30.0,
                    )
                    logger.info(
                        "dataset_files_deleted",
                        dataset_id=dataset_id_str,
                        storage_path=str(storage_path),
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "dataset_files_deletion_timeout",
                    dataset_id=dataset_id_str,
                    storage_path=str(storage_path),
                    message="File deletion timed out after 30 seconds",
                )
            except Exception as e:
                logger.warning(
                    "dataset_files_deletion_failed",
                    dataset_id=dataset_id_str,
                    storage_path=str(storage_path),
                    error=str(e),
                )
        
        # Start file deletion in background (don't wait for it)
        asyncio.create_task(delete_files_async())
    
    logger.info("dataset_deleted", dataset_id=dataset_id_str)


class ModelEvaluateRequest(BaseModel):
    """Request model for model evaluation."""
    dataset_id: UUID
    model_predictions: List[dict]


@router.post("/model/evaluate")
async def evaluate_model(
    request: ModelEvaluateRequest = Body(...),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Evaluate model predictions against dataset targets.
    
    This is a placeholder endpoint - full implementation would:
    1. Load dataset targets
    2. Compare with model predictions
    3. Compute metrics (accuracy, MSE, etc.)
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not initialized")
    
    dataset = await _metadata_storage.get_dataset(str(request.dataset_id))
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if dataset["status"] != DatasetStatus.READY.value:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset is not ready (status: {dataset['status']})"
        )
    
    # Placeholder implementation
    # In production, would:
    # 1. Load targets from dataset
    # 2. Match predictions with targets by timestamp
    # 3. Compute metrics based on target type
    
    target_config = dataset.get("target_config", {})
    target_type = target_config.get("type", "regression")
    
    if target_type == "regression":
        # Compute MSE, MAE, etc.
        metrics = {
            "mse": 0.0,  # Placeholder
            "mae": 0.0,  # Placeholder
            "rmse": 0.0,  # Placeholder
        }
    elif target_type == "classification":
        # Compute accuracy, precision, recall, etc.
        metrics = {
            "accuracy": 0.0,  # Placeholder
            "precision": 0.0,  # Placeholder
            "recall": 0.0,  # Placeholder
        }
    else:
        metrics = {}
    
    return {
        "dataset_id": str(request.dataset_id),
        "metrics": metrics,
        "predictions_count": len(request.model_predictions),
    }
