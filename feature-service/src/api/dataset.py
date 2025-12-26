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

from ..models.dataset import (
    Dataset,
    DatasetStatus,
    SplitStrategy,
    TargetConfig,
    WalkForwardConfig,
)
from .middleware.auth import verify_api_key
from ..storage.metadata_storage import MetadataStorage
from ..services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder
from ..services.target_registry_version_manager import TargetRegistryVersionManager
from ..storage.parquet_storage import ParquetStorage
from ..config import config
from .middleware.security import validate_path_safe, is_path_traversal_attempt

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dataset", tags=["Datasets"])

# Global instances (will be set by main.py)
_metadata_storage: Optional[MetadataStorage] = None
_dataset_builder: Optional[OptimizedDatasetBuilder] = None
_target_registry_version_manager: Optional[TargetRegistryVersionManager] = None


def set_metadata_storage(storage: MetadataStorage):
    """Set metadata storage instance."""
    global _metadata_storage
    _metadata_storage = storage


def set_dataset_builder(builder: OptimizedDatasetBuilder):
    """Set dataset builder instance."""
    global _dataset_builder
    _dataset_builder = builder


def set_target_registry_version_manager(manager: TargetRegistryVersionManager):
    """Set target registry version manager instance."""
    global _target_registry_version_manager
    _target_registry_version_manager = manager


from pydantic import BaseModel, Field


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
    target_registry_version: str = Field(description="Target Registry version")
    feature_registry_version: str
    output_format: str = "parquet"
    strategy_id: Optional[str] = Field(default=None, description="Trading strategy identifier (optional)")


class ComputeFeaturesRequest(BaseModel):
    """Request model for computing features and target at a specific timestamp."""
    symbol: str
    timestamp: datetime = Field(description="Timestamp to compute features and target for")
    target_registry_version: str = Field(description="Target Registry version")
    feature_registry_version: str
    strategy_id: Optional[str] = Field(default=None, description="Trading strategy identifier (optional)")


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
            strategy_id=request.strategy_id,
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
            target_registry_version=request.target_registry_version,
            train_period_start=request.train_period_start,
            train_period_end=request.train_period_end,
            validation_period_start=request.validation_period_start,
            validation_period_end=request.validation_period_end,
            test_period_start=request.test_period_start,
            test_period_end=request.test_period_end,
            walk_forward_config=request.walk_forward_config.model_dump() if request.walk_forward_config else None,
            output_format=request.output_format,
            feature_registry_version=request.feature_registry_version,
            strategy_id=request.strategy_id,
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


@router.post("/compute-features", status_code=200)
async def compute_features_at_timestamp(
    request: ComputeFeaturesRequest,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Compute features and target for a specific timestamp.
    
    Uses the same parameters as dataset build, but computes features and target
    for a single timestamp instead of building a full dataset.
    
    Returns:
        Dictionary with features and target values for the specified timestamp
    """
    if _dataset_builder is None:
        raise HTTPException(status_code=503, detail="Dataset builder not initialized")
    
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target registry version manager not initialized")
    
    try:
        from datetime import timezone, timedelta
        from src.services.optimized_dataset.streaming_builder import StreamingDatasetBuilder
        from src.services.optimized_dataset.requirements_analyzer import FeatureRequirementsAnalyzer
        from src.services.optimized_dataset.rolling_window import OptimizedRollingWindow
        from src.services.optimized_dataset.hybrid_feature_computer import HybridFeatureComputer
        from src.services.optimized_dataset.incremental_orderbook import IncrementalOrderbookManager
        from src.services.target_computation import TargetComputationEngine, TargetComputationPresets
        from src.models.orderbook_state import OrderbookState
        import pandas as pd
        
        logger.info(
            "compute_features_request_received",
            symbol=request.symbol,
            timestamp=request.timestamp.isoformat(),
            feature_registry_version=request.feature_registry_version,
            target_registry_version=request.target_registry_version,
            strategy_id=request.strategy_id,
        )
        
        # Ensure timestamp is timezone-aware UTC
        timestamp = request.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # Load Feature Registry
        feature_registry_loader = _dataset_builder._streaming_builder.feature_registry_loader
        if feature_registry_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Feature Registry loader not initialized"
            )
        
        # Load Feature Registry config for the specified version
        # Try to load from version manager if available
        feature_registry = None
        if feature_registry_loader._version_manager:
            try:
                # Get version record from DB
                version_record = await _metadata_storage.get_feature_registry_version(
                    request.feature_registry_version
                )
                if version_record:
                    # Load config from file
                    from pathlib import Path
                    file_path = Path(version_record["file_path"])
                    if file_path.exists():
                        import yaml
                        with open(file_path, "r") as f:
                            config_data = yaml.safe_load(f)
                        if config_data:
                            from src.models.feature_registry import FeatureRegistry
                            feature_registry = FeatureRegistry(**config_data)
            except Exception as e:
                logger.warning(
                    "failed_to_load_feature_registry_from_db",
                    version=request.feature_registry_version,
                    error=str(e),
                )
        
        # Fallback: try to load from file directly
        if feature_registry is None:
            try:
                from src.config import config as app_config
                from pathlib import Path
                import yaml
                from src.models.feature_registry import FeatureRegistry
                
                versions_dir = Path(app_config.feature_registry_versions_dir)
                file_path = versions_dir / f"feature_registry_v{request.feature_registry_version}.yaml"
                
                if file_path.exists():
                    with open(file_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    if config_data:
                        feature_registry = FeatureRegistry(**config_data)
            except Exception as e:
                logger.warning(
                    "failed_to_load_feature_registry_from_file",
                    version=request.feature_registry_version,
                    error=str(e),
                )
        
        if feature_registry is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry version {request.feature_registry_version} not found"
            )
        
        # Load Target Registry
        target_config_dict = await _target_registry_version_manager.get_version(request.target_registry_version)
        if target_config_dict is None:
            raise HTTPException(
                status_code=404,
                detail=f"Target Registry version {request.target_registry_version} not found"
            )
        
        # Convert to TargetConfig
        from src.models.dataset import TargetConfig
        target_config = TargetConfig(**target_config_dict)
        
        # Analyze requirements
        requirements_analyzer = FeatureRequirementsAnalyzer()
        requirements = requirements_analyzer.analyze(feature_registry)
        
        # Calculate target horizon
        target_horizon_seconds = target_config.horizon if target_config else 0
        target_horizon_minutes = (target_horizon_seconds + 59) // 60  # Round up to minutes
        
        # Initialize rolling window
        rolling_window = OptimizedRollingWindow(
            max_lookback_minutes=requirements.max_lookback_minutes,
            symbol=request.symbol,
        )
        
        # Load data for the timestamp (need lookback window)
        start_date = timestamp.date() - timedelta(days=1)  # Load previous day for lookback
        end_date = timestamp.date() + timedelta(days=1)  # Load next day for target
        
        # Load klines
        klines_df = await _dataset_builder._parquet_storage.read_klines_range(
            request.symbol, start_date, end_date
        )
        
        if klines_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No klines data found for {request.symbol} around {timestamp.isoformat()}"
            )
        
        # Filter klines to timestamp range (lookback + target horizon)
        lookback_start = timestamp - timedelta(minutes=requirements.max_lookback_minutes + 5)
        target_end = timestamp + timedelta(minutes=target_horizon_minutes + 5)
        
        klines_filtered = klines_df[
            (klines_df["timestamp"] >= lookback_start) &
            (klines_df["timestamp"] <= target_end)
        ].copy()
        
        if klines_filtered.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No klines data in required range for {timestamp.isoformat()}"
            )
        
        # Load trades if needed
        trades_df = pd.DataFrame()
        if requirements.needs_trades:
            # Load trades for the day
            date_str = timestamp.date().strftime("%Y-%m-%d")
            try:
                trades_df = await _dataset_builder._parquet_storage.read_trades(
                    request.symbol, date_str
                )
                if not trades_df.empty:
                    trades_filtered = trades_df[
                        (trades_df["timestamp"] >= lookback_start) &
                        (trades_df["timestamp"] <= timestamp)
                    ].copy()
                    trades_df = trades_filtered
            except FileNotFoundError:
                logger.warning("no_trades_data", symbol=request.symbol, date=date_str)
        
        # Add data to rolling window
        rolling_window.add_data(
            timestamp=timestamp,
            klines=klines_filtered,
            trades=trades_df if not trades_df.empty else None,
            skip_trim=True,  # Don't trim, we need all data
        )
        
        # Load orderbook if needed
        orderbook_states = {}
        if requirements.needs_orderbook:
            # Load orderbook snapshot for the day
            date_str = timestamp.date().strftime("%Y-%m-%d")
            try:
                snapshots = await _dataset_builder._parquet_storage.read_orderbook_snapshots(
                    request.symbol, date_str
                )
                if not snapshots.empty:
                    # Find closest snapshot before timestamp
                    snapshots_before = snapshots[snapshots["timestamp"] <= timestamp]
                    if not snapshots_before.empty:
                        closest_snapshot_row = snapshots_before.iloc[-1]
                        # Convert DataFrame row to dictionary
                        snapshot_dict = closest_snapshot_row.to_dict()
                        # Ensure required fields
                        if "symbol" not in snapshot_dict:
                            snapshot_dict["symbol"] = request.symbol
                        if "sequence" not in snapshot_dict:
                            snapshot_dict["sequence"] = 0
                        # Convert timestamp if needed
                        if isinstance(snapshot_dict.get("timestamp"), pd.Timestamp):
                            snapshot_dict["timestamp"] = snapshot_dict["timestamp"].to_pydatetime()
                        # Ensure bids and asks are lists
                        if "bids" not in snapshot_dict or not isinstance(snapshot_dict["bids"], list):
                            snapshot_dict["bids"] = []
                        if "asks" not in snapshot_dict or not isinstance(snapshot_dict["asks"], list):
                            snapshot_dict["asks"] = []
                        orderbook_state = OrderbookState.from_snapshot(snapshot_dict)
                        orderbook_states[timestamp] = orderbook_state
            except FileNotFoundError:
                logger.warning("no_orderbook_data", symbol=request.symbol, date=date_str)
        
        # Initialize feature computer
        feature_computer = HybridFeatureComputer(
            requirements=requirements,
            feature_registry_version=request.feature_registry_version,
            feature_registry=feature_registry,
            target_horizon_minutes=target_horizon_minutes,
        )
        
        # Compute features
        timestamps_series = pd.Series([timestamp])
        features_df = feature_computer.compute_features_batch(
            timestamps=timestamps_series,
            rolling_window=rolling_window,
            klines_df=klines_filtered,
            trades_df=trades_df,
            orderbook_states=orderbook_states if orderbook_states else None,
        )
        
        if features_df.empty:
            raise HTTPException(
                status_code=500,
                detail="Failed to compute features"
            )
        
        # Compute target
        # Merge features with prices for target computation
        price_for_merge = klines_filtered[["timestamp", "close"]].rename(
            columns={"close": "price"}
        )
        merged = features_df.merge(
            price_for_merge, on="timestamp", how="left"
        )
        
        logger.info(
            "merged_data_for_target",
            symbol=request.symbol,
            timestamp=timestamp.isoformat(),
            merged_rows=len(merged),
            merged_columns=list(merged.columns),
            has_price_column="price" in merged.columns,
            has_timestamp_column="timestamp" in merged.columns,
            price_values_count=merged["price"].notna().sum() if "price" in merged.columns else 0,
        )
        
        # Get computation config
        computation_config = TargetComputationPresets.get_computation_config(
            target_config.computation
        )
        
        # Compute target
        logger.info(
            "computing_target",
            symbol=request.symbol,
            timestamp=timestamp.isoformat(),
            target_horizon_seconds=target_config.horizon,
            target_type=target_config.type,
            klines_count=len(klines_filtered),
            klines_timestamp_range=(
                klines_filtered["timestamp"].min().isoformat() if not klines_filtered.empty else None,
                klines_filtered["timestamp"].max().isoformat() if not klines_filtered.empty else None,
            ),
        )
        
        if target_config.type == "regression":
            targets_df = TargetComputationEngine.compute_target(
                merged, target_config.horizon, computation_config, klines_filtered
            )
        elif target_config.type == "classification":
            targets_df = TargetComputationEngine.compute_target(
                merged, target_config.horizon, computation_config, klines_filtered
            )
            # Apply classification mapping if needed
            if not targets_df.empty and "target" in targets_df.columns:
                # Classification mapping logic would go here
                pass
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target type: {target_config.type}"
            )
        
        logger.info(
            "target_computation_result",
            symbol=request.symbol,
            timestamp=timestamp.isoformat(),
            targets_df_empty=targets_df.empty,
            targets_df_columns=list(targets_df.columns) if not targets_df.empty else [],
            has_target_column="target" in targets_df.columns if not targets_df.empty else False,
        )
        
        # Combine features and target
        result = features_df.copy()
        if not targets_df.empty and "target" in targets_df.columns:
            result = result.merge(
                targets_df[["timestamp", "target"]],
                on="timestamp",
                how="left"
            )
            logger.info(
                "target_merged",
                symbol=request.symbol,
                timestamp=timestamp.isoformat(),
                merged_target_value=result.iloc[0]["target"] if len(result) > 0 and "target" in result.columns else None,
            )
        
        # Convert to dict for response
        if len(result) == 0:
            raise HTTPException(
                status_code=500,
                detail="No result computed"
            )
        
        row = result.iloc[0]
        response_data = {
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "symbol": request.symbol,
            "features": {},
            "target": None,
        }
        
        # Extract features (exclude timestamp and target)
        for col in result.columns:
            if col not in ["timestamp", "target"]:
                value = row[col]
                # Convert numpy types to Python types
                if pd.isna(value):
                    response_data["features"][col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    response_data["features"][col] = value.isoformat()
                else:
                    response_data["features"][col] = float(value) if isinstance(value, (int, float)) else str(value)
        
        # Extract target
        if "target" in result.columns:
            target_value = row["target"]
            if pd.notna(target_value):
                response_data["target"] = float(target_value) if isinstance(target_value, (int, float)) else str(target_value)
        
        logger.info(
            "compute_features_completed",
            symbol=request.symbol,
            timestamp=timestamp.isoformat(),
            features_count=len(response_data["features"]),
            has_target=response_data["target"] is not None,
        )
        
        return response_data
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("compute_features_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
    
    # Load target_config from Target Registry if target_registry_version is present
    target_registry_version = dataset.get("target_registry_version")
    if target_registry_version and _target_registry_version_manager:
        try:
            target_config_dict = await _target_registry_version_manager.get_version(target_registry_version)
            if target_config_dict:
                # get_version returns the config dict from YAML file
                dataset["target_config"] = target_config_dict
        except Exception as e:
            logger.warning(
                "Failed to load target_config from Target Registry",
                dataset_id=dataset_id_str,
                target_registry_version=target_registry_version,
                error=str(e),
            )
            # Fallback: try to parse existing target_config if present
            if isinstance(dataset.get("target_config"), str):
                import json
                try:
                    dataset["target_config"] = json.loads(dataset["target_config"])
                except (json.JSONDecodeError, TypeError):
                    pass
    
    # Ensure target_config is a dict, not a JSON string (fallback for old datasets)
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
    
    # Validate storage_path for path traversal
    if is_path_traversal_attempt(storage_path):
        logger.warning(
            "Path traversal detected in storage_path",
            dataset_id=str(dataset_id),
            storage_path=storage_path,
        )
        raise HTTPException(status_code=400, detail="Invalid storage path")
    
    # Validate that storage_path is within allowed base directory
    base_path = PathLib(config.feature_service_dataset_storage_path)
    storage_path_obj = PathLib(storage_path)
    if not validate_path_safe(base_path, storage_path_obj):
        logger.warning(
            "Storage path outside allowed directory",
            dataset_id=str(dataset_id),
            storage_path=storage_path,
            base_path=str(base_path),
        )
        raise HTTPException(status_code=400, detail="Invalid storage path")
    
    output_format = dataset.get("output_format", "parquet")
    
    # Validate output_format
    if output_format not in ["parquet", "csv", "hdf5"]:
        raise HTTPException(status_code=400, detail="Invalid output format")
    
    # Validate split name for path traversal
    if is_path_traversal_attempt(split):
        logger.warning(
            "Path traversal detected in split name",
            dataset_id=str(dataset_id),
            split=split,
        )
        raise HTTPException(status_code=400, detail="Invalid split name")
    
    file_path = storage_path_obj / f"{split}.{output_format}"
    
    # Final validation: ensure file_path is still within base_path
    if not validate_path_safe(base_path, file_path):
        logger.warning(
            "File path outside allowed directory",
            dataset_id=str(dataset_id),
            file_path=str(file_path),
            base_path=str(base_path),
        )
        raise HTTPException(status_code=400, detail="Invalid file path")
    
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
