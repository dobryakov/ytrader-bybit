"""
Dataset Builder service for building training datasets from historical data.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from uuid import uuid4
from pathlib import Path
import pandas as pd
import structlog

from src.models.dataset import Dataset, DatasetStatus, SplitStrategy, TargetConfig
from src.models.feature_vector import FeatureVector
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.services.offline_engine import OfflineEngine
from src.services.feature_registry import FeatureRegistryLoader
from src.services.backfilling_service import BackfillingService
from src.config import config

if TYPE_CHECKING:
    from src.publishers.dataset_publisher import DatasetPublisher

logger = structlog.get_logger(__name__)


class DatasetBuilder:
    """Service for building training datasets from historical data."""
    
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        parquet_storage: ParquetStorage,
        dataset_storage_path: str,
        batch_size: int = 1000,
        feature_registry_version: str = "1.0.0",
        feature_registry_loader: Optional[FeatureRegistryLoader] = None,
        backfilling_service: Optional[BackfillingService] = None,
        dataset_publisher: Optional["DatasetPublisher"] = None,
    ):
        """
        Initialize dataset builder.
        
        Args:
            metadata_storage: Metadata storage for dataset records
            parquet_storage: Parquet storage for reading historical data
            dataset_storage_path: Base path for storing built datasets
            batch_size: Batch size for processing large datasets
            feature_registry_version: Feature Registry version to use
            feature_registry_loader: Optional Feature Registry loader for data type optimization
            backfilling_service: Optional backfilling service for automatic data fetching
        """
        self._metadata_storage = metadata_storage
        self._parquet_storage = parquet_storage
        self._dataset_storage_path = Path(dataset_storage_path)
        self._dataset_storage_path.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._feature_registry_version = feature_registry_version
        self._offline_engine = OfflineEngine(
            feature_registry_version=feature_registry_version,
            feature_registry_loader=feature_registry_loader,
        )
        self._active_builds: Dict[str, asyncio.Task] = {}
        self._feature_registry_loader = feature_registry_loader
        self._backfilling_service = backfilling_service
        self._dataset_publisher = dataset_publisher
    
    async def recover_incomplete_builds(self) -> None:
        """
        Recover incomplete dataset builds after service restart.
        
        Finds all datasets with status BUILDING and resumes their build process.
        This ensures that dataset builds survive container restarts.
        """
        try:
            # Find all datasets with BUILDING status
            building_datasets = await self._metadata_storage.list_datasets(
                status=DatasetStatus.BUILDING.value,
                limit=1000,  # Reasonable limit
            )
            
            if not building_datasets:
                logger.info("no_incomplete_builds_found")
                return
            
            logger.info(
                "recovering_incomplete_builds",
                count=len(building_datasets),
            )
            
            # Resume each incomplete build
            for dataset in building_datasets:
                dataset_id_raw = dataset.get("id")
                if not dataset_id_raw:
                    continue
                # Convert UUID to string if needed (asyncpg returns UUID objects)
                dataset_id = str(dataset_id_raw)
                
                try:
                    # Extract dataset parameters from stored metadata
                    symbol = dataset.get("symbol")
                    split_strategy_str = dataset.get("split_strategy")
                    target_config_dict = dataset.get("target_config", {})
                    
                    if not symbol or not split_strategy_str or not target_config_dict:
                        logger.warning(
                            "incomplete_dataset_metadata",
                            dataset_id=dataset_id,
                            missing_fields={
                                "symbol": symbol is None,
                                "split_strategy": split_strategy_str is None,
                                "target_config": target_config_dict is None,
                            },
                        )
                        # Mark as failed if critical metadata is missing
                        await self._metadata_storage.update_dataset(
                            dataset_id,
                            {
                                "status": DatasetStatus.FAILED.value,
                                "error_message": "Incomplete metadata after restart - cannot resume build",
                                "completed_at": datetime.now(timezone.utc),
                            },
                        )
                        continue
                    
                    # Parse split strategy
                    try:
                        split_strategy = SplitStrategy(split_strategy_str)
                    except ValueError:
                        logger.warning(
                            "invalid_split_strategy",
                            dataset_id=dataset_id,
                            split_strategy=split_strategy_str,
                        )
                        await self._metadata_storage.update_dataset(
                            dataset_id,
                            {
                                "status": DatasetStatus.FAILED.value,
                                "error_message": f"Invalid split strategy: {split_strategy_str}",
                                "completed_at": datetime.now(timezone.utc),
                            },
                        )
                        continue
                    
                    # Parse target config
                    # target_config may be stored as JSON string or dict (JSONB in PostgreSQL)
                    try:
                        import json
                        # asyncpg returns JSONB as dict, but check if it's a string
                        if isinstance(target_config_dict, str):
                            target_config_dict = json.loads(target_config_dict)
                        elif target_config_dict is None:
                            raise ValueError("target_config is None")
                        # Ensure it's a dict
                        if not isinstance(target_config_dict, dict):
                            raise ValueError(f"target_config must be dict, got {type(target_config_dict)}")
                        target_config = TargetConfig(**target_config_dict)
                    except Exception as e:
                        logger.warning(
                            "invalid_target_config",
                            dataset_id=dataset_id,
                            target_config_type=type(target_config_dict).__name__,
                            target_config_value=str(target_config_dict)[:100],
                            error=str(e),
                        )
                        await self._metadata_storage.update_dataset(
                            dataset_id,
                            {
                                "status": DatasetStatus.FAILED.value,
                                "error_message": f"Invalid target config: {str(e)}",
                                "completed_at": datetime.now(timezone.utc),
                            },
                        )
                        continue
                    
                    # Resume build task
                    build_task = asyncio.create_task(
                        self._build_dataset_task(dataset_id, symbol, split_strategy, target_config)
                    )
                    self._active_builds[dataset_id] = build_task
                    
                    logger.info(
                        "dataset_build_resumed",
                        dataset_id=dataset_id,
                        symbol=symbol,
                        split_strategy=split_strategy.value,
                    )
                
                except Exception as e:
                    logger.error(
                        "failed_to_resume_build",
                        dataset_id=dataset_id,
                        error=str(e),
                        exc_info=True,
                    )
                    # Mark as failed if we can't resume
                    try:
                        await self._metadata_storage.update_dataset(
                            dataset_id,
                            {
                                "status": DatasetStatus.FAILED.value,
                                "error_message": f"Failed to resume build after restart: {str(e)}",
                                "completed_at": datetime.now(timezone.utc),
                            },
                        )
                    except Exception as update_error:
                        logger.error(
                            "failed_to_update_failed_status",
                            dataset_id=dataset_id,
                            error=str(update_error),
                        )
        
        except Exception as e:
            logger.error(
                "recovery_error",
                error=str(e),
                exc_info=True,
            )
    
    async def build_dataset(
        self,
        symbol: str,
        split_strategy: SplitStrategy,
        target_config: TargetConfig,
        train_period_start: Optional[datetime] = None,
        train_period_end: Optional[datetime] = None,
        validation_period_start: Optional[datetime] = None,
        validation_period_end: Optional[datetime] = None,
        test_period_start: Optional[datetime] = None,
        test_period_end: Optional[datetime] = None,
        walk_forward_config: Optional[Dict[str, Any]] = None,
        output_format: str = "parquet",
        feature_registry_version: Optional[str] = None,
    ) -> str:
        """
        Build a dataset from historical data.
        
        Args:
            symbol: Trading pair symbol
            split_strategy: Split strategy (time_based or walk_forward)
            target_config: Target variable configuration
            train_period_start: Train period start (for time_based)
            train_period_end: Train period end (for time_based)
            validation_period_start: Validation period start (for time_based)
            validation_period_end: Validation period end (for time_based)
            test_period_start: Test period start (for time_based)
            test_period_end: Test period end (for time_based)
            walk_forward_config: Walk-forward configuration (for walk_forward)
            output_format: Output format (parquet, csv, hdf5)
            
        Returns:
            Dataset ID (UUID string)
        """
        # Normalize datetime objects to timezone-aware UTC before storing
        # This prevents asyncpg from mixing timezone-aware and timezone-naive datetimes
        def normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if not isinstance(dt, datetime):
                return dt
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        
        # Use version from request if provided, otherwise use default
        registry_version = feature_registry_version or self._feature_registry_version
        
        # Create dataset record
        dataset_data = {
            "symbol": symbol,
            "status": DatasetStatus.BUILDING.value,
            "split_strategy": split_strategy.value,
            "train_period_start": normalize_dt(train_period_start),
            "train_period_end": normalize_dt(train_period_end),
            "validation_period_start": normalize_dt(validation_period_start),
            "validation_period_end": normalize_dt(validation_period_end),
            "test_period_start": normalize_dt(test_period_start),
            "test_period_end": normalize_dt(test_period_end),
            "walk_forward_config": walk_forward_config,
            "target_config": target_config.model_dump(),
            "feature_registry_version": registry_version,
            "output_format": output_format,
        }
        
        dataset_id = await self._metadata_storage.create_dataset(dataset_data)
        
        # Start building in background
        build_task = asyncio.create_task(
            self._build_dataset_task(dataset_id, symbol, split_strategy, target_config)
        )
        self._active_builds[dataset_id] = build_task
        
        logger.info(
            "dataset_build_started",
            dataset_id=dataset_id,
            symbol=symbol,
            split_strategy=split_strategy.value,
        )
        
        return dataset_id
    
    async def _build_dataset_task(
        self,
        dataset_id: str,
        symbol: str,
        split_strategy: SplitStrategy,
        target_config: TargetConfig,
    ) -> None:
        """Background task for building dataset."""
        try:
            # Ensure dataset_id is a string (handle UUID objects)
            dataset_id = str(dataset_id)
            
            logger.info(
                "build_dataset_task_started",
                dataset_id=dataset_id,
                symbol=symbol,
                split_strategy=split_strategy.value,
            )
            
            # Get dataset record
            dataset = await self._metadata_storage.get_dataset(dataset_id)
            if dataset is None:
                logger.error("dataset_not_found_in_task", dataset_id=dataset_id)
                return
            
            logger.info(
                "dataset_record_loaded",
                dataset_id=dataset_id,
                dataset_status=dataset.get("status"),
                # Note: dataset.get("id") returns UUID object from asyncpg,
                # but we use dataset_id (string) everywhere in the code
                dataset_id_type=str(type(dataset.get("id"))),
            )
            
            # Determine date range
            logger.info(
                "determining_date_range",
                dataset_id=dataset_id,
                split_strategy=split_strategy.value,
            )
            logger.info(
                "determining_date_range",
                dataset_id=dataset_id,
                split_strategy=split_strategy.value,
            )
            if split_strategy == SplitStrategy.TIME_BASED:
                start_date = dataset["train_period_start"]
                end_date = dataset["test_period_end"]
                logger.info(
                    "date_range_determined",
                    dataset_id=dataset_id,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                )
                logger.info(
                    "date_range_determined",
                    dataset_id=dataset_id,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                )
            else:
                # Walk-forward: use config dates
                wf_config = dataset["walk_forward_config"]
                # Normalize datetime from ISO format to timezone-aware UTC
                # fromisoformat may return timezone-naive datetime if string doesn't contain timezone
                start_date_str = wf_config["start_date"]
                end_date_str = wf_config["end_date"]
                start_date = datetime.fromisoformat(start_date_str)
                end_date = datetime.fromisoformat(end_date_str)
                # Ensure timezone-aware UTC
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
                else:
                    start_date = start_date.astimezone(timezone.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                else:
                    end_date = end_date.astimezone(timezone.utc)
            
            # Check data availability
            logger.info(
                "checking_data_availability",
                dataset_id=dataset_id,
                symbol=symbol,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
            )
            available_period = await self._check_data_availability(symbol, start_date, end_date)
            logger.info(
                "data_availability_checked",
                dataset_id=dataset_id,
                available=available_period is not None,
            )
            if available_period is None:
                error_msg = f"Insufficient historical data for {symbol} in requested period"
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                logger.error("dataset_build_failed", dataset_id=dataset_id, error=error_msg)
                return
            
            # Read historical data
            logger.info("reading_historical_data", dataset_id=dataset_id, symbol=symbol)
            historical_data = await self._read_historical_data(
                symbol,
                available_period["start"],
                available_period["end"],
            )
            
            logger.info(
                "historical_data_read",
                dataset_id=dataset_id,
                symbol=symbol,
                trades_count=len(historical_data["trades"]),
                klines_count=len(historical_data["klines"]),
                snapshots_count=len(historical_data.get("snapshots", pd.DataFrame())),
                deltas_count=len(historical_data.get("deltas", pd.DataFrame())),
            )
            
            if historical_data["trades"].empty and historical_data["klines"].empty:
                error_msg = f"No historical data found for {symbol}"
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Compute features for all timestamps
            logger.info("computing_features", dataset_id=dataset_id)
            features_df = await self._compute_features_batch(
                symbol,
                historical_data,
                dataset_id,
            )
            
            if features_df.empty:
                error_msg = "No features computed"
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Validate features quality (check for NaN/None values)
            logger.info("validating_features_quality", dataset_id=dataset_id)
            validation_result = await self._validate_features_quality(features_df, dataset_id)
            if not validation_result["valid"]:
                error_msg = validation_result["error_message"]
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Update features_df if rows were filtered
            if validation_result.get("filtered_rows", 0) > 0:
                features_df = validation_result["filtered_features_df"]
                logger.info(
                    "features_quality_filtered",
                    dataset_id=dataset_id,
                    rows_before=validation_result["rows_before"],
                    rows_after=len(features_df),
                    filtered_rows=validation_result["filtered_rows"],
                )
            
            # Compute targets
            logger.info("computing_targets", dataset_id=dataset_id)
            targets_df = await self._compute_targets(
                features_df,
                historical_data,
                target_config,
            )
            
            # Validate no data leakage
            if not await self._validate_no_data_leakage(features_df, targets_df):
                error_msg = "Data leakage detected in features or targets"
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Split dataset
            logger.info("splitting_dataset", dataset_id=dataset_id, strategy=split_strategy.value)
            if split_strategy == SplitStrategy.TIME_BASED:
                splits = await self._split_time_based(
                    features_df,
                    targets_df,
                    dataset,
                )
            else:
                splits = await self._split_walk_forward(
                    features_df,
                    targets_df,
                    dataset,
                )
            
            # Write splits to storage
            logger.info("writing_dataset_splits", dataset_id=dataset_id)
            storage_path = await self._write_dataset_splits(
                dataset_id,
                splits,
                dataset["output_format"],
            )
            
            # Update dataset record
            total_train = len(splits["train"])
            total_val = len(splits["validation"])
            total_test = len(splits["test"])
            
            await self._metadata_storage.update_dataset(
                dataset_id,
                {
                    "status": DatasetStatus.READY.value,
                    "train_records": total_train,
                    "validation_records": total_val,
                    "test_records": total_test,
                    "storage_path": str(storage_path),
                    "completed_at": datetime.now(timezone.utc),
                    "estimated_completion": None,
                },
            )
            
            logger.info(
                "dataset_build_completed",
                dataset_id=dataset_id,
                train_records=total_train,
                validation_records=total_val,
                test_records=total_test,
            )
            
            # Publish completion notification
            if self._dataset_publisher:
                try:
                    await self._dataset_publisher.publish_dataset_ready(
                        dataset_id=dataset_id,
                        symbol=symbol,
                        status=DatasetStatus.READY.value,
                        train_records=total_train,
                        validation_records=total_val,
                        test_records=total_test,
                        trace_id=None,  # Could be extracted from dataset metadata if needed
                    )
                    logger.info(
                        "dataset_ready_notification_published",
                        dataset_id=dataset_id,
                        symbol=symbol,
                    )
                except Exception as e:
                    logger.warning(
                        "dataset_ready_notification_failed",
                        dataset_id=dataset_id,
                        symbol=symbol,
                        error=str(e),
                        exc_info=True,
                    )
            else:
                logger.debug(
                    "dataset_ready_notification_skipped",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    reason="dataset_publisher not initialized",
                )
        
        except Exception as e:
            logger.error(
                "dataset_build_error",
                dataset_id=dataset_id,
                error=str(e),
                exc_info=True,
            )
            await self._metadata_storage.update_dataset(
                dataset_id,
                {
                    "status": DatasetStatus.FAILED.value,
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc),
                },
            )
        finally:
            # Remove from active builds
            self._active_builds.pop(dataset_id, None)
    
    async def _check_data_availability(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Dict[str, datetime]]:
        """
        Check data availability and suggest available period.
        
        Args:
            symbol: Trading pair symbol
            start_date: Requested start date
            end_date: Requested end date
            
        Returns:
            Available period dict with start/end, or None if no data
        """
        # Check if data exists for the period
        # This is a simplified check - in production, would query Parquet metadata
        start_date_obj = start_date.date()
        end_date_obj = end_date.date()
        
        logger.info(
            "data_availability_check_start",
            symbol=symbol,
            start_date=start_date_obj.isoformat(),
            end_date=end_date_obj.isoformat(),
            start_datetime=start_date.isoformat() if start_date else None,
            end_datetime=end_date.isoformat() if end_date else None,
        )
        
        try:
            # Try to read a sample of data
            sample_trades = None
            sample_klines = None
            
            # Check trades
            logger.debug(
                "checking_trades_availability",
                symbol=symbol,
                start_date=start_date_obj.isoformat(),
                end_date=end_date_obj.isoformat(),
            )
            try:
                sample_trades = await self._parquet_storage.read_trades_range(
                    symbol, start_date_obj, end_date_obj
                )
                logger.debug(
                    "trades_range_read",
                    symbol=symbol,
                    start_date=start_date_obj.isoformat(),
                    end_date=end_date_obj.isoformat(),
                    count=len(sample_trades),
                    empty=sample_trades.empty,
                )
            except Exception as e:
                logger.warning(
                    "trades_range_read_error",
                    symbol=symbol,
                    start_date=start_date_obj.isoformat(),
                    end_date=end_date_obj.isoformat(),
                    error=str(e),
                    exc_info=True,
                )
                sample_trades = pd.DataFrame()
            
            # Check klines
            logger.debug(
                "checking_klines_availability",
                symbol=symbol,
                start_date=start_date_obj.isoformat(),
                end_date=end_date_obj.isoformat(),
            )
            try:
                sample_klines = await self._parquet_storage.read_klines_range(
                    symbol, start_date_obj, end_date_obj
                )
                logger.debug(
                    "klines_range_read",
                    symbol=symbol,
                    start_date=start_date_obj.isoformat(),
                    end_date=end_date_obj.isoformat(),
                    count=len(sample_klines),
                    empty=sample_klines.empty,
                )
            except Exception as e:
                logger.warning(
                    "klines_range_read_error",
                    symbol=symbol,
                    start_date=start_date_obj.isoformat(),
                    end_date=end_date_obj.isoformat(),
                    error=str(e),
                    exc_info=True,
                )
                sample_klines = pd.DataFrame()
            
            # Ensure we have DataFrames
            if sample_trades is None:
                sample_trades = pd.DataFrame()
            if sample_klines is None:
                sample_klines = pd.DataFrame()
            
            logger.info(
                "data_availability_check",
                symbol=symbol,
                start_date=start_date_obj.isoformat(),
                end_date=end_date_obj.isoformat(),
                trades_count=len(sample_trades),
                trades_empty=sample_trades.empty,
                klines_count=len(sample_klines),
                klines_empty=sample_klines.empty,
            )
            
            # If both are empty, we have insufficient data
            if sample_trades.empty and sample_klines.empty:
                # Check if automatic backfilling is enabled
                from src.config import config
                
                if (
                    config.feature_service_backfill_enabled
                    and config.feature_service_backfill_auto
                    and self._backfilling_service is not None
                ):
                    # Determine which data types need backfilling
                    data_types = None
                    if self._feature_registry_loader is not None:
                        try:
                            required_types = self._feature_registry_loader.get_required_data_types()
                            data_type_mapping = self._feature_registry_loader.get_data_type_mapping()
                            # Map input sources to storage types
                            data_types = []
                            for input_source in required_types:
                                if input_source in data_type_mapping:
                                    if "klines" in data_type_mapping[input_source]:
                                        data_types.append("klines")
                            logger.info(
                                "automatic_backfilling_triggered",
                                symbol=symbol,
                                missing_period_start=start_date_obj.isoformat(),
                                missing_period_end=end_date_obj.isoformat(),
                                required_data_types=data_types,
                            )
                        except Exception as e:
                            logger.warning(
                                "feature_registry_analysis_failed_for_backfill",
                                error=str(e),
                                fallback="backfilling_all_data_types",
                            )
                            data_types = ["klines"]  # Default to klines
                    else:
                        data_types = ["klines"]  # Default to klines
                    
                    # Trigger automatic backfilling
                    try:
                        job_id = await self._backfilling_service.backfill_historical(
                            symbol=symbol,
                            start_date=start_date_obj,
                            end_date=end_date_obj,
                            data_types=data_types,
                        )
                        
                        # Wait for backfilling to complete (with timeout)
                        max_wait_seconds = 300  # 5 minutes timeout
                        wait_interval = 5  # Check every 5 seconds
                        waited = 0
                        
                        while waited < max_wait_seconds:
                            job_status = self._backfilling_service.get_job_status(job_id)
                            if job_status is None:
                                break
                            
                            if job_status["status"] == "completed":
                                logger.info(
                                    "automatic_backfilling_completed",
                                    job_id=job_id,
                                    symbol=symbol,
                                    completed_dates=len(job_status.get("completed_dates", [])),
                                )
                                # Re-check data availability after backfilling
                                sample_trades = await self._parquet_storage.read_trades_range(
                                    symbol, start_date_obj, end_date_obj
                                )
                                sample_klines = await self._parquet_storage.read_klines_range(
                                    symbol, start_date_obj, end_date_obj
                                )
                                if not sample_trades.empty or not sample_klines.empty:
                                    return {
                                        "start": start_date,
                                        "end": end_date,
                                    }
                                break
                            elif job_status["status"] == "failed":
                                logger.warning(
                                    "automatic_backfilling_failed",
                                    job_id=job_id,
                                    symbol=symbol,
                                    error=job_status.get("error_message"),
                                )
                                break
                            
                            await asyncio.sleep(wait_interval)
                            waited += wait_interval
                        
                        if waited >= max_wait_seconds:
                            logger.warning(
                                "automatic_backfilling_timeout",
                                job_id=job_id,
                                symbol=symbol,
                                timeout_seconds=max_wait_seconds,
                            )
                    
                    except Exception as e:
                        logger.error(
                            "automatic_backfilling_error",
                            symbol=symbol,
                            error=str(e),
                            exc_info=True,
                        )
                        # Fall through to return None
                
                # Try to find available period
                # For now, return None - in production would search for available dates
                return None
            
            # Return available period (could be adjusted based on actual data)
            logger.info(
                "data_availability_check_success",
                symbol=symbol,
                start_date=start_date_obj.isoformat(),
                end_date=end_date_obj.isoformat(),
                trades_count=len(sample_trades),
                klines_count=len(sample_klines),
            )
            return {
                "start": start_date,
                "end": end_date,
            }
        except Exception as e:
            logger.error(
                "data_availability_check_error",
                symbol=symbol,
                start_date=start_date_obj.isoformat() if 'start_date_obj' in locals() else None,
                end_date=end_date_obj.isoformat() if 'end_date_obj' in locals() else None,
                error=str(e),
                exc_info=True,
            )
            return None
    
    async def _read_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """
        Read historical data for date range.
        
        Uses Feature Registry to determine which data types to load if available.
        Falls back to loading all data types if Feature Registry not provided.
        """
        start_date_obj = start_date.date()
        end_date_obj = end_date.date()
        
        # Determine required data types from Feature Registry if available
        required_data_types = None
        data_type_mapping = None
        
        if self._feature_registry_loader is not None:
            try:
                required_data_types = self._feature_registry_loader.get_required_data_types()
                data_type_mapping = self._feature_registry_loader.get_data_type_mapping()
                logger.info(
                    "loading_required_data_types",
                    symbol=symbol,
                    required_types=sorted(required_data_types),
                    mapping=data_type_mapping,
                )
            except Exception as e:
                logger.warning(
                    "feature_registry_analysis_failed",
                    error=str(e),
                    fallback="loading_all_data_types",
                )
        
        # Determine which data types to load
        load_orderbook = True
        load_trades = True
        load_klines = True
        load_ticker = True
        load_funding = True
        
        if required_data_types is not None:
            # Only load data types required by features
            load_orderbook = "orderbook" in required_data_types
            load_trades = "trades" in required_data_types
            load_klines = "kline" in required_data_types
            load_ticker = "ticker" in required_data_types
            load_funding = "funding" in required_data_types
        
        # Read required data types in parallel
        read_tasks = []
        
        if load_orderbook:
            read_tasks.append(
                ("snapshots", self._parquet_storage.read_orderbook_snapshots_range(
                    symbol, start_date_obj, end_date_obj
                ))
            )
            read_tasks.append(
                ("deltas", self._parquet_storage.read_orderbook_deltas_range(
                    symbol, start_date_obj, end_date_obj
                ))
            )
        else:
            read_tasks.append(("snapshots", asyncio.to_thread(lambda: pd.DataFrame())))
            read_tasks.append(("deltas", asyncio.to_thread(lambda: pd.DataFrame())))
        
        if load_trades:
            read_tasks.append(
                ("trades", self._parquet_storage.read_trades_range(symbol, start_date_obj, end_date_obj))
            )
        else:
            read_tasks.append(("trades", asyncio.to_thread(lambda: pd.DataFrame())))
        
        if load_klines:
            read_tasks.append(
                ("klines", self._parquet_storage.read_klines_range(symbol, start_date_obj, end_date_obj))
            )
        else:
            read_tasks.append(("klines", asyncio.to_thread(lambda: pd.DataFrame())))
        
        if load_ticker:
            # Ticker not implemented yet, return empty DataFrame
            read_tasks.append(("ticker", asyncio.to_thread(lambda: pd.DataFrame())))
        else:
            read_tasks.append(("ticker", asyncio.to_thread(lambda: pd.DataFrame())))
        
        if load_funding:
            # Funding not implemented yet, return empty DataFrame
            read_tasks.append(("funding", asyncio.to_thread(lambda: pd.DataFrame())))
        else:
            read_tasks.append(("funding", asyncio.to_thread(lambda: pd.DataFrame())))
        
        # Execute all reads in parallel
        results = await asyncio.gather(*[task[1] for task in read_tasks])
        
        # Build result dictionary
        result = {}
        for i, (name, _) in enumerate(read_tasks):
            result[name] = results[i]
        
        logger.debug(
            "historical_data_loaded",
            symbol=symbol,
            data_types_loaded=[name for name, _ in read_tasks if not result[name].empty],
            optimization_enabled=required_data_types is not None,
        )
        
        return result
    
    async def _update_progress_safe(
        self,
        dataset_id: str,
        update_data: Dict[str, Any],
        progress: float,
        processed: int,
        total: int,
    ) -> None:
        """Safely update dataset progress with error handling."""
        try:
            await self._metadata_storage.update_dataset(dataset_id, update_data)
        except Exception as e:
            # Log but don't fail - progress update is not critical
            logger.warning(
                "progress_update_failed",
                dataset_id=dataset_id,
                error=str(e),
                error_type=type(e).__name__,
                progress=f"{progress:.1f}%",
                processed=processed,
                total=total,
            )
    
    async def _compute_features_batch(
        self,
        symbol: str,
        historical_data: Dict[str, pd.DataFrame],
        dataset_id: str,
    ) -> pd.DataFrame:
        """Compute features for all timestamps in batch."""
        # Get all unique timestamps from trades and klines
        timestamps = set()
        
        if not historical_data["trades"].empty:
            timestamps.update(historical_data["trades"]["timestamp"].tolist())
        
        if not historical_data["klines"].empty:
            timestamps.update(historical_data["klines"]["timestamp"].tolist())
        
        if not timestamps:
            logger.warning(
                "no_timestamps_found",
                symbol=symbol,
                dataset_id=dataset_id,
                trades_empty=historical_data["trades"].empty,
                klines_empty=historical_data["klines"].empty,
            )
            return pd.DataFrame()
        
        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        logger.info(
            "feature_computation_started",
            dataset_id=dataset_id,
            symbol=symbol,
            total_timestamps=len(sorted_timestamps),
        )
        
        # Compute features for each timestamp
        features_list = []
        total = len(sorted_timestamps)
        
        for i, timestamp in enumerate(sorted_timestamps):
            if i % self._batch_size == 0:
                # Update progress (non-blocking to avoid connection pool exhaustion)
                progress = (i / total) * 100
                estimated_completion = datetime.now(timezone.utc) + timedelta(
                    seconds=(total - i) * 0.1
                )  # Rough estimate
                
                # Update progress in background task to avoid blocking on connection pool
                try:
                    # Use asyncio.create_task to make it non-blocking
                    # If connection pool is exhausted, this will fail gracefully
                    asyncio.create_task(
                        self._update_progress_safe(
                            dataset_id,
                            {"estimated_completion": estimated_completion},
                            progress,
                            i,
                            total,
                        )
                    )
                except Exception as e:
                    # Log but don't fail - progress update is not critical
                    logger.debug(
                        "progress_update_failed",
                        dataset_id=dataset_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                
                logger.info(
                    "feature_computation_progress",
                    dataset_id=dataset_id,
                    progress=f"{progress:.1f}%",
                    processed=i,
                    total=total,
                )
            
            # Compute features at timestamp
            feature_vector = await self._offline_engine.compute_features_at_timestamp(
                symbol=symbol,
                timestamp=timestamp if isinstance(timestamp, datetime) else pd.to_datetime(timestamp),
                orderbook_snapshots=historical_data["snapshots"],
                orderbook_deltas=historical_data["deltas"],
                trades=historical_data["trades"],
                klines=historical_data["klines"],
                ticker=historical_data.get("ticker"),
                funding=historical_data.get("funding"),
            )
            
            if feature_vector:
                # Convert to DataFrame row
                row = {
                    "timestamp": feature_vector.timestamp,
                    "symbol": feature_vector.symbol,
                    **feature_vector.features,
                }
                features_list.append(row)
            elif i < 10:  # Log first 10 failures for debugging
                logger.warning(
                    "feature_computation_failed_at_timestamp",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    timestamp=timestamp,
                    trades_count=len(historical_data["trades"]),
                    klines_count=len(historical_data["klines"]),
                    snapshots_count=len(historical_data.get("snapshots", pd.DataFrame())),
                    deltas_count=len(historical_data.get("deltas", pd.DataFrame())),
                )
        
        if not features_list:
            logger.warning(
                "no_features_computed",
                dataset_id=dataset_id,
                symbol=symbol,
                total_timestamps=total,
                features_computed=len(features_list),
            )
            return pd.DataFrame()
        
        logger.info(
            "features_computation_completed",
            dataset_id=dataset_id,
            symbol=symbol,
            total_timestamps=total,
            features_computed=len(features_list),
            success_rate=f"{(len(features_list) / total * 100):.1f}%",
        )
        
        return pd.DataFrame(features_list)
    
    async def _validate_features_quality(
        self,
        features_df: pd.DataFrame,
        dataset_id: str,
    ) -> Dict[str, Any]:
        """
        Validate features quality and check for NaN/None values.
        
        Args:
            features_df: DataFrame with computed features
            dataset_id: Dataset ID for logging
            
        Returns:
            Dict with validation results:
                - valid: bool - whether dataset is valid
                - error_message: str - error message if invalid
                - filtered_features_df: pd.DataFrame - filtered DataFrame if rows were removed
                - filtered_rows: int - number of rows filtered
                - rows_before: int - number of rows before filtering
                - nan_stats: dict - statistics about NaN values
        """
        if features_df.empty:
            return {
                "valid": False,
                "error_message": "Features DataFrame is empty after validation",
            }
        
        # Identify feature columns (exclude metadata columns)
        metadata_columns = {"timestamp", "symbol"}
        feature_columns = [col for col in features_df.columns if col not in metadata_columns]
        
        if not feature_columns:
            return {
                "valid": False,
                "error_message": "No feature columns found in DataFrame",
            }
        
        # Count NaN values per feature column
        nan_counts_per_feature = {}
        nan_ratios_per_feature = {}
        total_rows = len(features_df)
        
        for col in feature_columns:
            nan_count = features_df[col].isna().sum()
            nan_ratio = nan_count / total_rows if total_rows > 0 else 0.0
            nan_counts_per_feature[col] = int(nan_count)
            nan_ratios_per_feature[col] = float(nan_ratio)
        
        # Find features with high NaN ratio
        max_nan_ratio = config.dataset_max_feature_nan_ratio
        high_nan_features = {
            col: ratio
            for col, ratio in nan_ratios_per_feature.items()
            if ratio > max_nan_ratio
        }
        
        # Count NaN values per row (across all features)
        features_only_df = features_df[feature_columns]
        nan_counts_per_row = features_only_df.isna().sum(axis=1)
        nan_ratios_per_row = nan_counts_per_row / len(feature_columns)
        
        # Find rows with high NaN ratio
        max_row_nan_ratio = config.dataset_max_row_nan_ratio
        min_valid_features_ratio = config.dataset_min_valid_features_ratio
        
        # Filter rows: drop rows where NaN ratio is too high OR valid features ratio is too low
        rows_to_keep = (
            (nan_ratios_per_row <= max_row_nan_ratio) &
            ((1.0 - nan_ratios_per_row) >= min_valid_features_ratio)
        )
        
        filtered_features_df = features_df[rows_to_keep].copy()
        filtered_rows_count = int((~rows_to_keep).sum())
        
        # Calculate statistics
        total_nan_values = int(features_only_df.isna().sum().sum())
        total_cells = len(feature_columns) * total_rows
        overall_nan_ratio = total_nan_values / total_cells if total_cells > 0 else 0.0
        
        # Log statistics
        logger.info(
            "features_quality_check",
            dataset_id=dataset_id,
            total_rows=total_rows,
            total_features=len(feature_columns),
            total_nan_values=total_nan_values,
            overall_nan_ratio=f"{overall_nan_ratio:.2%}",
            features_with_high_nan=len(high_nan_features),
            rows_filtered=filtered_rows_count,
            rows_after_filtering=len(filtered_features_df),
        )
        
        # Log features with high NaN ratio (top 10)
        if high_nan_features:
            sorted_high_nan = sorted(
                high_nan_features.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            logger.warning(
                "features_with_high_nan_ratio",
                dataset_id=dataset_id,
                max_allowed_ratio=f"{max_nan_ratio:.2%}",
                high_nan_features=[
                    {"feature": col, "nan_ratio": f"{ratio:.2%}", "nan_count": nan_counts_per_feature[col]}
                    for col, ratio in sorted_high_nan
                ],
            )
        
        # Check if dataset should fail due to high NaN ratio
        if config.dataset_fail_on_high_nan_ratio and high_nan_features:
            error_msg = (
                f"Dataset build failed: {len(high_nan_features)} feature(s) have NaN ratio "
                f"above threshold ({max_nan_ratio:.2%}). "
                f"Top problematic features: {', '.join([f'{col} ({ratio:.2%})' for col, ratio in sorted(high_nan_features.items(), key=lambda x: x[1], reverse=True)[:5]])}"
            )
            return {
                "valid": False,
                "error_message": error_msg,
                "nan_stats": {
                    "high_nan_features": high_nan_features,
                    "overall_nan_ratio": overall_nan_ratio,
                },
            }
        
        # Check if too many rows were filtered
        rows_after = len(filtered_features_df)
        if rows_after == 0:
            error_msg = (
                f"Dataset build failed: All rows were filtered out due to high NaN ratio. "
                f"Original rows: {total_rows}, filtered: {filtered_rows_count}"
            )
            return {
                "valid": False,
                "error_message": error_msg,
                "nan_stats": {
                    "rows_before": total_rows,
                    "rows_after": 0,
                    "filtered_rows": filtered_rows_count,
                },
            }
        
        # Log summary statistics
        if rows_after < total_rows * 0.5:
            logger.warning(
                "features_quality_warning_high_filtering",
                dataset_id=dataset_id,
                rows_before=total_rows,
                rows_after=rows_after,
                filtered_ratio=f"{(filtered_rows_count / total_rows):.2%}",
                warning="More than 50% of rows were filtered due to high NaN ratio",
            )
        
        return {
            "valid": True,
            "filtered_features_df": filtered_features_df,
            "filtered_rows": filtered_rows_count,
            "rows_before": total_rows,
            "rows_after": rows_after,
            "nan_stats": {
                "overall_nan_ratio": overall_nan_ratio,
                "high_nan_features": high_nan_features,
                "nan_counts_per_feature": nan_counts_per_feature,
                "nan_ratios_per_feature": nan_ratios_per_feature,
                "rows_filtered": filtered_rows_count,
                "rows_before": total_rows,
                "rows_after": rows_after,
            },
        }
    
    async def _compute_targets(
        self,
        features_df: pd.DataFrame,
        historical_data: Dict[str, pd.DataFrame],
        target_config: TargetConfig,
    ) -> pd.DataFrame:
        """Compute target variables."""
        if features_df.empty:
            logger.warning("_compute_targets: features_df is empty")
            return pd.DataFrame()
        
        # Merge with price data for target computation
        # Use klines for price data
        price_df = historical_data["klines"].copy()
        if price_df.empty:
            logger.warning("_compute_targets: klines is empty, trying trades")
            # Fallback to trades
            price_df = historical_data["trades"].copy()
            if not price_df.empty:
                price_df = price_df.groupby("timestamp")["price"].last().reset_index()
        
        if price_df.empty:
            logger.warning("_compute_targets: price_df is empty after fallback")
            return pd.DataFrame()
        
        # Check available columns in price_df
        logger.info(
            "_compute_targets: price_df info",
            columns=list(price_df.columns),
            shape=price_df.shape,
            timestamp_dtype=str(price_df["timestamp"].dtype) if "timestamp" in price_df.columns else "missing",
        )
        
        # Determine price column name (could be "close" or "price")
        price_col = None
        if "close" in price_df.columns:
            price_col = "close"
        elif "price" in price_df.columns:
            price_col = "price"
        else:
            logger.error(
                "_compute_targets: no price column found",
                available_columns=list(price_df.columns),
            )
            return pd.DataFrame()
        
        logger.info(
            "_compute_targets: features_df info before merge",
            features_shape=features_df.shape,
            features_timestamp_dtype=str(features_df["timestamp"].dtype) if "timestamp" in features_df.columns else "missing",
            features_timestamp_sample=features_df["timestamp"].head(3).tolist() if "timestamp" in features_df.columns and not features_df.empty else [],
        )
        
        logger.info(
            "_compute_targets: price_df info before merge",
            price_shape=price_df.shape,
            price_timestamp_sample=price_df["timestamp"].head(3).tolist() if "timestamp" in price_df.columns and not price_df.empty else [],
            price_col=price_col,
            price_col_sample=price_df[price_col].head(5).tolist() if price_col and not price_df.empty else [],
            price_col_min=float(price_df[price_col].min()) if price_col and not price_df.empty else None,
            price_col_max=float(price_df[price_col].max()) if price_col and not price_df.empty else None,
        )
        
        # Merge features with prices
        price_for_merge = price_df[["timestamp", price_col]].rename(columns={price_col: "price"})
        logger.info(
            "_compute_targets: price_for_merge",
            columns=list(price_for_merge.columns),
            shape=price_for_merge.shape,
            price_sample=price_for_merge["price"].head(5).tolist() if "price" in price_for_merge.columns else [],
        )
        
        merged = features_df.merge(
            price_for_merge,
            on="timestamp",
            how="left",
        )
        
        logger.info(
            "_compute_targets: after merge",
            merged_shape=merged.shape,
            merged_columns=list(merged.columns),
            features_count=len(features_df),
            price_count=len(price_df),
            merged_with_price_count=merged["price"].notna().sum() if "price" in merged.columns else 0,
            merged_price_sample=merged["price"].head(5).tolist() if "price" in merged.columns else [],
            merged_price_min=float(merged["price"].min()) if "price" in merged.columns and merged["price"].notna().any() else None,
            merged_price_max=float(merged["price"].max()) if "price" in merged.columns and merged["price"].notna().any() else None,
        )
        
        # Compute targets based on type
        if target_config.type == "regression":
            return self._compute_regression_targets(merged, target_config.horizon)
        elif target_config.type == "classification":
            from src.config import config
            threshold = target_config.threshold or config.model_classification_threshold
            return self._compute_classification_targets(
                merged, target_config.horizon, threshold
            )
        else:  # risk_adjusted
            return self._compute_risk_adjusted_targets(merged, target_config.horizon)
    
    def _compute_regression_targets(
        self,
        data: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Compute regression targets (returns).
        
        Args:
            data: DataFrame with timestamp and price columns
            horizon: Prediction horizon in seconds
        """
        if data.empty:
            logger.warning("_compute_regression_targets: input data is empty")
            return pd.DataFrame()
        
        if "price" not in data.columns:
            logger.error("_compute_regression_targets: 'price' column not found", columns=list(data.columns))
            return pd.DataFrame()
        
        # Check how many rows have valid prices
        valid_prices = data["price"].notna().sum()
        logger.info(
            "_compute_regression_targets: input data",
            total_rows=len(data),
            valid_prices=valid_prices,
            price_nan_count=data["price"].isna().sum(),
        )
        
        if valid_prices == 0:
            logger.warning("_compute_regression_targets: no valid prices in data")
            return pd.DataFrame()
        
        # Sort by timestamp
        data = data.sort_values("timestamp").copy()
        
        # Compute future price by time, not by index
        # Create future timestamps
        data["future_timestamp"] = data["timestamp"] + pd.Timedelta(seconds=horizon)
        
        # Save original price column BEFORE any sorting or merging
        # Store as DataFrame with timestamp for reliable merge
        original_price_df = data[["timestamp", "price"]].copy()
        
        logger.info(
            "_compute_regression_targets: before creating price_lookup",
            data_price_sample=data["price"].head(5).tolist(),
            data_price_min=float(data["price"].min()) if data["price"].notna().any() else None,
            data_price_max=float(data["price"].max()) if data["price"].notna().any() else None,
        )
        
        # Create price lookup DataFrame - only use rows with valid prices
        # Rename price to future_price in lookup to avoid conflict
        price_lookup = data[["timestamp", "price"]].copy()
        price_lookup = price_lookup[price_lookup["price"].notna()].copy()
        price_lookup = price_lookup.rename(columns={"price": "future_price"})
        price_lookup = price_lookup.sort_values("timestamp")
        
        if price_lookup.empty:
            logger.warning("_compute_regression_targets: price_lookup is empty after filtering")
            return pd.DataFrame()
        
        logger.info(
            "_compute_regression_targets: before merge_asof",
            data_rows=len(data),
            price_lookup_rows=len(price_lookup),
            horizon_seconds=horizon,
            data_columns=list(data.columns),
            price_lookup_columns=list(price_lookup.columns),
            price_lookup_future_price_sample=price_lookup["future_price"].head(5).tolist(),
        )
        
        # Sort data by future_timestamp for merge_asof
        data_sorted = data.sort_values("future_timestamp").copy()
        
        # Use merge_asof to find nearest future price
        # direction='forward' ensures we only look forward in time
        data_merged = pd.merge_asof(
            data_sorted,
            price_lookup,
            left_on="future_timestamp",
            right_on="timestamp",
            direction="forward",
            suffixes=("", "_future"),
        )
        
        # Restore original price column by merging with original_price_df
        # This ensures we get the correct price for each timestamp
        data_merged = data_merged.merge(
            original_price_df,
            on="timestamp",
            how="left",
            suffixes=("", "_original"),
        )
        
        # Use the original price (from merge), drop the one that might have been overwritten
        if "price_original" in data_merged.columns:
            data_merged["price"] = data_merged["price_original"]
            data_merged = data_merged.drop(columns=["price_original"])
        
        logger.info(
            "_compute_regression_targets: after merge_asof and price restoration",
            price_sample=data_merged["price"].head(5).tolist() if "price" in data_merged.columns else [],
            price_min=float(data_merged["price"].min()) if "price" in data_merged.columns and data_merged["price"].notna().any() else None,
            price_max=float(data_merged["price"].max()) if "price" in data_merged.columns and data_merged["price"].notna().any() else None,
            price_notna_count=data_merged["price"].notna().sum() if "price" in data_merged.columns else 0,
            timestamp_sample=data_merged["timestamp"].head(3).tolist() if "timestamp" in data_merged.columns else [],
        )
        
        data = data_merged
        
        # Log columns after merge_asof
        logger.info(
            "_compute_regression_targets: after merge_asof",
            columns=list(data.columns),
            total_rows=len(data),
            future_price_exists="future_price" in data.columns,
            price_exists="price" in data.columns,
            future_price_notna=data["future_price"].notna().sum() if "future_price" in data.columns else 0,
            price_notna=data["price"].notna().sum() if "price" in data.columns else 0,
        )
        
        # Check that future_price column exists (we renamed it before merge)
        if "future_price" not in data.columns:
            logger.error(
                "_compute_regression_targets: future_price column not found after merge_asof",
                available_columns=list(data.columns),
            )
            return pd.DataFrame()
        
        # Drop helper column
        data = data.drop(columns=["future_timestamp", "timestamp_future"], errors="ignore")
        
        # Check how many rows have future prices
        future_price_count = data["future_price"].notna().sum()
        price_count = data["price"].notna().sum() if "price" in data.columns else 0
        
        logger.info(
            "_compute_regression_targets: after merge_asof (after rename)",
            total_rows=len(data),
            columns=list(data.columns),
            future_price_count=future_price_count,
            future_price_nan_count=data["future_price"].isna().sum(),
            price_count=price_count,
            price_nan_count=data["price"].isna().sum() if "price" in data.columns else 0,
            both_prices_valid=(data["future_price"].notna() & data["price"].notna()).sum() if "price" in data.columns else 0,
        )
        
        if future_price_count == 0:
            logger.warning("_compute_regression_targets: no future prices found after merge_asof")
            return pd.DataFrame()
        
        if price_count == 0:
            logger.warning("_compute_regression_targets: no current prices found after merge_asof")
            return pd.DataFrame()
        
        # Check for rows where both prices are valid
        valid_rows = data["future_price"].notna() & data["price"].notna()
        if valid_rows.sum() == 0:
            logger.warning(
                "_compute_regression_targets: no rows with both future_price and price valid",
                future_price_valid=future_price_count,
                price_valid=price_count,
            )
            return pd.DataFrame()
        
        # Compute return only for rows with both prices valid
        logger.info(
            "_compute_regression_targets: before target computation",
            valid_rows_count=valid_rows.sum(),
            sample_price=data["price"][valid_rows].head(3).tolist() if valid_rows.any() else [],
            sample_future_price=data["future_price"][valid_rows].head(3).tolist() if valid_rows.any() else [],
        )
        
        # Compute return
        data["target"] = (data["future_price"] - data["price"]) / data["price"]
        
        # Check target values before dropna
        target_stats = {
            "total": len(data),
            "notna": data["target"].notna().sum(),
            "isna": data["target"].isna().sum(),
            "inf": (data["target"] == float("inf")).sum() if "target" in data.columns else 0,
            "neginf": (data["target"] == float("-inf")).sum() if "target" in data.columns else 0,
        }
        
        if "target" in data.columns and data["target"].notna().any():
            target_stats["min"] = float(data["target"].min())
            target_stats["max"] = float(data["target"].max())
            target_stats["mean"] = float(data["target"].mean())
        
        logger.info(
            "_compute_regression_targets: target computation stats",
            **target_stats,
        )
        
        # Remove rows where target cannot be computed (NaN, inf, -inf)
        data = data.dropna(subset=["target"])
        data = data[~data["target"].isin([float("inf"), float("-inf")])]
        
        logger.info(
            "_compute_regression_targets: final result",
            target_rows=len(data),
            target_computed_count=data["target"].notna().sum() if "target" in data.columns else 0,
        )
        
        if data.empty:
            logger.warning(
                "_compute_regression_targets: result is empty after filtering",
                original_rows=target_stats["total"],
                dropped_na=target_stats["isna"],
                dropped_inf=target_stats["inf"] + target_stats["neginf"],
            )
        
        return data[["timestamp", "target"]]
    
    def _compute_classification_targets(
        self,
        data: pd.DataFrame,
        horizon: int,
        threshold: float,
    ) -> pd.DataFrame:
        """Compute classification targets (direction)."""
        # First compute returns
        targets_df = self._compute_regression_targets(data, horizon)
        
        # Classify: 1 for positive above threshold, -1 for negative below -threshold, 0 otherwise
        targets_df["target"] = targets_df["target"].apply(
            lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)
        )
        
        return targets_df
    
    def _compute_risk_adjusted_targets(
        self,
        data: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """Compute risk-adjusted targets (Sharpe ratio)."""
        # Compute returns first
        returns_df = self._compute_regression_targets(data, horizon)
        
        # Compute rolling volatility
        returns_df["volatility"] = returns_df["target"].rolling(window=20).std()
        
        # Compute Sharpe ratio
        returns_df["target"] = returns_df["target"] / returns_df["volatility"]
        returns_df = returns_df.dropna(subset=["target"])
        
        return returns_df[["timestamp", "target"]]
    
    async def _validate_no_data_leakage(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
    ) -> bool:
        """Validate no data leakage in features or targets."""
        logger.debug(
            "_validate_no_data_leakage: input",
            features_shape=features_df.shape,
            features_columns=list(features_df.columns) if not features_df.empty else [],
            targets_shape=targets_df.shape,
            targets_columns=list(targets_df.columns) if not targets_df.empty else [],
        )
        
        if features_df.empty:
            logger.warning("_validate_no_data_leakage: features_df is empty")
            return False
        
        if targets_df.empty:
            logger.warning("_validate_no_data_leakage: targets_df is empty")
            return False
        
        # Merge to check timestamps
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        logger.debug(
            "_validate_no_data_leakage: after merge",
            merged_shape=merged.shape,
            features_timestamps_count=len(features_df["timestamp"].unique()) if "timestamp" in features_df.columns else 0,
            targets_timestamps_count=len(targets_df["timestamp"].unique()) if "timestamp" in targets_df.columns else 0,
        )
        
        if merged.empty:
            logger.warning(
                "_validate_no_data_leakage: merged is empty",
                features_timestamp_range=(
                    (features_df["timestamp"].min(), features_df["timestamp"].max())
                    if "timestamp" in features_df.columns and not features_df.empty
                    else None
                ),
                targets_timestamp_range=(
                    (targets_df["timestamp"].min(), targets_df["timestamp"].max())
                    if "timestamp" in targets_df.columns and not targets_df.empty
                    else None
                ),
            )
            return False
        
        # For each row, verify:
        # - Features use only data <= timestamp
        # - Targets use only data > timestamp
        # This is a simplified check - in production would validate against Feature Registry
        
        logger.debug("_validate_no_data_leakage: validation passed", merged_rows=len(merged))
        return True
    
    async def _split_time_based(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        dataset: dict,
    ) -> Dict[str, pd.DataFrame]:
        """Split dataset using time-based strategy."""
        # Merge features and targets
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        # Normalize timestamp column to timezone-aware UTC if needed
        # This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
        if merged["timestamp"].dtype.tz is None:
            # If timestamp is timezone-naive, assume UTC
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        else:
            # If timestamp is timezone-aware, convert to UTC
            merged["timestamp"] = merged["timestamp"].dt.tz_convert(timezone.utc)
        
        # Normalize datetime objects from dataset to timezone-aware UTC
        train_period_start = dataset["train_period_start"]
        if isinstance(train_period_start, datetime):
            if train_period_start.tzinfo is None:
                train_period_start = train_period_start.replace(tzinfo=timezone.utc)
            else:
                train_period_start = train_period_start.astimezone(timezone.utc)
        
        train_period_end = dataset["train_period_end"]
        if isinstance(train_period_end, datetime):
            if train_period_end.tzinfo is None:
                train_period_end = train_period_end.replace(tzinfo=timezone.utc)
            else:
                train_period_end = train_period_end.astimezone(timezone.utc)
        
        validation_period_start = dataset["validation_period_start"]
        if isinstance(validation_period_start, datetime):
            if validation_period_start.tzinfo is None:
                validation_period_start = validation_period_start.replace(tzinfo=timezone.utc)
            else:
                validation_period_start = validation_period_start.astimezone(timezone.utc)
        
        validation_period_end = dataset["validation_period_end"]
        if isinstance(validation_period_end, datetime):
            if validation_period_end.tzinfo is None:
                validation_period_end = validation_period_end.replace(tzinfo=timezone.utc)
            else:
                validation_period_end = validation_period_end.astimezone(timezone.utc)
        
        test_period_start = dataset["test_period_start"]
        if isinstance(test_period_start, datetime):
            if test_period_start.tzinfo is None:
                test_period_start = test_period_start.replace(tzinfo=timezone.utc)
            else:
                test_period_start = test_period_start.astimezone(timezone.utc)
        
        test_period_end = dataset["test_period_end"]
        if isinstance(test_period_end, datetime):
            if test_period_end.tzinfo is None:
                test_period_end = test_period_end.replace(tzinfo=timezone.utc)
            else:
                test_period_end = test_period_end.astimezone(timezone.utc)
        
        # Split by periods
        train = merged[
            (merged["timestamp"] >= train_period_start) &
            (merged["timestamp"] < train_period_end)
        ]
        
        validation = merged[
            (merged["timestamp"] >= validation_period_start) &
            (merged["timestamp"] < validation_period_end)
        ]
        
        test = merged[
            (merged["timestamp"] >= test_period_start) &
            (merged["timestamp"] <= test_period_end)
        ]
        
        return {
            "train": train,
            "validation": validation,
            "test": test,
        }
    
    async def _split_walk_forward(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        dataset: dict,
    ) -> Dict[str, pd.DataFrame]:
        """Split dataset using walk-forward strategy."""
        # Merge features and targets
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp")
        
        wf_config = dataset["walk_forward_config"]
        train_days = wf_config["train_window_days"]
        val_days = wf_config["validation_window_days"]
        test_days = wf_config["test_window_days"]
        step_days = wf_config["step_days"]
        
        # Normalize datetime from ISO format to timezone-aware UTC
        # fromisoformat may return timezone-naive datetime if string doesn't contain timezone
        start_date_str = wf_config["start_date"]
        end_date_str = wf_config["end_date"]
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
        # Ensure timezone-aware UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        else:
            start_date = start_date.astimezone(timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        else:
            end_date = end_date.astimezone(timezone.utc)
        
        # For simplicity, use the first fold
        # In production, would generate multiple folds
        train_start = start_date
        train_end = train_start + timedelta(days=train_days)
        val_start = train_end
        val_end = val_start + timedelta(days=val_days)
        test_start = val_end
        test_end = test_start + timedelta(days=test_days)
        
        # Normalize timestamp column to timezone-aware UTC if needed
        # This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
        if merged["timestamp"].dtype.tz is None:
            # If timestamp is timezone-naive, assume UTC
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        else:
            # If timestamp is timezone-aware, convert to UTC
            merged["timestamp"] = merged["timestamp"].dt.tz_convert(timezone.utc)
        
        # Ensure all datetime objects are timezone-aware UTC
        if train_start.tzinfo is None:
            train_start = train_start.replace(tzinfo=timezone.utc)
        else:
            train_start = train_start.astimezone(timezone.utc)
        
        if train_end.tzinfo is None:
            train_end = train_end.replace(tzinfo=timezone.utc)
        else:
            train_end = train_end.astimezone(timezone.utc)
        
        if val_start.tzinfo is None:
            val_start = val_start.replace(tzinfo=timezone.utc)
        else:
            val_start = val_start.astimezone(timezone.utc)
        
        if val_end.tzinfo is None:
            val_end = val_end.replace(tzinfo=timezone.utc)
        else:
            val_end = val_end.astimezone(timezone.utc)
        
        if test_start.tzinfo is None:
            test_start = test_start.replace(tzinfo=timezone.utc)
        else:
            test_start = test_start.astimezone(timezone.utc)
        
        if test_end.tzinfo is None:
            test_end = test_end.replace(tzinfo=timezone.utc)
        else:
            test_end = test_end.astimezone(timezone.utc)
        
        train = merged[
            (merged["timestamp"] >= train_start) &
            (merged["timestamp"] < train_end)
        ]
        
        validation = merged[
            (merged["timestamp"] >= val_start) &
            (merged["timestamp"] < val_end)
        ]
        
        test = merged[
            (merged["timestamp"] >= test_start) &
            (merged["timestamp"] <= test_end)
        ]
        
        return {
            "train": train,
            "validation": validation,
            "test": test,
        }
    
    async def _write_dataset_splits(
        self,
        dataset_id: str,
        splits: Dict[str, pd.DataFrame],
        output_format: str,
    ) -> Path:
        """Write dataset splits to storage."""
        # Ensure dataset_id is a string (handle UUID objects)
        dataset_id_str = str(dataset_id)
        dataset_dir = self._dataset_storage_path / dataset_id_str
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Write each split
        for split_name, split_df in splits.items():
            if split_df.empty:
                continue
            
            file_path = dataset_dir / f"{split_name}.{output_format}"
            
            if output_format == "parquet":
                await asyncio.to_thread(
                    split_df.to_parquet,
                    file_path,
                    index=False,
                )
            elif output_format == "csv":
                await asyncio.to_thread(
                    split_df.to_csv,
                    file_path,
                    index=False,
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        
        return dataset_dir
    
    async def get_build_progress(self, dataset_id: str) -> Optional[dict]:
        """Get build progress for dataset."""
        dataset = await self._metadata_storage.get_dataset(dataset_id)
        if dataset is None:
            return None
        
        return {
            "status": dataset["status"],
            "estimated_completion": dataset.get("estimated_completion"),
            "train_records": dataset.get("train_records", 0),
            "validation_records": dataset.get("validation_records", 0),
            "test_records": dataset.get("test_records", 0),
        }
    
    async def resplit_dataset(
        self,
        dataset_id: str,
        train_period_start: Optional[datetime] = None,
        train_period_end: Optional[datetime] = None,
        validation_period_start: Optional[datetime] = None,
        validation_period_end: Optional[datetime] = None,
        test_period_start: Optional[datetime] = None,
        test_period_end: Optional[datetime] = None,
    ) -> None:
        """
        Resplit an existing ready dataset with new time periods.
        
        Loads all existing splits (train, validation, test), merges them,
        and applies new time-based splitting. Only works for time_based split strategy.
        
        Args:
            dataset_id: Existing dataset ID
            train_period_start: New train period start (optional, uses existing if not provided)
            train_period_end: New train period end (optional, uses existing if not provided)
            validation_period_start: New validation period start (optional, uses existing if not provided)
            validation_period_end: New validation period end (optional, uses existing if not provided)
            test_period_start: New test period start (optional, uses existing if not provided)
            test_period_end: New test period end (optional, uses existing if not provided)
        """
        # Get existing dataset
        dataset = await self._metadata_storage.get_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if dataset["status"] != DatasetStatus.READY.value:
            raise ValueError(f"Dataset {dataset_id} is not ready (status: {dataset['status']})")
        
        if dataset["split_strategy"] != SplitStrategy.TIME_BASED.value:
            raise ValueError("Resplitting only supported for time_based split strategy")
        
        storage_path = dataset.get("storage_path")
        if not storage_path:
            raise ValueError(f"Dataset {dataset_id} has no storage path")
        
        output_format = dataset.get("output_format", "parquet")
        dataset_dir = Path(storage_path)
        
        # Load all existing splits
        logger.info("loading_existing_splits", dataset_id=dataset_id)
        all_splits = []
        
        for split_name in ["train", "validation", "test"]:
            split_file = dataset_dir / f"{split_name}.{output_format}"
            if split_file.exists():
                try:
                    if output_format == "parquet":
                        split_df = await asyncio.to_thread(pd.read_parquet, split_file)
                    elif output_format == "csv":
                        split_df = await asyncio.to_thread(pd.read_csv, split_file)
                    else:
                        logger.warning(
                            "unsupported_format_for_resplit",
                            dataset_id=dataset_id,
                            format=output_format,
                            split=split_name,
                        )
                        continue
                    
                    if not split_df.empty:
                        all_splits.append(split_df)
                        logger.info(
                            "loaded_split",
                            dataset_id=dataset_id,
                            split=split_name,
                            records=len(split_df),
                        )
                except Exception as e:
                    logger.warning(
                        "failed_to_load_split",
                        dataset_id=dataset_id,
                        split=split_name,
                        error=str(e),
                    )
        
        if not all_splits:
            raise ValueError(f"No splits found for dataset {dataset_id}")
        
        # Merge all splits
        merged_df = pd.concat(all_splits, ignore_index=True)
        merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(
            "merged_splits",
            dataset_id=dataset_id,
            total_records=len(merged_df),
            date_range_start=merged_df["timestamp"].min(),
            date_range_end=merged_df["timestamp"].max(),
        )
        
        # Prepare new periods (use provided or existing)
        new_periods = {
            "train_period_start": train_period_start or dataset["train_period_start"],
            "train_period_end": train_period_end or dataset["train_period_end"],
            "validation_period_start": validation_period_start or dataset["validation_period_start"],
            "validation_period_end": validation_period_end or dataset["validation_period_end"],
            "test_period_start": test_period_start or dataset["test_period_start"],
            "test_period_end": test_period_end or dataset["test_period_end"],
        }
        
        # Validate all periods are set
        if not all(new_periods.values()):
            raise ValueError("All periods must be specified for resplitting")
        
        # Normalize datetime objects to timezone-aware UTC
        for key, value in new_periods.items():
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    new_periods[key] = value.replace(tzinfo=timezone.utc)
                else:
                    new_periods[key] = value.astimezone(timezone.utc)
        
        # Split merged data by new periods
        logger.info("resplitting_dataset", dataset_id=dataset_id, periods=new_periods)
        
        # Separate features and targets (target column should exist)
        if "target" not in merged_df.columns:
            raise ValueError("Dataset does not contain 'target' column")
        
        features_df = merged_df.drop(columns=["target"])
        targets_df = merged_df[["timestamp", "target"]]
        
        # Apply new splitting
        new_splits = await self._split_time_based(
            features_df,
            targets_df,
            new_periods,
        )
        
        # Write new splits
        logger.info("writing_resplit_splits", dataset_id=dataset_id)
        storage_path = await self._write_dataset_splits(
            dataset_id,
            new_splits,
            output_format,
        )
        
        # Update dataset metadata
        total_train = len(new_splits["train"])
        total_val = len(new_splits["validation"])
        total_test = len(new_splits["test"])
        
        update_data = {
            "train_records": total_train,
            "validation_records": total_val,
            "test_records": total_test,
            "storage_path": str(storage_path),
            "completed_at": datetime.now(timezone.utc),
        }
        
        # Update periods if they were changed
        if train_period_start is not None:
            update_data["train_period_start"] = new_periods["train_period_start"]
        if train_period_end is not None:
            update_data["train_period_end"] = new_periods["train_period_end"]
        if validation_period_start is not None:
            update_data["validation_period_start"] = new_periods["validation_period_start"]
        if validation_period_end is not None:
            update_data["validation_period_end"] = new_periods["validation_period_end"]
        if test_period_start is not None:
            update_data["test_period_start"] = new_periods["test_period_start"]
        if test_period_end is not None:
            update_data["test_period_end"] = new_periods["test_period_end"]
        
        await self._metadata_storage.update_dataset(dataset_id, update_data)
        
        logger.info(
            "dataset_resplit_completed",
            dataset_id=dataset_id,
            train_records=total_train,
            validation_records=total_val,
            test_records=total_test,
        )