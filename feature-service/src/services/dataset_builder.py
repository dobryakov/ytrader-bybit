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
        self._offline_engine = OfflineEngine(feature_registry_version)
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
                dataset_id = dataset.get("id")
                if not dataset_id:
                    continue
                
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
            "feature_registry_version": self._feature_registry_version,
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
            
            # Get dataset record
            dataset = await self._metadata_storage.get_dataset(dataset_id)
            if dataset is None:
                logger.error(f"Dataset {dataset_id} not found")
                return
            
            # Determine date range
            if split_strategy == SplitStrategy.TIME_BASED:
                start_date = dataset["train_period_start"]
                end_date = dataset["test_period_end"]
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
            available_period = await self._check_data_availability(symbol, start_date, end_date)
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
        try:
            # Try to read a sample of data
            start_date_obj = start_date.date()
            end_date_obj = end_date.date()
            
            # Check both trades and klines - we need at least one of them
            sample_trades = await self._parquet_storage.read_trades_range(
                symbol, start_date_obj, end_date_obj
            )
            sample_klines = await self._parquet_storage.read_klines_range(
                symbol, start_date_obj, end_date_obj
            )
            
            logger.info(
                "data_availability_check",
                symbol=symbol,
                start_date=start_date_obj.isoformat(),
                end_date=end_date_obj.isoformat(),
                trades_count=len(sample_trades),
                klines_count=len(sample_klines),
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
            return {
                "start": start_date,
                "end": end_date,
            }
        except Exception as e:
            logger.warning(f"Error checking data availability: {e}")
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
                # Update progress
                progress = (i / total) * 100
                estimated_completion = datetime.now(timezone.utc) + timedelta(
                    seconds=(total - i) * 0.1
                )  # Rough estimate
                
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {"estimated_completion": estimated_completion},
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
    
    async def _compute_targets(
        self,
        features_df: pd.DataFrame,
        historical_data: Dict[str, pd.DataFrame],
        target_config: TargetConfig,
    ) -> pd.DataFrame:
        """Compute target variables."""
        if features_df.empty:
            return pd.DataFrame()
        
        # Merge with price data for target computation
        # Use klines for price data
        price_df = historical_data["klines"].copy()
        if price_df.empty:
            # Fallback to trades
            price_df = historical_data["trades"].copy()
            if not price_df.empty:
                price_df = price_df.groupby("timestamp")["price"].last().reset_index()
        
        if price_df.empty:
            return pd.DataFrame()
        
        # Merge features with prices
        merged = features_df.merge(
            price_df[["timestamp", "close"]].rename(columns={"close": "price"}),
            on="timestamp",
            how="left",
        )
        
        # Compute targets based on type
        if target_config.type == "regression":
            return self._compute_regression_targets(merged, target_config.horizon)
        elif target_config.type == "classification":
            threshold = target_config.threshold or 0.001  # Default 0.1%
            return self._compute_classification_targets(
                merged, target_config.horizon, threshold
            )
        else:  # risk_adjusted
            return self._compute_risk_adjusted_targets(merged, target_config.horizon)
    
    def _compute_regression_targets(
        self,
        data: pd.DataFrame,
        horizon: str,
    ) -> pd.DataFrame:
        """Compute regression targets (returns)."""
        # Map horizon to seconds
        horizon_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
        }[horizon]
        
        # Sort by timestamp
        data = data.sort_values("timestamp").copy()
        
        # Compute future price by time, not by index
        # Create future timestamps
        data["future_timestamp"] = data["timestamp"] + pd.Timedelta(seconds=horizon_seconds)
        
        # Create price lookup DataFrame
        price_lookup = data[["timestamp", "price"]].copy()
        price_lookup = price_lookup.sort_values("timestamp")
        
        # Use merge_asof to find nearest future price
        # direction='forward' ensures we only look forward in time
        data = pd.merge_asof(
            data.sort_values("future_timestamp"),
            price_lookup,
            left_on="future_timestamp",
            right_on="timestamp",
            direction="forward",
            suffixes=("", "_future"),
        )
        
        # Rename future price column
        data = data.rename(columns={"price_future": "future_price"})
        
        # Drop helper column
        data = data.drop(columns=["future_timestamp", "timestamp_future"], errors="ignore")
        
        # Compute return
        data["target"] = (data["future_price"] - data["price"]) / data["price"]
        
        # Remove rows where target cannot be computed
        data = data.dropna(subset=["target"])
        
        return data[["timestamp", "target"]]
    
    def _compute_classification_targets(
        self,
        data: pd.DataFrame,
        horizon: str,
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
        horizon: str,
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
        if features_df.empty or targets_df.empty:
            return False
        
        # Merge to check timestamps
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        if merged.empty:
            return False
        
        # For each row, verify:
        # - Features use only data <= timestamp
        # - Targets use only data > timestamp
        # This is a simplified check - in production would validate against Feature Registry
        
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
        
        # Split by periods
        train = merged[
            (merged["timestamp"] >= dataset["train_period_start"]) &
            (merged["timestamp"] < dataset["train_period_end"])
        ]
        
        validation = merged[
            (merged["timestamp"] >= dataset["validation_period_start"]) &
            (merged["timestamp"] < dataset["validation_period_end"])
        ]
        
        test = merged[
            (merged["timestamp"] >= dataset["test_period_start"]) &
            (merged["timestamp"] <= dataset["test_period_end"])
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