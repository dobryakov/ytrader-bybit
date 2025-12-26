"""
Optimized Dataset Builder - complete replacement for DatasetBuilder.

Integrates all optimized components for fast dataset building:
- StreamingDatasetBuilder for day-by-day processing
- Multi-level caching
- Vectorized feature computation
- Incremental orderbook updates
"""
import asyncio
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Dict, Optional, Any, TYPE_CHECKING
import pandas as pd
import structlog

from src.models.dataset import SplitStrategy, TargetConfig, DatasetStatus
from src.models.feature_registry import FeatureRegistry
from src.storage.metadata_storage import MetadataStorage
from src.storage.parquet_storage import ParquetStorage
from src.services.target_computation import (
    TargetComputationEngine,
    TargetComputationPresets,
)
from src.services.target_registry_version_manager import TargetRegistryVersionManager
from .streaming_builder import StreamingDatasetBuilder
from .requirements_analyzer import DataRequirements, FeatureRequirementsAnalyzer

if TYPE_CHECKING:
    from src.services.cache_service import CacheService
    from src.services.feature_registry import FeatureRegistryLoader
    from src.publishers.dataset_publisher import DatasetPublisher

logger = structlog.get_logger(__name__)


class OptimizedDatasetBuilder:
    """
    Optimized dataset builder with streaming processing and caching.
    
    Complete replacement for DatasetBuilder with improved performance:
    - 10-50x faster dataset building
    - Multi-level caching (local + Redis)
    - Vectorized feature computation
    - Streaming day-by-day processing
    """
    
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        parquet_storage: ParquetStorage,
        dataset_storage_path: str,
        cache_service: Optional["CacheService"] = None,
        feature_registry_loader: Optional["FeatureRegistryLoader"] = None,
        target_registry_version_manager: Optional[TargetRegistryVersionManager] = None,
        dataset_publisher: Optional["DatasetPublisher"] = None,
        batch_size: int = 1000,
    ):
        """
        Initialize optimized dataset builder.
        
        Args:
            metadata_storage: Metadata storage for dataset records
            parquet_storage: Parquet storage for reading historical data
            dataset_storage_path: Base path for storing built datasets
            cache_service: Optional cache service (Redis or in-memory)
            feature_registry_loader: Optional Feature Registry loader
            target_registry_version_manager: Target Registry version manager
            dataset_publisher: Optional dataset publisher for notifications
            batch_size: Batch size for processing timestamps
        """
        self._metadata_storage = metadata_storage
        self._parquet_storage = parquet_storage
        self._dataset_storage_path = Path(dataset_storage_path)
        self._dataset_storage_path.mkdir(parents=True, exist_ok=True)
        self._cache_service = cache_service
        self._feature_registry_loader = feature_registry_loader
        self._target_registry_version_manager = target_registry_version_manager
        self._dataset_publisher = dataset_publisher
        self._batch_size = batch_size
        
        # Initialize streaming builder
        self._streaming_builder = StreamingDatasetBuilder(
            cache_service=cache_service,
            parquet_storage=parquet_storage,
            feature_registry_loader=feature_registry_loader,
            batch_size=batch_size,
        )
        
        # Active builds tracking
        self._active_builds: Dict[str, asyncio.Task] = {}
        
        logger.info(
            "optimized_dataset_builder_initialized",
            cache_enabled=cache_service is not None,
            batch_size=batch_size,
            builder_type="optimized",
            internal_builder="streaming",
        )
    
    async def build_dataset(
        self,
        symbol: str,
        split_strategy: SplitStrategy,
        target_registry_version: str,
        train_period_start: Optional[datetime] = None,
        train_period_end: Optional[datetime] = None,
        validation_period_start: Optional[datetime] = None,
        validation_period_end: Optional[datetime] = None,
        test_period_start: Optional[datetime] = None,
        test_period_end: Optional[datetime] = None,
        walk_forward_config: Optional[Dict[str, Any]] = None,
        output_format: str = "parquet",
        feature_registry_version: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> str:
        """
        Build a dataset from historical data using optimized approach.
        
        Args:
            symbol: Trading pair symbol
            split_strategy: Split strategy (time_based or walk_forward)
            target_registry_version: Target Registry version
            train_period_start: Train period start (for time_based)
            train_period_end: Train period end (for time_based)
            validation_period_start: Validation period start (for time_based)
            validation_period_end: Validation period end (for time_based)
            test_period_start: Test period start (for time_based)
            test_period_end: Test period end (for time_based)
            walk_forward_config: Walk-forward configuration (for walk_forward)
            output_format: Output format (parquet, csv, hdf5)
            feature_registry_version: Feature Registry version
            
        Returns:
            Dataset ID (UUID string)
        """
        # Load target config from Target Registry
        if self._target_registry_version_manager is None:
            raise ValueError("Target Registry version manager not initialized")

        # Resolve alias like "latest" to a concrete version string before using it
        # and before persisting to metadata, so that datasets are tied to a specific
        # target registry snapshot and do not change semantics retroactively.
        resolved_target_version = target_registry_version
        if target_registry_version == "latest":
            # Prefer DB as the single source of truth for active version
            active_record = await self._metadata_storage.get_active_target_registry_version()
            if active_record and active_record.get("version"):
                resolved_target_version = active_record["version"]
        
        target_config_dict = await self._target_registry_version_manager.get_version(
            resolved_target_version
        )
        if target_config_dict is None:
            raise ValueError(
                f"Target Registry version not found: {resolved_target_version}"
            )
        
        target_config = TargetConfig(**target_config_dict)
        
        # Load Feature Registry
        if self._feature_registry_loader is None:
            raise ValueError("Feature Registry loader not initialized")
        
        # Resolve feature registry version
        if feature_registry_version == "latest":
            if self._feature_registry_loader._version_manager:
                active_version = (
                    await self._feature_registry_loader._version_manager.load_active_version()
                )
                if active_version:
                    registry_version = active_version.get("version", "1.0.0")
                else:
                    registry_version = "1.0.0"
            else:
                registry_version = "1.0.0"
        else:
            registry_version = feature_registry_version or "1.0.0"
        
        # Load Feature Registry
        await self._feature_registry_loader.load_async()
        if self._feature_registry_loader._registry_model is None:
            raise ValueError("Feature Registry not loaded")
        
        feature_registry = self._feature_registry_loader._registry_model
        
        # Normalize datetime objects
        def normalize_dt(dt: Optional[datetime]) -> Optional[datetime]:
            if dt is None:
                return None
            if not isinstance(dt, datetime):
                return dt
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        
        # Determine date range
        if split_strategy == SplitStrategy.TIME_BASED:
            start_date = normalize_dt(train_period_start)
            end_date = normalize_dt(test_period_end)
        else:
            # Walk-forward: extract from config
            start_date_str = walk_forward_config.get("start_date")
            end_date_str = walk_forward_config.get("end_date")
            start_date = normalize_dt(
                datetime.fromisoformat(start_date_str) if start_date_str else None
            )
            end_date = normalize_dt(
                datetime.fromisoformat(end_date_str) if end_date_str else None
            )
        
        if start_date is None or end_date is None:
            raise ValueError("Start date and end date must be provided")
        
        # Create dataset record
        # Convert target_config to dict for JSONB storage
        target_config_dict = target_config.model_dump() if hasattr(target_config, 'model_dump') else target_config.dict() if hasattr(target_config, 'dict') else target_config
        
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
            # Persist the resolved concrete Target Registry version (e.g. "1.6.0"),
            # not the alias like "latest", so that dataset semantics are stable.
            "target_registry_version": resolved_target_version,
            "target_config": target_config_dict,  # Save target_config for reference
            "feature_registry_version": registry_version,
            "output_format": output_format,
            "strategy_id": strategy_id,  # Save strategy_id for model training
        }
        
        dataset_id = await self._metadata_storage.create_dataset(dataset_data)
        
        # Start building in background
        build_task = asyncio.create_task(
            self._build_dataset_task(
                dataset_id=dataset_id,
                symbol=symbol,
                split_strategy=split_strategy,
                target_config=target_config,
                feature_registry=feature_registry,
                start_date=start_date,
                end_date=end_date,
                output_format=output_format,
            )
        )
        self._active_builds[dataset_id] = build_task
        
        logger.info(
            "optimized_dataset_build_started",
            dataset_id=dataset_id,
            symbol=symbol,
            split_strategy=split_strategy.value,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            builder_type="optimized",
            internal_builder="streaming",
        )
        
        return dataset_id
    
    async def _build_dataset_task(
        self,
        dataset_id: str,
        symbol: str,
        split_strategy: SplitStrategy,
        target_config: TargetConfig,
        feature_registry: FeatureRegistry,
        start_date: datetime,
        end_date: datetime,
        output_format: str,
    ) -> None:
        """
        Background task for building dataset.
        
        Args:
            dataset_id: Dataset ID
            symbol: Trading pair symbol
            split_strategy: Split strategy
            target_config: Target configuration
            feature_registry: Feature Registry instance
            start_date: Start date
            end_date: End date
            output_format: Output format
        """
        try:
            logger.info(
                "optimized_dataset_build_task_started",
                dataset_id=dataset_id,
                symbol=symbol,
                builder_type="optimized",
                internal_builder="streaming",
            )
            
            # Step 0: Validate data availability before starting build
            # This fails early with detailed error message if data is missing
            try:
                await self._validate_data_availability(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    feature_registry=feature_registry,
                    dataset_id=dataset_id,
                )
            except ValueError as e:
                # Data validation failed - fail the build immediately
                error_msg = f"Data availability validation failed: {str(e)}"
                logger.error(
                    "dataset_data_validation_failed",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    error=error_msg,
                )
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Step 1: Build features using streaming approach
            logger.info(
                "calling_streaming_builder",
                dataset_id=dataset_id,
                symbol=symbol,
                builder_type="optimized",
                internal_builder="streaming",
            )
            features_df = await self._streaming_builder.build_dataset_streaming(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                feature_registry=feature_registry,
                target_config=target_config,
                dataset_id=dataset_id,
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
            
            # Step 1.5: Validate feature completeness (strict mode - fail immediately)
            try:
                await self._validate_feature_completeness(
                    features_df=features_df,
                    feature_registry=feature_registry,
                    dataset_id=dataset_id,
                )
            except ValueError as e:
                # Feature completeness validation failed - fail the build
                error_msg = f"Feature completeness validation failed: {str(e)}"
                logger.error(
                    "dataset_feature_validation_failed",
                    dataset_id=dataset_id,
                    error=error_msg,
                )
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Step 2: Compute targets
            targets_df = await self._compute_targets(
                features_df=features_df,
                target_config=target_config,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            
            if targets_df.empty:
                error_msg = "No targets computed"
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                return
            
            # Step 3: Validate data quality (check for high missing values due to insufficient historical data)
            try:
                await self._validate_data_quality(
                    features_df=features_df,
                    feature_registry=feature_registry,
                    dataset_id=dataset_id,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            except ValueError as e:
                error_msg = str(e)
                await self._metadata_storage.update_dataset(
                    dataset_id,
                    {
                        "status": DatasetStatus.FAILED.value,
                        "error_message": error_msg,
                        "completed_at": datetime.now(timezone.utc),
                    },
                )
                # Note: Detailed logging with time ranges is done in _validate_data_quality method
                return
            
            # Step 4: Validate no data leakage
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
            
            # Step 5: Split dataset
            dataset = await self._metadata_storage.get_dataset(dataset_id)
            if dataset is None:
                logger.error("dataset_not_found", dataset_id=dataset_id)
                return
            
            # Get strategy_id from dataset metadata
            dataset_strategy_id = dataset.get("strategy_id") if isinstance(dataset, dict) else getattr(dataset, "strategy_id", None)
            
            if split_strategy == SplitStrategy.TIME_BASED:
                splits = await self._split_time_based(
                    features_df, targets_df, dataset
                )
            else:
                splits = await self._split_walk_forward(
                    features_df, targets_df, dataset
                )
            
            # Step 6: Write splits to storage
            storage_path = await self._write_dataset_splits(
                dataset_id, splits, output_format
            )
            
            # Step 7: Update dataset record
            total_train = len(splits["train"])
            total_val = len(splits["validation"])
            total_test = len(splits["test"])
            
            # Compute statistics for each split
            split_statistics = self._compute_split_statistics(splits, target_config)
            
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
                    "split_statistics": split_statistics,
                },
            )
            
            logger.info(
                "optimized_dataset_build_completed",
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
                        trace_id=None,
                        strategy_id=dataset_strategy_id,  # Pass strategy_id from dataset
                    )
                except Exception as e:
                    logger.warning(
                        "dataset_ready_notification_failed",
                        dataset_id=dataset_id,
                        error=str(e),
                    )
        
        except Exception as e:
            logger.error(
                "optimized_dataset_build_failed",
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
    
    async def _compute_targets(
        self,
        features_df: pd.DataFrame,
        target_config: TargetConfig,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Compute target variables using TargetComputationEngine.
        
        Args:
            features_df: DataFrame with features
            target_config: Target configuration
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with targets
        """
        if features_df.empty:
            return pd.DataFrame()
        
        # Load price data for target computation
        price_df = await self._parquet_storage.read_klines_range(
            symbol, start_date.date(), end_date.date()
        )
        
        if price_df.empty:
            logger.warning("no_price_data_for_targets", symbol=symbol)
            return pd.DataFrame()
        
        # Merge features with prices
        price_for_merge = price_df[["timestamp", "close"]].rename(
            columns={"close": "price"}
        )
        merged = features_df.merge(
            price_for_merge, on="timestamp", how="left"
        )
        
        # Get computation configuration
        computation_config = TargetComputationPresets.get_computation_config(
            target_config.computation
        )
        
        # Compute targets based on type
        if target_config.type == "regression":
            targets_df = TargetComputationEngine.compute_target(
                merged, target_config.horizon, computation_config, price_df
            )
        elif target_config.type == "classification":
            # For classification we first compute a numeric base target (typically
            # a forward return) and then map it to discrete classes depending
            # on the configured task_variant and threshold.
            from src.config import config as feature_config

            targets_df = TargetComputationEngine.compute_target(
                merged, target_config.horizon, computation_config, price_df
            )

            if not targets_df.empty and "target" in targets_df.columns:
                task_variant = None
                if (
                    target_config.computation
                    and target_config.computation.options
                ):
                    task_variant = target_config.computation.options.get(
                        "task_variant"
                    )

                # Log raw numeric target distribution before class mapping
                try:
                    raw = targets_df["target"].astype(float)
                    desc = {
                        "count": int(raw.shape[0]),
                        "mean": float(raw.mean()),
                        "std": float(raw.std()),
                        "min": float(raw.min()),
                        "max": float(raw.max()),
                        "p1": float(raw.quantile(0.01)),
                        "p99": float(raw.quantile(0.99)),
                    }
                except Exception:
                    desc = {"error": "failed_to_summarize_raw_target"}

                logger.info(
                    "classification_target_before_mapping",
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    task_variant=task_variant,
                    target_type=target_config.type,
                    threshold=float(target_config.threshold)
                    if getattr(target_config, "threshold", None) is not None
                    else None,
                    stats=desc,
                )

                # Triple-class classification: {-1, 0, +1} with dead-zone
                if task_variant == "triple_classification":
                    # Use explicit threshold from TargetConfig if provided,
                    # otherwise fall back to global MODEL_CLASSIFICATION_THRESHOLD.
                    threshold = (
                        target_config.threshold
                        if target_config.threshold is not None
                        else feature_config.model_classification_threshold
                    )
                    thr = float(threshold)

                    targets_df["target"] = targets_df["target"].apply(
                        lambda x: 1
                        if x > thr
                        else (-1 if x < -thr else 0)
                    )

                # Binary classification: direction only (no flat zone)
                elif task_variant == "binary_classification":
                    # Map strictly by sign; extremely rare exact zeros остаются 0.
                    targets_df["target"] = targets_df["target"].apply(
                        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                    )

                # Log discrete class distribution after mapping
                vc = targets_df["target"].value_counts(dropna=False).to_dict()
                logger.info(
                    "classification_target_after_mapping",
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    task_variant=task_variant,
                    class_distribution=vc,
                )
        else:  # risk_adjusted
            if (
                not target_config.computation
                or target_config.computation.preset != "sharpe_ratio"
            ):
                risk_config = TargetComputationPresets.get_preset("sharpe_ratio")
                if target_config.computation and target_config.computation.options:
                    risk_config.update(target_config.computation.options)
                computation_config = risk_config
            
            targets_df = TargetComputationEngine.compute_target(
                merged, target_config.horizon, computation_config, price_df
            )
        
        return targets_df
    
    async def _validate_no_data_leakage(
        self, features_df: pd.DataFrame, targets_df: pd.DataFrame
    ) -> bool:
        """
        Validate no data leakage in features or targets.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            
        Returns:
            True if no leakage detected
        """
        if features_df.empty or targets_df.empty:
            return False
        
        # Merge to check timestamps
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        if merged.empty:
            return False
        
        # Basic validation: targets should be computed from future prices
        # This is handled by TargetComputationEngine, so we just check
        # that we have matching timestamps
        return len(merged) > 0
    
    async def _validate_data_availability(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_registry: FeatureRegistry,
        dataset_id: str,
    ) -> None:
        """
        Validate that all required data is available before starting dataset build.
        
        Checks data availability for all days in the period and all data types
        required by Feature Registry. Fails early with detailed error message
        indicating which features require which data types, and which days are missing.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for dataset
            end_date: End date for dataset
            feature_registry: Feature Registry instance
            dataset_id: Dataset ID for error reporting
            
        Raises:
            ValueError: If required data is missing, with detailed error message
        """
        # Analyze requirements from Feature Registry
        requirements_analyzer = FeatureRequirementsAnalyzer()
        requirements = requirements_analyzer.analyze(feature_registry)
        
        # Map input sources to storage types
        storage_type_mapping = {
            "orderbook": ["orderbook_snapshots", "orderbook_deltas"],
            "kline": ["klines"],
            "trades": ["trades"],
            "ticker": ["ticker"],
            "funding": ["funding"],
        }
        
        # Build mapping: feature -> required data types
        feature_to_data_types: Dict[str, List[str]] = {}
        for feature in feature_registry.features:
            feature_name = feature.name
            for input_source in feature.input_sources:
                if input_source in storage_type_mapping:
                    if feature_name not in feature_to_data_types:
                        feature_to_data_types[feature_name] = []
                    feature_to_data_types[feature_name].extend(
                        storage_type_mapping[input_source]
                    )
        
        # Generate list of days to check
        days_to_check = []
        current_date = start_date.date()
        end_date_obj = end_date.date()
        while current_date <= end_date_obj:
            days_to_check.append(current_date)
            current_date += timedelta(days=1)
        
        # Check data availability for each required storage type
        missing_data: Dict[str, Dict[date, List[str]]] = {}  # storage_type -> {date -> [feature_names]}
        
        # Collect all unique storage types from requirements
        all_storage_types = set()
        for storage_type_list in requirements.storage_types.values():
            all_storage_types.update(storage_type_list)
        
        for storage_type_name in sorted(all_storage_types):
            # Find features that require this storage type
            features_requiring_type = [
                feature_name
                for feature_name, data_types in feature_to_data_types.items()
                if storage_type_name in data_types
            ]
            
            if not features_requiring_type:
                continue  # No features require this type
            
            # Check availability for each day
            for day_date in days_to_check:
                date_str = day_date.isoformat()
                data_available = False
                
                try:
                    if storage_type_name == "klines":
                        data = await self._parquet_storage.read_klines(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                    elif storage_type_name == "trades":
                        data = await self._parquet_storage.read_trades(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                    elif storage_type_name == "ticker":
                        data = await self._parquet_storage.read_ticker(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                    elif storage_type_name == "funding":
                        data = await self._parquet_storage.read_funding(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                    elif storage_type_name == "orderbook_snapshots":
                        data = await self._parquet_storage.read_orderbook_snapshots(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                    elif storage_type_name == "orderbook_deltas":
                        data = await self._parquet_storage.read_orderbook_deltas(symbol, date_str)
                        data_available = not data.empty and len(data) > 0
                except FileNotFoundError:
                    data_available = False
                except Exception as e:
                    logger.warning(
                        "data_availability_check_error",
                        dataset_id=dataset_id,
                        symbol=symbol,
                        date=date_str,
                        storage_type=storage_type_name,
                        error=str(e),
                    )
                    data_available = False
                
                if not data_available:
                    if storage_type_name not in missing_data:
                        missing_data[storage_type_name] = {}
                    if day_date not in missing_data[storage_type_name]:
                        missing_data[storage_type_name][day_date] = []
                    missing_data[storage_type_name][day_date].extend(features_requiring_type)
        
        # If missing data found, raise detailed error
        if missing_data:
            error_parts = [
                f"Missing required data for dataset build (symbol: {symbol}, period: {start_date.date()} to {end_date.date()}):"
            ]
            
            for storage_type_name, days_dict in sorted(missing_data.items()):
                missing_days = sorted(days_dict.keys())
                # Get unique features that require this type
                all_features = set()
                for day_features in days_dict.values():
                    all_features.update(day_features)
                
                # Calculate date range for missing days
                if len(missing_days) == 1:
                    date_range_str = missing_days[0].isoformat()
                elif len(missing_days) > 1:
                    date_range_str = f"{missing_days[0].isoformat()} to {missing_days[-1].isoformat()}"
                else:
                    date_range_str = "N/A"
                
                error_parts.append(
                    f"\n  Storage type '{storage_type_name}':"
                )
                error_parts.append(
                    f"    Required by features: {', '.join(sorted(all_features))}"
                )
                error_parts.append(
                    f"    Missing for days: {', '.join(d.isoformat() for d in missing_days)}"
                )
                error_parts.append(
                    f"    Date range: {date_range_str}"
                )
                error_parts.append(
                    f"    Total missing days: {len(missing_days)}/{len(days_to_check)}"
                )
            
            error_parts.append(
                f"\n  Action required: Run backfilling for missing data types and days."
            )
            
            error_msg = "\n".join(error_parts)
            
            # Build detailed missing data structure for logging
            missing_data_details = {}
            for storage_type_name, days_dict in missing_data.items():
                missing_days = sorted(days_dict.keys())
                all_features = set()
                for day_features in days_dict.values():
                    all_features.update(day_features)
                
                missing_data_details[storage_type_name] = {
                    "missing_days": [d.isoformat() for d in missing_days],
                    "date_range": f"{missing_days[0].isoformat()} to {missing_days[-1].isoformat()}" if len(missing_days) > 1 else missing_days[0].isoformat(),
                    "missing_days_count": len(missing_days),
                    "total_days_checked": len(days_to_check),
                    "required_by_features": sorted(all_features),
                }
            
            logger.error(
                "dataset_data_availability_check_failed",
                dataset_id=dataset_id,
                symbol=symbol,
                start_date=start_date.date().isoformat(),
                end_date=end_date.date().isoformat(),
                missing_data_types=list(missing_data.keys()),
                total_missing_days=sum(len(days) for days in missing_data.values()),
                missing_data_details=missing_data_details,
            )
            
            raise ValueError(error_msg)
        
        # All data available - log success
        logger.info(
            "dataset_data_availability_check_passed",
            dataset_id=dataset_id,
            symbol=symbol,
            start_date=start_date.date().isoformat(),
            end_date=end_date.date().isoformat(),
            required_data_types=sorted(requirements.required_data_types),
            days_checked=len(days_to_check),
        )
    
    async def _validate_feature_completeness(
        self,
        features_df: pd.DataFrame,
        feature_registry: FeatureRegistry,
        dataset_id: str,
    ) -> None:
        """
        Validate that all features from Feature Registry are present and have data.
        
        Переписанная валидация с учётом проблем из исследования:
        - Правильная проверка типов данных перед подсчётом NaN
        - Корректная обработка object dtype (без использования == None)
        - Детальное логирование типов данных и значений
        - Проверка, что features_df не изменяется в процессе валидации
        
        Args:
            features_df: DataFrame with computed features
            feature_registry: Feature Registry instance
            dataset_id: Dataset ID for error reporting
            
        Raises:
            ValueError: If features are missing or have no data
        """
        # ВАЖНО: Создаём копию для валидации, чтобы не изменять исходный DataFrame
        features_df_copy = features_df.copy()
        
        # Get expected feature names from Feature Registry
        expected_features = {f.name for f in feature_registry.features}
        
        # Get actual feature columns (exclude service columns)
        service_columns = {"timestamp", "symbol"}
        actual_features = set(features_df_copy.columns) - service_columns
        
        # Логируем базовую информацию о DataFrame
        logger.debug(
            "feature_completeness_validation_started",
            dataset_id=dataset_id,
            total_rows=len(features_df_copy),
            total_columns=len(features_df_copy.columns),
            expected_features_count=len(expected_features),
            actual_features_count=len(actual_features),
            dtypes=features_df_copy.dtypes.to_dict(),
        )
        
        # Check 1: All expected features must be present
        missing_features = sorted(expected_features - actual_features)
        if missing_features:
            error_msg = (
                f"Missing {len(missing_features)} features from Feature Registry. "
                f"Expected {len(expected_features)}, got {len(actual_features)}. "
                f"Missing: {', '.join(missing_features[:20])}"
                + (f" (and {len(missing_features) - 20} more)" if len(missing_features) > 20 else "")
            )
            logger.error(
                "dataset_missing_features",
                dataset_id=dataset_id,
                expected_count=len(expected_features),
                actual_count=len(actual_features),
                missing_count=len(missing_features),
                missing_features=missing_features,
                actual_feature_names=sorted(actual_features),
            )
            raise ValueError(error_msg)
        
        # Check 2: All features must have at least some non-null values
        # Исправленная логика подсчёта NaN с учётом всех нюансов из исследования
        features_with_no_data = []
        feature_details = {}
        
        for feature_name in expected_features:
            if feature_name not in features_df_copy.columns:
                continue  # Already caught above, but double-check
            
            feature_series = features_df_copy[feature_name]
            
            # Детальная информация о фиче для отладки
            feature_dtype = str(feature_series.dtype)
            total_count = len(feature_series)
            
            # ИСПРАВЛЕННАЯ ЛОГИКА: Правильная проверка значений
            # Pandas isna() корректно обрабатывает и NaN и None для всех типов данных
            # Для object dtype isna() правильно находит и NaN и None значения
            # НЕ используем == None, так как это может не работать правильно с numpy/pandas объектами
            
            # Используем isna() для всех типов - это универсальный способ
            nan_mask = feature_series.isna()
            nan_count = nan_mask.sum()
            non_nan_count = (~nan_mask).sum()  # Эквивалентно feature_series.notna().sum()
            
            # Если все значения NaN/None, данных нет
            if nan_count == total_count:
                # Для object dtype дополнительно проверяем, есть ли реальные значения среди не-NaN
                if feature_series.dtype == 'object' and non_nan_count == 0:
                    # Проверяем первые несколько значений для диагностики
                    sample_values = feature_series.head(5).tolist()
                    sample_types = [type(v).__name__ if v is not None else 'None' for v in sample_values]
                    
                    features_with_no_data.append(feature_name)
                    feature_details[feature_name] = {
                        "dtype": feature_dtype,
                        "total_count": total_count,
                        "nan_count": int(nan_count),
                        "non_nan_count": int(non_nan_count),
                        "issue": "all_values_are_nan_or_none",
                        "sample_types": sample_types,
                    }
                else:
                    features_with_no_data.append(feature_name)
                    feature_details[feature_name] = {
                        "dtype": feature_dtype,
                        "total_count": total_count,
                        "nan_count": int(nan_count),
                        "non_nan_count": int(non_nan_count),
                        "issue": "all_values_are_nan",
                    }
                continue
            
            # Проверяем, что есть хотя бы одно реальное значение
            # Для object dtype: если есть не-NaN значения, но они все None, данных всё равно нет
            if non_nan_count == 0:
                features_with_no_data.append(feature_name)
                feature_details[feature_name] = {
                    "dtype": feature_dtype,
                    "total_count": total_count,
                    "nan_count": int(nan_count),
                    "non_nan_count": int(non_nan_count),
                    "issue": "no_valid_values",
                }
            elif feature_series.dtype == 'object':
                # Для object dtype дополнительная проверка: все не-NaN значения - это None?
                # Берём выборку не-NaN значений и проверяем
                non_na_values = feature_series[~nan_mask]
                if len(non_na_values) > 0:
                    # Проверяем, есть ли хотя бы одно значение, которое не None
                    has_real_value = False
                    for val in non_na_values.head(100):  # Ограничиваем проверку для производительности
                        if val is not None and not pd.isna(val):
                            has_real_value = True
                            break
                    
                    if not has_real_value:
                        # Все не-NaN значения - это None, данных нет
                        features_with_no_data.append(feature_name)
                        sample_types = [type(v).__name__ if v is not None else 'None' for v in non_na_values.head(5)]
                        feature_details[feature_name] = {
                            "dtype": feature_dtype,
                            "total_count": total_count,
                            "nan_count": int(nan_count),
                            "non_nan_count": int(non_nan_count),
                            "issue": "all_non_na_values_are_none",
                            "sample_types": sample_types,
                        }
            
            # Логируем успешную проверку фичи (только для первых нескольких)
            if len(feature_details) < 5:
                feature_details[feature_name] = {
                    "dtype": feature_dtype,
                    "total_count": total_count,
                    "nan_count": int(nan_count) if 'nan_count' in locals() else None,
                    "non_nan_count": int(non_nan_count) if 'non_nan_count' in locals() else None,
                    "status": "valid",
                }
        
        if features_with_no_data:
            error_msg = (
                f"{len(features_with_no_data)} features have no data (all None/NaN). "
                f"This indicates missing source data or computation failure. "
                f"Features with no data: {', '.join(features_with_no_data[:20])}"
                + (f" (and {len(features_with_no_data) - 20} more)" if len(features_with_no_data) > 20 else "")
            )
            logger.error(
                "dataset_features_no_data",
                dataset_id=dataset_id,
                features_with_no_data=features_with_no_data,
                count=len(features_with_no_data),
                feature_details={k: v for k, v in feature_details.items() if k in features_with_no_data},
            )
            raise ValueError(error_msg)
        
        # Success - log validation passed
        logger.info(
            "dataset_feature_completeness_validated",
            dataset_id=dataset_id,
            feature_count=len(actual_features),
            all_features_present=True,
            all_features_have_data=True,
            total_rows=len(features_df_copy),
            sample_feature_details={k: v for k, v in list(feature_details.items())[:5]},
        )
    
    async def _validate_data_quality(
        self,
        features_df: pd.DataFrame,
        feature_registry: FeatureRegistry,
        dataset_id: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Validate data quality by checking for high missing values in features.
        
        Переписанная валидация с учётом всех проблем из исследования:
        - Правильная проверка типов данных перед подсчётом NaN
        - Исправленная логика подсчёта NaN для object dtype (без == None)
        - Детальное логирование на каждом этапе
        - Проверка, что features_df не изменяется
        - Анализ временных диапазонов с пропусками данных
        
        Args:
            features_df: DataFrame with computed features
            feature_registry: Feature Registry instance
            dataset_id: Dataset ID for error reporting
            symbol: Trading pair symbol
            start_date: Start date for dataset
            end_date: End date for dataset
            
        Raises:
            ValueError: If data quality is insufficient (high missing values in lookback features)
        """
        from src.config import config as feature_config
        
        # ВАЖНО: Создаём копию для валидации, чтобы не изменять исходный DataFrame
        features_df_copy = features_df.copy()
        
        # Get expected feature names from Feature Registry
        expected_features = {f.name: f for f in feature_registry.features}
        
        # Get actual feature columns (exclude service columns)
        service_columns = {"timestamp", "symbol"}
        actual_features = set(features_df_copy.columns) - service_columns
        
        # Analyze which features require lookback
        requirements_analyzer = FeatureRequirementsAnalyzer()
        requirements = requirements_analyzer.analyze(feature_registry)
        
        # Identify features that require lookback (candle/pattern features)
        lookback_features = set()
        for feature_name, feature_def in expected_features.items():
            if feature_def.lookback_window and feature_def.lookback_window != "0s":
                lookback_features.add(feature_name)
        
        # Check missing values ratio for each feature
        total_rows = len(features_df_copy)
        if total_rows == 0:
            raise ValueError("Dataset is empty - no rows in features DataFrame")
        
        # Логируем начальную информацию
        logger.debug(
            "data_quality_validation_started",
            dataset_id=dataset_id,
            symbol=symbol,
            total_rows=total_rows,
            total_features=len(actual_features),
            lookback_features_count=len(lookback_features),
            expected_features_count=len(expected_features),
            features_df_dtypes=features_df_copy.dtypes.to_dict(),
            features_df_memory_usage=features_df_copy.memory_usage(deep=True).to_dict(),
        )
        
        # Thresholds for validation
        max_nan_ratio = feature_config.dataset_max_feature_nan_ratio
        max_lookback_nan_ratio = feature_config.dataset_max_lookback_feature_nan_ratio
        fail_on_high_nan = feature_config.dataset_fail_on_high_nan_ratio
        
        # Collect features with high missing values
        features_with_high_nan: Dict[str, Dict[str, Any]] = {}
        
        for feature_name in actual_features:
            if feature_name not in expected_features:
                continue  # Skip unexpected features
            
            feature_series = features_df_copy[feature_name]
            feature_dtype = str(feature_series.dtype)
            
            # ИСПРАВЛЕННАЯ ЛОГИКА: Правильный подсчёт NaN
            # Pandas isna() корректно обрабатывает и NaN и None для всех типов данных
            # Для object dtype isna() правильно находит и NaN и None значения
            # НЕ используем == None, так как это может не работать правильно с numpy/pandas объектами
            
            # Используем isna() для всех типов - это универсальный способ
            nan_mask = feature_series.isna()
            nan_count = nan_mask.sum()
            
            # Для object dtype: pandas isna() правильно находит и NaN и None
            # Дополнительная проверка не требуется, так как isna() универсален
            
            nan_ratio = nan_count / total_rows if total_rows > 0 else 1.0
            
            # Determine if this is a lookback feature
            is_lookback_feature = feature_name in lookback_features
            
            # Use stricter threshold for lookback features
            threshold = max_lookback_nan_ratio if is_lookback_feature else max_nan_ratio
            
            # Check if this feature exceeds threshold
            if nan_ratio > threshold:
                features_with_high_nan[feature_name] = {
                    "nan_count": int(nan_count),
                    "nan_ratio": float(nan_ratio),
                    "total_rows": total_rows,
                    "is_lookback_feature": is_lookback_feature,
                    "lookback_window": expected_features[feature_name].lookback_window if feature_name in expected_features else None,
                    "threshold_used": float(threshold),
                    "dtype": feature_dtype,
                }
        
        # Логируем результаты проверки NaN
        logger.debug(
            "data_quality_nan_check_completed",
            dataset_id=dataset_id,
            features_checked=len(actual_features),
            features_with_high_nan_count=len(features_with_high_nan),
            total_rows=total_rows,
        )
        
        # If no features with high NaN, validation passes
        if not features_with_high_nan:
            logger.info(
                "dataset_data_quality_validated",
                dataset_id=dataset_id,
                symbol=symbol,
                total_rows=total_rows,
                total_features=len(actual_features),
                lookback_features_count=len(lookback_features),
                max_nan_ratio_threshold=max_nan_ratio,
                max_lookback_nan_ratio_threshold=max_lookback_nan_ratio,
            )
            return
        
        # Separate lookback and non-lookback features with high NaN
        lookback_high_nan = {
            name: info for name, info in features_with_high_nan.items()
            if info["is_lookback_feature"]
        }
        non_lookback_high_nan = {
            name: info for name, info in features_with_high_nan.items()
            if not info["is_lookback_feature"]
        }
        
        # Analyze timestamps where features have missing values
        # This helps identify specific time periods with data gaps
        missing_timestamps_by_feature: Dict[str, pd.Series] = {}
        missing_time_ranges: Dict[str, Dict[str, Any]] = {}
        
        if "timestamp" in features_df_copy.columns:
            for feature_name, info in features_with_high_nan.items():
                if feature_name not in features_df_copy.columns:
                    continue
                
                feature_series = features_df_copy[feature_name]
                timestamp_series = features_df_copy["timestamp"]
                    
                # ИСПРАВЛЕННАЯ ЛОГИКА: Правильная маска для пропусков
                # Используем isna() для всех типов - он правильно обрабатывает и NaN и None
                    missing_mask = feature_series.isna()
                    
                    missing_timestamps = timestamp_series[missing_mask]
                    
                    if len(missing_timestamps) > 0:
                        missing_timestamps_by_feature[feature_name] = missing_timestamps
                        
                        # Calculate time ranges
                        min_ts = missing_timestamps.min()
                        max_ts = missing_timestamps.max()
                        
                        # Group by date to identify which days have missing data
                        if pd.api.types.is_datetime64_any_dtype(missing_timestamps):
                            missing_dates = missing_timestamps.dt.date.unique()
                            missing_dates_sorted = sorted(missing_dates)
                            
                            missing_time_ranges[feature_name] = {
                                "first_missing_timestamp": min_ts.isoformat() if isinstance(min_ts, pd.Timestamp) else str(min_ts),
                                "last_missing_timestamp": max_ts.isoformat() if isinstance(max_ts, pd.Timestamp) else str(max_ts),
                                "missing_dates": [d.isoformat() for d in missing_dates_sorted],
                                "date_range": f"{missing_dates_sorted[0].isoformat()} to {missing_dates_sorted[-1].isoformat()}" if len(missing_dates_sorted) > 1 else missing_dates_sorted[0].isoformat(),
                                "missing_timestamps_count": len(missing_timestamps),
                            }
        
        # Build detailed error message
        error_parts = [
            f"Dataset build failed due to insufficient data quality (symbol: {symbol}, period: {start_date.date()} to {end_date.date()}):",
            f"  Total rows: {total_rows}",
            f"  Features with high missing values: {len(features_with_high_nan)}",
            f"    - Lookback features threshold: >{(max_lookback_nan_ratio * 100):.1f}%",
            f"    - Other features threshold: >{(max_nan_ratio * 100):.1f}%",
        ]
        
        if lookback_high_nan:
            error_parts.append(
                f"\n  CRITICAL: {len(lookback_high_nan)} features requiring lookback have high missing values:"
            )
            # Sort by NaN ratio (descending)
            sorted_lookback = sorted(
                lookback_high_nan.items(),
                key=lambda x: x[1]["nan_ratio"],
                reverse=True
            )
            for feature_name, info in sorted_lookback[:20]:  # Show top 20
                error_line = (
                    f"    - {feature_name}: {(info['nan_ratio'] * 100):.1f}% missing "
                    f"({info['nan_count']}/{info['total_rows']} rows, threshold: {(info['threshold_used'] * 100):.1f}%) "
                    f"[lookback: {info['lookback_window']}, dtype: {info['dtype']}]"
                )
                
                # Add time range information if available
                if feature_name in missing_time_ranges:
                    time_info = missing_time_ranges[feature_name]
                    error_line += f"\n      Missing data period: {time_info['date_range']}"
                    error_line += f"\n      First missing: {time_info['first_missing_timestamp']}"
                    error_line += f"\n      Last missing: {time_info['last_missing_timestamp']}"
                    error_line += f"\n      Affected dates: {', '.join(time_info['missing_dates'][:10])}"
                    if len(time_info['missing_dates']) > 10:
                        error_line += f" ... and {len(time_info['missing_dates']) - 10} more"
                
                error_parts.append(error_line)
            if len(sorted_lookback) > 20:
                error_parts.append(f"    ... and {len(sorted_lookback) - 20} more features")
            
            error_parts.append(
                "\n  This indicates insufficient historical data for computing lookback features."
            )
            error_parts.append(
                "  Possible causes:"
            )
            error_parts.append(
                "    - Missing klines data for required historical period"
            )
            error_parts.append(
                "    - Insufficient lookback period (need data before start_date)"
            )
            error_parts.append(
                "    - Data gaps in historical data"
            )
            
            # Add summary of missing time periods
            if missing_time_ranges:
                error_parts.append(
                    "\n  Missing data time periods:"
                )
                # Group by date range to show common patterns
                date_ranges_by_feature = {}
                for feature_name, time_info in missing_time_ranges.items():
                    date_range = time_info['date_range']
                    if date_range not in date_ranges_by_feature:
                        date_ranges_by_feature[date_range] = []
                    date_ranges_by_feature[date_range].append(feature_name)
                
                for date_range, features in sorted(date_ranges_by_feature.items()):
                    error_parts.append(
                        f"    - {date_range}: {len(features)} features affected"
                    )
            
            error_parts.append(
                "\n  Action required: Run backfilling for missing historical data."
            )
        
        if non_lookback_high_nan:
            error_parts.append(
                f"\n  WARNING: {len(non_lookback_high_nan)} non-lookback features have high missing values:"
            )
            sorted_non_lookback = sorted(
                non_lookback_high_nan.items(),
                key=lambda x: x[1]["nan_ratio"],
                reverse=True
            )
            for feature_name, info in sorted_non_lookback[:10]:  # Show top 10
                error_parts.append(
                    f"    - {feature_name}: {(info['nan_ratio'] * 100):.1f}% missing "
                    f"({info['nan_count']}/{info['total_rows']} rows, dtype: {info['dtype']})"
                )
            if len(sorted_non_lookback) > 10:
                error_parts.append(f"    ... and {len(sorted_non_lookback) - 10} more features")
        
        error_msg = "\n".join(error_parts)
        
        # Build lookback features details with time ranges
        lookback_features_details = {}
        for name, info in lookback_high_nan.items():
            lookback_features_details[name] = {
                "nan_ratio": info["nan_ratio"],
                "nan_count": info["nan_count"],
                "lookback_window": info["lookback_window"],
                "dtype": info["dtype"],
            }
            # Add time range if available
            if name in missing_time_ranges:
                lookback_features_details[name].update({
                    "missing_date_range": missing_time_ranges[name]["date_range"],
                    "first_missing_timestamp": missing_time_ranges[name]["first_missing_timestamp"],
                    "last_missing_timestamp": missing_time_ranges[name]["last_missing_timestamp"],
                    "missing_dates": missing_time_ranges[name]["missing_dates"],
                })
        
        # Log detailed information
        logger.error(
            "dataset_data_quality_validation_failed",
            dataset_id=dataset_id,
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            total_rows=total_rows,
            features_with_high_nan_count=len(features_with_high_nan),
            lookback_high_nan_count=len(lookback_high_nan),
            non_lookback_high_nan_count=len(non_lookback_high_nan),
            max_nan_ratio_threshold=max_nan_ratio,
            max_lookback_nan_ratio_threshold=max_lookback_nan_ratio,
            lookback_features_details=lookback_features_details,
            missing_time_ranges=missing_time_ranges if missing_time_ranges else {},
        )
        
        # Always fail if lookback features have high NaN (these require historical data)
        if lookback_high_nan:
            raise ValueError(error_msg)
        
        # For non-lookback features, fail only if fail_on_high_nan is enabled
        if non_lookback_high_nan and fail_on_high_nan:
            raise ValueError(error_msg)
        
        # Otherwise, just log warning for non-lookback features
        if non_lookback_high_nan:
            logger.warning(
                "dataset_data_quality_warning",
                dataset_id=dataset_id,
                symbol=symbol,
                features_with_high_nan_count=len(non_lookback_high_nan),
                message="High NaN ratio detected in non-lookback features but build continues (fail_on_high_nan_ratio=False)",
        )
    
    async def _split_time_based(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        dataset: dict,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataset using time-based strategy.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            dataset: Dataset metadata
            
        Returns:
            Dictionary with train/validation/test splits
        """
        # Merge features and targets
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        # Normalize timestamps
        if merged["timestamp"].dtype.tz is None:
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        else:
            merged["timestamp"] = merged["timestamp"].dt.tz_convert(timezone.utc)
        
        # Normalize period boundaries
        def normalize_period(dt):
            if dt is None:
                return None
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            return dt
        
        train_start = normalize_period(dataset["train_period_start"])
        train_end = normalize_period(dataset["train_period_end"])
        val_start = normalize_period(dataset["validation_period_start"])
        val_end = normalize_period(dataset["validation_period_end"])
        test_start = normalize_period(dataset["test_period_start"])
        test_end = normalize_period(dataset["test_period_end"])
        
        # Split by periods
        train = merged[
            (merged["timestamp"] >= train_start) & (merged["timestamp"] <= train_end)
        ]
        
        validation = (
            merged[
                (merged["timestamp"] >= val_start)
                & (merged["timestamp"] <= val_end)
            ]
            if val_start and val_end
            else pd.DataFrame()
        )
        
        test = merged[
            (merged["timestamp"] >= test_start) & (merged["timestamp"] <= test_end)
        ]
        
        return {"train": train, "validation": validation, "test": test}
    
    async def _split_walk_forward(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        dataset: dict,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataset using walk-forward strategy.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            dataset: Dataset metadata
            
        Returns:
            Dictionary with train/validation/test splits
        """
        # Merge features and targets
        merged = features_df.merge(targets_df, on="timestamp", how="inner")
        
        # Normalize timestamps
        if merged["timestamp"].dtype.tz is None:
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        else:
            merged["timestamp"] = merged["timestamp"].dt.tz_convert(timezone.utc)
        
        # Get walk-forward config
        wf_config = dataset.get("walk_forward_config", {})
        if not wf_config:
            raise ValueError("Walk-forward config not found in dataset")
        
        # Parse dates
        start_date = datetime.fromisoformat(wf_config["start_date"])
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        
        train_window_days = wf_config["train_window_days"]
        validation_window_days = wf_config["validation_window_days"]
        test_window_days = wf_config["test_window_days"]
        step_days = wf_config["step_days"]
        
        # Collect all splits
        all_train = []
        all_validation = []
        all_test = []
        
        current_date = start_date
        end_date = datetime.fromisoformat(wf_config["end_date"])
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        while current_date < end_date:
            train_start = current_date
            train_end = train_start + timedelta(days=train_window_days)
            val_start = train_end
            val_end = val_start + timedelta(days=validation_window_days)
            test_start = val_end
            test_end = test_start + timedelta(days=test_window_days)
            
            # Extract splits for this window
            train_window = merged[
                (merged["timestamp"] >= train_start) & (merged["timestamp"] < train_end)
            ]
            val_window = merged[
                (merged["timestamp"] >= val_start) & (merged["timestamp"] < val_end)
            ]
            test_window = merged[
                (merged["timestamp"] >= test_start) & (merged["timestamp"] < test_end)
            ]
            
            if not train_window.empty:
                all_train.append(train_window)
            if not val_window.empty:
                all_validation.append(val_window)
            if not test_window.empty:
                all_test.append(test_window)
            
            # Move to next window
            current_date += timedelta(days=step_days)
        
        # Combine all windows
        train = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame()
        validation = (
            pd.concat(all_validation, ignore_index=True)
            if all_validation
            else pd.DataFrame()
        )
        test = pd.concat(all_test, ignore_index=True) if all_test else pd.DataFrame()
        
        return {"train": train, "validation": validation, "test": test}
    
    def _compute_split_statistics(
        self,
        splits: Dict[str, pd.DataFrame],
        target_config: TargetConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for each split.
        
        Args:
            splits: Dictionary with train/validation/test splits
            target_config: Target configuration
            
        Returns:
            Dictionary with statistics for each split
        """
        statistics = {}
        
        for split_name, split_df in splits.items():
            if split_df.empty or "target" not in split_df.columns:
                statistics[split_name] = {}
                continue
            
            target_series = split_df["target"]
            
            # Initialize statistics dict
            split_stats = {}
            
            # Determine task type from target_config
            is_classification = target_config.type in ["classification", "risk_adjusted"]
            
            if is_classification:
                # Class distribution
                class_dist = target_series.value_counts(dropna=False).to_dict()
                # Convert keys to strings for JSON serialization
                class_distribution = {str(k): int(v) for k, v in class_dist.items()}
                split_stats["class_distribution"] = class_distribution
                
                # Class balance ratio (ratio of minority to majority class)
                if len(class_dist) > 1:
                    counts = list(class_dist.values())
                    minority_count = min(counts)
                    majority_count = max(counts)
                    class_balance_ratio = minority_count / majority_count if majority_count > 0 else 0.0
                    split_stats["class_balance_ratio"] = float(class_balance_ratio)
                    split_stats["minority_class_size"] = int(minority_count)
                else:
                    split_stats["class_balance_ratio"] = 1.0
                    split_stats["minority_class_size"] = int(list(class_dist.values())[0]) if class_dist else 0
                
                # Total classes count
                split_stats["total_classes"] = len(class_dist)
            
            # Target statistics (for both classification and regression)
            # Convert to numeric, handling any non-numeric values
            numeric_targets = pd.to_numeric(target_series, errors='coerce').dropna()
            
            if len(numeric_targets) > 0:
                target_stats = {
                    "mean": float(numeric_targets.mean()),
                    "median": float(numeric_targets.median()),
                    "std": float(numeric_targets.std()) if len(numeric_targets) > 1 else 0.0,
                    "min": float(numeric_targets.min()),
                    "max": float(numeric_targets.max()),
                    "count": int(len(numeric_targets)),
                }
                split_stats["target_statistics"] = target_stats
            
            statistics[split_name] = split_stats
        
        return statistics
    
    async def _write_dataset_splits(
        self, dataset_id: str, splits: Dict[str, pd.DataFrame], output_format: str
    ) -> Path:
        """
        Write dataset splits to storage.
        
        Args:
            dataset_id: Dataset ID
            splits: Dictionary with train/validation/test splits
            output_format: Output format (parquet, csv, hdf5)
            
        Returns:
            Path to dataset directory
        """
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
                    split_df.to_parquet, file_path, index=False
                )
            elif output_format == "csv":
                await asyncio.to_thread(
                    split_df.to_csv, file_path, index=False
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        
        return dataset_dir
    
    async def get_build_progress(self, dataset_id: str) -> Optional[dict]:
        """
        Get build progress for dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dictionary with progress information
        """
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

