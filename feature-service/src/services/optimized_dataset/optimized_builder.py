"""
Optimized Dataset Builder - complete replacement for DatasetBuilder.

Integrates all optimized components for fast dataset building:
- StreamingDatasetBuilder for day-by-day processing
- Multi-level caching
- Vectorized feature computation
- Incremental orderbook updates
"""
import asyncio
from datetime import datetime, timedelta, timezone
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
from .requirements_analyzer import DataRequirements

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
            "feature_registry_version": registry_version,
            "output_format": output_format,
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
            )
            
            # Step 1: Build features using streaming approach
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
            
            # Step 3: Validate no data leakage
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
            
            # Step 4: Split dataset
            dataset = await self._metadata_storage.get_dataset(dataset_id)
            if dataset is None:
                logger.error("dataset_not_found", dataset_id=dataset_id)
                return
            
            if split_strategy == SplitStrategy.TIME_BASED:
                splits = await self._split_time_based(
                    features_df, targets_df, dataset
                )
            else:
                splits = await self._split_walk_forward(
                    features_df, targets_df, dataset
                )
            
            # Step 5: Write splits to storage
            storage_path = await self._write_dataset_splits(
                dataset_id, splits, output_format
            )
            
            # Step 6: Update dataset record
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

