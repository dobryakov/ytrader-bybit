"""
Streaming Dataset Builder for optimized dataset building.

Processes data in streaming fashion, day by day, with optimized caching
and vectorized feature computation.
"""
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Set
import pandas as pd
import structlog

from src.models.dataset import SplitStrategy, TargetConfig
from src.models.feature_registry import FeatureRegistry
from .requirements_analyzer import (
    FeatureRequirementsAnalyzer,
    DataRequirements,
    TimestampStrategy,
)
from .rolling_window import OptimizedRollingWindow
from .cache_strategy import AdaptiveCacheStrategy, CacheStrategy
from .daily_cache import OptimizedDailyDataCache
from .prefetcher import AdaptivePrefetcher
from .hybrid_feature_computer import HybridFeatureComputer
from .incremental_orderbook import IncrementalOrderbookManager

logger = structlog.get_logger(__name__)


class StreamingDatasetBuilder:
    """
    Streams dataset building with optimized caching and vectorization.
    
    Processes data day by day, using:
    - Multi-level caching (local + Redis)
    - Adaptive prefetching
    - Vectorized feature computation
    - Incremental orderbook updates
    """
    
    def __init__(
        self,
        cache_service: Optional["CacheService"],
        parquet_storage: "ParquetStorage",
        feature_registry_loader: Optional["FeatureRegistryLoader"],
        batch_size: int = 1000,
    ):
        """
        Initialize streaming dataset builder.
        
        Args:
            cache_service: Cache service (Redis or in-memory)
            parquet_storage: Parquet storage service
            feature_registry_loader: Feature Registry loader
            batch_size: Batch size for processing timestamps
        """
        self.cache_service = cache_service
        self.parquet_storage = parquet_storage
        self.feature_registry_loader = feature_registry_loader
        self.batch_size = batch_size
        
        self._requirements_analyzer = FeatureRequirementsAnalyzer()
        self._cache_strategy_selector = AdaptiveCacheStrategy()
        
        logger.info(
            "streaming_dataset_builder_initialized",
            batch_size=batch_size,
            cache_enabled=cache_service is not None,
        )
    
    async def build_dataset_streaming(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        feature_registry: FeatureRegistry,
        target_config: TargetConfig,
        dataset_id: str,
    ) -> pd.DataFrame:
        """
        Build dataset using streaming approach.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for dataset
            end_date: End date for dataset
            feature_registry: Feature Registry instance
            target_config: Target configuration
            dataset_id: Dataset ID for progress tracking
            
        Returns:
            DataFrame with features for all timestamps
        """
        logger.info(
            "streaming_dataset_build_started",
            dataset_id=dataset_id,
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        
        # Step 1: Analyze requirements
        requirements = self._requirements_analyzer.analyze(feature_registry)
        
        # Step 2: Determine cache strategy
        period_days = (end_date.date() - start_date.date()).days + 1
        cache_strategy = self._cache_strategy_selector.determine_strategy(
            period_days=period_days,
            symbol=symbol,
            data_types=list(requirements.required_data_types),
            max_lookback_minutes=requirements.max_lookback_minutes,
        )
        
        # Step 3: Initialize components
        # Collect all storage types needed
        storage_types_list = []
        for storage_type_list in requirements.storage_types.values():
            storage_types_list.extend(storage_type_list)
        storage_types_list = list(set(storage_types_list))  # Remove duplicates
        
        cache = OptimizedDailyDataCache(
            redis_cache=self.cache_service,
            parquet_storage=self.parquet_storage,
            strategy=cache_strategy,
            symbol=symbol,
            data_types=storage_types_list,
        ) if self.cache_service else None
        
        rolling_window = OptimizedRollingWindow(
            max_lookback_minutes=requirements.max_lookback_minutes,
            symbol=symbol,
        )
        
        orderbook_manager = None
        if requirements.needs_orderbook:
            orderbook_manager = IncrementalOrderbookManager(
                symbol=symbol,
                snapshot_refresh_interval=3600,
            )
        
        prefetcher = None
        if cache_strategy.prefetch_enabled and cache:
            prefetcher = AdaptivePrefetcher(
                cache=cache,
                target_buffer_hours=(
                    cache_strategy.prefetch_ahead_hours or 1.0
                ),
            )
        
        feature_computer = HybridFeatureComputer(
            requirements=requirements,
            feature_registry_version=feature_registry.version,
        )
        
        # Step 4: Generate list of days to process
        days_to_process = self._generate_days_list(start_date, end_date)
        
        logger.info(
            "streaming_dataset_initialized",
            dataset_id=dataset_id,
            days_count=len(days_to_process),
            cache_strategy=cache_strategy.cache_unit.value,
            needs_orderbook=requirements.needs_orderbook,
        )
        
        # Step 5: Load previous day data for lookback (if needed)
        # This ensures first day has sufficient data for features requiring lookback
        if days_to_process:
            first_day = days_to_process[0]
            previous_day = first_day - timedelta(days=1)
            
            # Load previous day data to provide lookback for first day
            logger.info(
                "loading_previous_day_for_lookback",
                dataset_id=dataset_id,
                first_day=first_day.isoformat(),
                previous_day=previous_day.isoformat(),
                max_lookback_minutes=requirements.max_lookback_minutes,
            )
            
            try:
                if cache:
                    prev_day_data = await cache.get_day_data(previous_day)
                else:
                    prev_day_data = await self._read_day_from_parquet(
                        symbol, previous_day, requirements
                    )
                
                if prev_day_data:
                    # Add previous day data to rolling window
                    prev_klines = prev_day_data.get("klines", pd.DataFrame())
                    prev_trades = prev_day_data.get("trades", pd.DataFrame())
                    
                    if not prev_klines.empty or not prev_trades.empty:
                        prev_day_end = datetime.combine(
                            previous_day, datetime.max.time(), tzinfo=timezone.utc
                        )
                        rolling_window.add_data(
                            timestamp=prev_day_end,
                            trades=prev_trades,
                            klines=prev_klines,
                            skip_trim=True,
                        )
                        logger.info(
                            "previous_day_data_loaded",
                            dataset_id=dataset_id,
                            previous_day=previous_day.isoformat(),
                            klines_count=len(prev_klines),
                            trades_count=len(prev_trades),
                        )
                    else:
                        logger.warning(
                            "previous_day_data_empty",
                            dataset_id=dataset_id,
                            previous_day=previous_day.isoformat(),
                        )
                else:
                    logger.warning(
                        "previous_day_data_unavailable",
                        dataset_id=dataset_id,
                        previous_day=previous_day.isoformat(),
                        message="Previous day data not available, first day may have insufficient lookback",
                    )
            except Exception as e:
                logger.warning(
                    "previous_day_load_failed",
                    dataset_id=dataset_id,
                    previous_day=previous_day.isoformat(),
                    error=str(e),
                    message="Continuing without previous day data, first day may have insufficient lookback",
                )
        
        # Step 6: Process days sequentially
        all_features = []
        
        for day_idx, day_date in enumerate(days_to_process):
            logger.info(
                "processing_day",
                dataset_id=dataset_id,
                day=day_date.isoformat(),
                day_index=day_idx + 1,
                total_days=len(days_to_process),
            )
            
            # Process day
            day_features = await self._process_day(
                symbol=symbol,
                day_date=day_date,
                requirements=requirements,
                cache=cache,
                rolling_window=rolling_window,
                orderbook_manager=orderbook_manager,
                feature_computer=feature_computer,
                prefetcher=prefetcher,
                dataset_id=dataset_id,
            )
            
            if not day_features.empty:
                all_features.append(day_features)
            
            # Prefetch next day if enabled
            if prefetcher and day_idx < len(days_to_process) - 1:
                next_day = days_to_process[day_idx + 1]
                await prefetcher.prefetch_day(next_day)
        
        # Step 6: Combine all features
        if not all_features:
            logger.warning(
                "no_features_computed",
                dataset_id=dataset_id,
            )
            return pd.DataFrame()
        
        features_df = pd.concat(all_features, ignore_index=True)
        features_df = features_df.sort_values("timestamp").reset_index(drop=True)

        # Step 7: Keep only features that are explicitly declared in the Feature Registry.
        # This prevents internal/base features (like volume_3s, vwap_*, etc.) that are
        # not part of the active registry from leaking into the final dataset.
        registry_feature_names = {f.name for f in feature_registry.features}
        original_columns = list(features_df.columns)

        # Always keep timestamp; filter the rest by registry names.
        allowed_columns = ["timestamp"] + [
            col for col in original_columns
            if col != "timestamp" and col in registry_feature_names
        ]

        # If for some reason no declared features are present, we still return timestamp-only
        # frame instead of failing hard; this will be caught by downstream quality checks.
        filtered_df = features_df[allowed_columns]

        logger.info(
            "streaming_dataset_build_completed",
            dataset_id=dataset_id,
            total_features=len(filtered_df),
            days_processed=len(days_to_process),
            original_feature_columns=len(original_columns),
            kept_feature_columns=len(allowed_columns),
        )
        
        return filtered_df
    
    def _generate_days_list(
        self, start_date: datetime, end_date: datetime
    ) -> List[date]:
        """
        Generate list of days to process.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of date objects
        """
        days = []
        current_date = start_date.date()
        end_date_obj = end_date.date()
        
        while current_date <= end_date_obj:
            days.append(current_date)
            current_date += timedelta(days=1)
        
        return days
    
    async def _process_day(
        self,
        symbol: str,
        day_date: date,
        requirements: DataRequirements,
        cache: Optional[OptimizedDailyDataCache],
        rolling_window: OptimizedRollingWindow,
        orderbook_manager: Optional[IncrementalOrderbookManager],
        feature_computer: HybridFeatureComputer,
        prefetcher: Optional[AdaptivePrefetcher],
        dataset_id: str,
    ) -> pd.DataFrame:
        """
        Process a single day of data.
        
        Args:
            symbol: Trading pair symbol
            day_date: Date to process
            requirements: Data requirements
            cache: Data cache
            rolling_window: Rolling window instance
            orderbook_manager: Orderbook manager (if needed)
            feature_computer: Feature computer
            prefetcher: Prefetcher (if enabled)
            dataset_id: Dataset ID for logging
            
        Returns:
            DataFrame with features for the day
        """
        # Load day data
        if cache:
            day_data = await cache.get_day_data(day_date)
        else:
            # Fallback: read directly from Parquet
            day_data = await self._read_day_from_parquet(symbol, day_date, requirements)
        
        if not day_data:
            logger.warning(
                "day_data_empty",
                dataset_id=dataset_id,
                day=day_date.isoformat(),
            )
            return pd.DataFrame()
        
        # Generate timestamps for the day
        timestamps = self._generate_timestamps_for_day(
            day_date, day_data, requirements
        )
        
        if timestamps.empty:
            logger.warning(
                "no_timestamps_for_day",
                dataset_id=dataset_id,
                day=day_date.isoformat(),
            )
            return pd.DataFrame()
        
        # Prepare data for rolling window
        # Normalize keys: cache returns "orderbook_snapshots"/"orderbook_deltas", 
        # but we use "snapshots"/"deltas" internally
        klines_df = day_data.get("klines", pd.DataFrame())
        trades_df = day_data.get("trades", pd.DataFrame())
        
        # Normalize orderbook keys
        if "orderbook_snapshots" in day_data:
            day_data["snapshots"] = day_data["orderbook_snapshots"]
        if "orderbook_deltas" in day_data:
            day_data["deltas"] = day_data["orderbook_deltas"]
        
        # Update rolling window with day data
        # For historical data, we add all data at once and skip trimming
        # Trimming will happen per-timestamp in get_window() if needed
        day_end = datetime.combine(day_date, datetime.max.time(), tzinfo=timezone.utc)
        rolling_window.add_data(
            timestamp=day_end,
            trades=trades_df,
            klines=klines_df,
            skip_trim=True,  # Don't trim when loading historical data
        )
        
        # Process timestamps in batches
        all_features = []
        
        for batch_start in range(0, len(timestamps), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(timestamps))
            batch_timestamps = timestamps.iloc[batch_start:batch_end]
            
            # Prepare orderbook states for batch (if needed)
            orderbook_states = None
            if requirements.needs_orderbook and orderbook_manager:
                orderbook_states = self._prepare_orderbook_states(
                    batch_timestamps,
                    day_data,
                    orderbook_manager,
                )
            
            # Prepare funding rates (if needed)
            funding_rates = None
            next_funding_times = None
            if requirements.needs_funding:
                funding_rates, next_funding_times = self._prepare_funding_rates(
                    batch_timestamps,
                    day_data.get("funding", pd.DataFrame()),
                )
            
            # Compute features for batch
            batch_features = feature_computer.compute_features_batch(
                timestamps=batch_timestamps,
                rolling_window=rolling_window,
                klines_df=klines_df,
                trades_df=trades_df,
                orderbook_states=orderbook_states,
                funding_rates=funding_rates,
                next_funding_times=next_funding_times,
            )
            
            if not batch_features.empty:
                all_features.append(batch_features)
            
            # Update prefetcher processing speed
            if prefetcher and len(batch_timestamps) > 0:
                last_timestamp = batch_timestamps.iloc[-1]
                # Normalize timestamp: convert pd.Timestamp to datetime and ensure timezone-aware
                if isinstance(last_timestamp, pd.Timestamp):
                    last_timestamp = last_timestamp.to_pydatetime()
                # Ensure timezone-aware UTC
                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
                else:
                    last_timestamp = last_timestamp.astimezone(timezone.utc)
                prefetcher.update_processing_speed(last_timestamp, len(batch_timestamps))
        
        # Combine batch results
        if not all_features:
            return pd.DataFrame()
        
        return pd.concat(all_features, ignore_index=True)
    
    def _generate_timestamps_for_day(
        self,
        day_date: date,
        day_data: Dict[str, pd.DataFrame],
        requirements: DataRequirements,
    ) -> pd.Series:
        """
        Generate timestamps for a day based on strategy.
        
        Args:
            day_date: Date
            day_data: Dictionary with data for the day
            requirements: Data requirements
            
        Returns:
            Series of timestamps
        """
        timestamps = set()
        
        if requirements.timestamp_strategy == TimestampStrategy.KLINES_ONLY:
            # Only klines timestamps
            klines = day_data.get("klines", pd.DataFrame())
            if not klines.empty and "timestamp" in klines.columns:
                timestamps.update(klines["timestamp"].tolist())
        
        elif requirements.timestamp_strategy == TimestampStrategy.TRADES_ONLY:
            # All trades timestamps
            trades = day_data.get("trades", pd.DataFrame())
            if not trades.empty and "timestamp" in trades.columns:
                timestamps.update(trades["timestamp"].tolist())
        
        elif requirements.timestamp_strategy == TimestampStrategy.KLINES_WITH_TRADES:
            # Klines + trades timestamps
            klines = day_data.get("klines", pd.DataFrame())
            trades = day_data.get("trades", pd.DataFrame())
            
            if not klines.empty and "timestamp" in klines.columns:
                timestamps.update(klines["timestamp"].tolist())
            if not trades.empty and "timestamp" in trades.columns:
                timestamps.update(trades["timestamp"].tolist())
        
        if not timestamps:
            return pd.Series(dtype="datetime64[ns, UTC]")
        
        # Convert to sorted Series
        timestamps_list = sorted(timestamps)
        return pd.Series(timestamps_list, name="timestamp")
    
    async def _read_day_from_parquet(
        self,
        symbol: str,
        day_date: date,
        requirements: DataRequirements,
    ) -> Dict[str, pd.DataFrame]:
        """
        Read day data directly from Parquet (fallback if cache not available).
        
        Args:
            symbol: Trading pair symbol
            day_date: Date
            requirements: Data requirements
            
        Returns:
            Dictionary with data
        """
        day_str = day_date.isoformat()
        data = {}
        
        # Read data based on requirements
        if requirements.needs_klines:
            data["klines"] = await self.parquet_storage.read_klines(symbol, day_str)
        if requirements.needs_trades:
            data["trades"] = await self.parquet_storage.read_trades(symbol, day_str)
        if requirements.needs_orderbook:
            data["snapshots"] = await self.parquet_storage.read_orderbook_snapshots(
                symbol, day_str
            )
            data["deltas"] = await self.parquet_storage.read_orderbook_deltas(
                symbol, day_str
            )
        if requirements.needs_ticker:
            data["ticker"] = await self.parquet_storage.read_ticker(symbol, day_str)
        if requirements.needs_funding:
            data["funding"] = await self.parquet_storage.read_funding(symbol, day_str)
        
        return data
    
    def _prepare_orderbook_states(
        self,
        timestamps: pd.Series,
        day_data: Dict[str, pd.DataFrame],
        orderbook_manager: IncrementalOrderbookManager,
    ) -> Dict[datetime, Optional["OrderbookState"]]:
        """
        Prepare orderbook states for timestamps.
        
        Args:
            timestamps: Series of timestamps
            day_data: Dictionary with day data
            orderbook_manager: Orderbook manager
            
        Returns:
            Dictionary mapping timestamp to OrderbookState
        """
        snapshots = day_data.get("snapshots", pd.DataFrame())
        deltas = day_data.get("deltas", pd.DataFrame())
        
        orderbook_states = {}
        
        for ts in timestamps:
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            
            state = orderbook_manager.update_to_timestamp(
                timestamp=ts,
                snapshots=snapshots,
                deltas=deltas,
            )
            orderbook_states[ts] = state
        
        return orderbook_states
    
    def _prepare_funding_rates(
        self,
        timestamps: pd.Series,
        funding_df: pd.DataFrame,
    ) -> tuple[Dict[datetime, float], Dict[datetime, int]]:
        """
        Prepare funding rates for timestamps.
        
        Args:
            timestamps: Series of timestamps
            funding_df: DataFrame with funding data
            
        Returns:
            Tuple of (funding_rates dict, next_funding_times dict)
        """
        funding_rates = {}
        next_funding_times = {}
        
        if funding_df.empty:
            return funding_rates, next_funding_times
        
        # Ensure timestamp is datetime
        if "timestamp" in funding_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(funding_df["timestamp"]):
                funding_df = funding_df.copy()
                funding_df["timestamp"] = pd.to_datetime(
                    funding_df["timestamp"], utc=True
                )
        
        for ts in timestamps:
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            
            # Get latest funding rate before timestamp
            funding_before = funding_df[funding_df["timestamp"] <= ts]
            if not funding_before.empty:
                latest = funding_before.iloc[-1]
                funding_rates[ts] = float(latest.get("funding_rate", 0.0))
                next_funding_times[ts] = int(latest.get("next_funding_time", 0))
        
        return funding_rates, next_funding_times

