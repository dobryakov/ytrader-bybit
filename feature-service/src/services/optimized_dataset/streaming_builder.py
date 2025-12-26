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
            builder_type="streaming",
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
        
        # Calculate target horizon minutes early (needed for HybridFeatureComputer)
        target_horizon_seconds = target_config.horizon if target_config else 0
        target_horizon_minutes = (target_horizon_seconds + 59) // 60  # Round up to minutes
        
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
            feature_registry=feature_registry,
            target_horizon_minutes=target_horizon_minutes,
        )
        
        # Step 4: Calculate buffer period and load all required data upfront
        # Buffer for safety (extra minutes beyond max_lookback)
        buffer_minutes = 20
        
        # Calculate data loading period:
        # Start: from midnight of first day, go back by max_lookback + buffer
        # End: to end of last day + target horizon + buffer
        first_day_midnight = datetime.combine(
            start_date.date(), datetime.min.time(), tzinfo=timezone.utc
        )
        data_start = first_day_midnight - timedelta(
            minutes=requirements.max_lookback_minutes + buffer_minutes
        )
        data_end = datetime.combine(
            end_date.date(), datetime.max.time(), tzinfo=timezone.utc
        ) + timedelta(minutes=target_horizon_minutes + buffer_minutes)
        
        logger.info(
            "streaming_dataset_initialized",
            dataset_id=dataset_id,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            data_start=data_start.isoformat(),
            data_end=data_end.isoformat(),
            max_lookback_minutes=requirements.max_lookback_minutes,
            buffer_minutes=buffer_minutes,
            target_horizon_minutes=target_horizon_minutes,
            cache_strategy=cache_strategy.cache_unit.value,
            needs_orderbook=requirements.needs_orderbook,
            builder_type="streaming",
        )
        
        # Step 5: Load all required data into rolling window buffer
        logger.info(
            "loading_data_buffer",
            dataset_id=dataset_id,
            data_start=data_start.isoformat(),
            data_end=data_end.isoformat(),
            builder_type="streaming",
        )
        
        all_klines = pd.DataFrame()
        all_trades = pd.DataFrame()
        
        # Generate list of days to load
        days_to_load = self._generate_days_list(
            data_start, data_end
        )
        
        for day_date in days_to_load:
            try:
                if cache:
                    day_data = await cache.get_day_data(day_date)
                else:
                    day_data = await self._read_day_from_parquet(
                        symbol, day_date, requirements
                    )
                
                if day_data:
                    day_klines = day_data.get("klines", pd.DataFrame())
                    day_trades = day_data.get("trades", pd.DataFrame())
                    
                    if not day_klines.empty:
                        # Filter klines within data period
                        if "timestamp" in day_klines.columns:
                            day_klines = day_klines[
                                (day_klines["timestamp"] >= data_start) &
                                (day_klines["timestamp"] <= data_end)
                            ]
                        all_klines = pd.concat([all_klines, day_klines], ignore_index=True)
                    
                    if not day_trades.empty:
                        # Filter trades within data period
                        if "timestamp" in day_trades.columns:
                            day_trades = day_trades[
                                (day_trades["timestamp"] >= data_start) &
                                (day_trades["timestamp"] <= data_end)
                            ]
                        all_trades = pd.concat([all_trades, day_trades], ignore_index=True)
            except Exception as e:
                logger.warning(
                    "day_data_load_failed",
                    dataset_id=dataset_id,
                    day=day_date.isoformat(),
                    error=str(e),
                )
        
        # Sort and deduplicate
        if not all_klines.empty and "timestamp" in all_klines.columns:
            all_klines = all_klines.sort_values("timestamp").reset_index(drop=True)
            all_klines = all_klines.drop_duplicates(subset=["timestamp"], keep="last")
        
        if not all_trades.empty and "timestamp" in all_trades.columns:
            all_trades = all_trades.sort_values("timestamp").reset_index(drop=True)
            all_trades = all_trades.drop_duplicates(subset=["timestamp"], keep="last")
        
        # Initialize rolling window with all loaded data
        if not all_klines.empty or not all_trades.empty:
            rolling_window.add_data(
                timestamp=data_end,
                trades=all_trades,
                klines=all_klines,
                skip_trim=True,  # Don't trim - we want all data in buffer
            )
            logger.info(
                "data_buffer_loaded",
                dataset_id=dataset_id,
                klines_count=len(all_klines),
                trades_count=len(all_trades),
                data_start=data_start.isoformat(),
                data_end=data_end.isoformat(),
                builder_type="streaming",
            )
        else:
            logger.warning(
                "data_buffer_empty",
                dataset_id=dataset_id,
                data_start=data_start.isoformat(),
                data_end=data_end.isoformat(),
            )
        
        # Step 6: Generate all timestamps from start_date to end_date with 1-minute step
        timestamps = self._generate_timestamps_for_period(
            start_date, end_date, all_klines, all_trades, requirements
        )
        
        if timestamps.empty:
            logger.warning(
                "no_timestamps_generated",
                dataset_id=dataset_id,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            return pd.DataFrame()
        
        logger.info(
            "timestamps_generated",
            dataset_id=dataset_id,
            timestamps_count=len(timestamps),
            first_timestamp=timestamps.iloc[0].isoformat() if len(timestamps) > 0 else None,
            last_timestamp=timestamps.iloc[-1].isoformat() if len(timestamps) > 0 else None,
            builder_type="streaming",
        )
        
        # Step 7: Process timestamps sequentially in batches
        all_features = []
        
        for batch_start in range(0, len(timestamps), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(timestamps))
            # Reset index to ensure RangeIndex starting from 0 for each batch
            # This prevents index alignment issues when merging results from different batches
            batch_timestamps = timestamps.iloc[batch_start:batch_end].reset_index(drop=True)
            
            # Normalize batch timestamps for period-based data loading
            batch_start_ts = batch_timestamps.iloc[0]
            batch_end_ts = batch_timestamps.iloc[-1]
            
            # Normalize timestamps to datetime
            if isinstance(batch_start_ts, pd.Timestamp):
                batch_start_ts = batch_start_ts.to_pydatetime()
            if isinstance(batch_end_ts, pd.Timestamp):
                batch_end_ts = batch_end_ts.to_pydatetime()
            
            # Ensure timezone-aware UTC
            if batch_start_ts.tzinfo is None:
                batch_start_ts = batch_start_ts.replace(tzinfo=timezone.utc)
            else:
                batch_start_ts = batch_start_ts.astimezone(timezone.utc)
            if batch_end_ts.tzinfo is None:
                batch_end_ts = batch_end_ts.replace(tzinfo=timezone.utc)
            else:
                batch_end_ts = batch_end_ts.astimezone(timezone.utc)
            
            # Prepare orderbook states for batch (if needed)
            orderbook_states = None
            if requirements.needs_orderbook and orderbook_manager:
                orderbook_states = await self._prepare_orderbook_states_for_period(
                    batch_timestamps,
                    batch_start_ts,
                    batch_end_ts,
                    symbol,
                    requirements,
                    cache,
                    orderbook_manager,
                )
            
            # Prepare funding rates (if needed)
            funding_rates = None
            next_funding_times = None
            if requirements.needs_funding:
                funding_data = await self._load_funding_data_for_period(
                    symbol, batch_start_ts, batch_end_ts, cache, requirements
                )
                if not funding_data.empty:
                    funding_rates, next_funding_times = self._prepare_funding_rates(
                        batch_timestamps,
                        funding_data,
                    )
            
            # Compute features for batch
            batch_features = feature_computer.compute_features_batch(
                timestamps=batch_timestamps,
                rolling_window=rolling_window,
                klines_df=all_klines,
                trades_df=all_trades,
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
        
        # Step 8: Combine all features
        if not all_features:
            logger.warning(
                "no_features_computed",
                dataset_id=dataset_id,
                symbol=symbol,
            )
            return pd.DataFrame()
        
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Normalize all timestamps to timezone-aware UTC before sorting
        # This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
        if "timestamp" in features_df.columns:
            if pd.api.types.is_datetime64_any_dtype(features_df["timestamp"]):
                # Convert to timezone-aware UTC if needed
                if features_df["timestamp"].dt.tz is None:
                    # Timezone-naive: assume UTC and make aware
                    features_df["timestamp"] = features_df["timestamp"].dt.tz_localize(timezone.utc)
                else:
                    # Timezone-aware: convert to UTC
                    features_df["timestamp"] = features_df["timestamp"].dt.tz_convert(timezone.utc)
            else:
                # Not datetime dtype: convert to datetime first
                features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
        
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

        # Calculate period days for logging
        period_days = (end_date.date() - start_date.date()).days + 1
        
        logger.info(
            "streaming_dataset_build_completed",
            dataset_id=dataset_id,
            total_features=len(filtered_df),
            period_days=period_days,
            original_feature_columns=len(original_columns),
            kept_feature_columns=len(allowed_columns),
            builder_type="streaming",
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
        
        # Normalize all timestamps to timezone-aware UTC before creating Series
        # This prevents "Cannot compare tz-naive and tz-aware timestamps" errors
        normalized_timestamps = []
        for ts in timestamps:
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            elif not isinstance(ts, datetime):
                # Convert to datetime if needed
                ts = pd.to_datetime(ts, utc=True).to_pydatetime()
            
            # Ensure timezone-aware UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            
            normalized_timestamps.append(ts)
        
        # Convert to sorted Series
        timestamps_list = sorted(normalized_timestamps)
        return pd.Series(timestamps_list, name="timestamp", dtype="datetime64[ns, UTC]")
    
    def _generate_timestamps_for_period(
        self,
        start_date: datetime,
        end_date: datetime,
        all_klines: pd.DataFrame,
        all_trades: pd.DataFrame,
        requirements: DataRequirements,
    ) -> pd.Series:
        """
        Generate timestamps for entire period with 1-minute step.
        
        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            all_klines: All klines DataFrame
            all_trades: All trades DataFrame
            requirements: Data requirements
            
        Returns:
            Series of timestamps with 1-minute intervals
        """
        # Generate 1-minute intervals from start_date to end_date
        timestamps = []
        current = start_date
        
        # Ensure timezone-aware
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Round start_date to minute boundary
        current = current.replace(second=0, microsecond=0)
        
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(minutes=1)
        
        if not timestamps:
            return pd.Series(dtype="datetime64[ns, UTC]")
        
        # Convert to Series
        return pd.Series(timestamps, name="timestamp", dtype="datetime64[ns, UTC]")
    
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
    
    async def _prepare_orderbook_states_for_period(
        self,
        timestamps: pd.Series,
        batch_start_ts: datetime,
        batch_end_ts: datetime,
        symbol: str,
        requirements: DataRequirements,
        cache: Optional[OptimizedDailyDataCache],
        orderbook_manager: IncrementalOrderbookManager,
    ) -> Dict[datetime, Optional["OrderbookState"]]:
        """
        Prepare orderbook states for timestamps in a period.
        
        Args:
            timestamps: Series of timestamps
            batch_start_ts: Batch start timestamp
            batch_end_ts: Batch end timestamp
            symbol: Trading pair symbol
            requirements: Data requirements
            cache: Data cache
            orderbook_manager: Orderbook manager
            
        Returns:
            Dictionary mapping timestamp to OrderbookState
        """
        # Load orderbook data for the period
        days_to_load = self._generate_days_list(batch_start_ts, batch_end_ts)
        
        all_snapshots = pd.DataFrame()
        all_deltas = pd.DataFrame()
        
        for day_date in days_to_load:
            try:
                if cache:
                    day_data = await cache.get_day_data(day_date)
                else:
                    day_data = await self._read_day_from_parquet(
                        symbol, day_date, requirements
                    )
                
                if day_data:
                    snapshots = day_data.get("snapshots", pd.DataFrame())
                    deltas = day_data.get("deltas", pd.DataFrame())
                    
                    if not snapshots.empty:
                        # Filter snapshots within batch period
                        if "timestamp" in snapshots.columns:
                            snapshots = snapshots[
                                (snapshots["timestamp"] >= batch_start_ts) &
                                (snapshots["timestamp"] <= batch_end_ts)
                            ]
                        all_snapshots = pd.concat([all_snapshots, snapshots], ignore_index=True)
                    
                    if not deltas.empty:
                        # Filter deltas within batch period
                        if "timestamp" in deltas.columns:
                            deltas = deltas[
                                (deltas["timestamp"] >= batch_start_ts) &
                                (deltas["timestamp"] <= batch_end_ts)
                            ]
                        all_deltas = pd.concat([all_deltas, deltas], ignore_index=True)
            except Exception as e:
                logger.warning(
                    "orderbook_data_load_failed",
                    day=day_date.isoformat(),
                    error=str(e),
                )
        
        # Sort and deduplicate
        if not all_snapshots.empty and "timestamp" in all_snapshots.columns:
            all_snapshots = all_snapshots.sort_values("timestamp").reset_index(drop=True)
            all_snapshots = all_snapshots.drop_duplicates(subset=["timestamp"], keep="last")
        
        if not all_deltas.empty and "timestamp" in all_deltas.columns:
            all_deltas = all_deltas.sort_values("timestamp").reset_index(drop=True)
            all_deltas = all_deltas.drop_duplicates(subset=["timestamp"], keep="last")
        
        # Prepare orderbook states using existing method
        day_data = {
            "snapshots": all_snapshots,
            "deltas": all_deltas,
        }
        return self._prepare_orderbook_states(
            timestamps, day_data, orderbook_manager
        )
    
    async def _load_funding_data_for_period(
        self,
        symbol: str,
        batch_start_ts: datetime,
        batch_end_ts: datetime,
        cache: Optional[OptimizedDailyDataCache],
        requirements: DataRequirements,
    ) -> pd.DataFrame:
        """
        Load funding data for a period.
        
        Args:
            symbol: Trading pair symbol
            batch_start_ts: Batch start timestamp
            batch_end_ts: Batch end timestamp
            cache: Data cache
            requirements: Data requirements
            
        Returns:
            DataFrame with funding data
        """
        if not requirements.needs_funding:
            return pd.DataFrame()
        
        # Load funding data for the period
        days_to_load = self._generate_days_list(batch_start_ts, batch_end_ts)
        
        all_funding = pd.DataFrame()
        
        for day_date in days_to_load:
            try:
                if cache:
                    day_data = await cache.get_day_data(day_date)
                else:
                    day_data = await self._read_day_from_parquet(
                        symbol, day_date, requirements
                    )
                
                if day_data:
                    funding = day_data.get("funding", pd.DataFrame())
                    
                    if not funding.empty:
                        # Filter funding within batch period
                        if "timestamp" in funding.columns:
                            funding = funding[
                                (funding["timestamp"] >= batch_start_ts) &
                                (funding["timestamp"] <= batch_end_ts)
                            ]
                        all_funding = pd.concat([all_funding, funding], ignore_index=True)
            except Exception as e:
                logger.warning(
                    "funding_data_load_failed",
                    day=day_date.isoformat(),
                    error=str(e),
                )
        
        # Sort and deduplicate
        if not all_funding.empty and "timestamp" in all_funding.columns:
            all_funding = all_funding.sort_values("timestamp").reset_index(drop=True)
            all_funding = all_funding.drop_duplicates(subset=["timestamp"], keep="last")
        
        return all_funding

