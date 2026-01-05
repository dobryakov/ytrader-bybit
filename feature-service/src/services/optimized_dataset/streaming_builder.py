"""
Streaming Dataset Builder for optimized dataset building.

Processes data in streaming fashion, day by day, with optimized caching
and vectorized feature computation.
"""
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Set
import pandas as pd
import structlog
import asyncio
import re

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
        backfilling_service: Optional[Any] = None,
    ):
        # Store problematic periods info for the last build
        self._last_problematic_periods: Dict[str, List[Dict[str, Any]]] = {}
        """
        Initialize streaming dataset builder.
        
        Args:
            cache_service: Cache service (Redis or in-memory)
            parquet_storage: Parquet storage service
            feature_registry_loader: Feature Registry loader
            batch_size: Batch size for processing timestamps
            backfilling_service: Optional BackfillingService for automatic backfill
        """
        self.cache_service = cache_service
        self.parquet_storage = parquet_storage
        self.feature_registry_loader = feature_registry_loader
        self.batch_size = batch_size
        self.backfilling_service = backfilling_service
        
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
            backfilling_service=self.backfilling_service,
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
        
        # Step 6: Generate all timestamps from start_date to end_date with configurable step
        timestamp_interval_minutes = feature_registry.get_timestamp_interval_minutes()
        timestamps = self._generate_timestamps_for_period(
            start_date, end_date, all_klines, all_trades, requirements, timestamp_interval_minutes
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
        
        # Step 6.5: Filter out problematic timestamps (identical OHLC periods)
        problematic_periods = []
        if not all_klines.empty and not timestamps.empty:
            timestamps, problematic_periods = self._filter_problematic_timestamps(
                timestamps, all_klines, dataset_id, symbol, target_horizon_seconds
            )
            
            # Store problematic periods for this dataset
            self._last_problematic_periods[dataset_id] = problematic_periods
            
            if problematic_periods:
                logger.warning(
                    "problematic_periods_excluded",
                    dataset_id=dataset_id,
                    symbol=symbol,
                    excluded_count=len(problematic_periods),
                    excluded_periods=[f"{p['start']} to {p['end']}" for p in problematic_periods[:10]],  # First 10
                    total_excluded_periods=len(problematic_periods),
                    remaining_timestamps=len(timestamps),
                    message="Excluded timestamps with identical OHLC (data quality issue)",
                )
        
        if timestamps.empty:
            logger.warning(
                "all_timestamps_filtered_out",
                dataset_id=dataset_id,
                symbol=symbol,
                message="All timestamps were filtered out due to data quality issues",
            )
            return pd.DataFrame()
        
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
            # Retry logic for automatic backfill
            max_retries = 2  # Allow 2 retries (initial attempt + 2 retries after backfill)
            retry_count = 0
            batch_features = None
            
            while retry_count <= max_retries:
                try:
                    batch_features = feature_computer.compute_features_batch(
                        timestamps=batch_timestamps,
                        rolling_window=rolling_window,
                        klines_df=all_klines,
                        trades_df=all_trades,
                        orderbook_states=orderbook_states,
                        funding_rates=funding_rates,
                        next_funding_times=next_funding_times,
                    )
                    break  # Success, exit retry loop
                except ValueError as e:
                    error_msg = str(e)
                    logger.debug(
                        "value_error_caught_in_batch",
                        dataset_id=dataset_id,
                        symbol=symbol,
                        error_message=error_msg,
                        retry_count=retry_count,
                        max_retries=max_retries,
                        has_backfilling_service=self.backfilling_service is not None,
                        is_missing_data="Missing required data" in error_msg,
                    )
                    # Check if this is a missing data error
                    if "Missing required data" in error_msg and retry_count < max_retries and self.backfilling_service:
                        # Extract timestamp from error message
                        # Error format: "Missing required data at timestamp 2026-01-03T01:15:00+00:00: ..."
                        # Use more specific regex to match ISO format timestamp
                        timestamp_match = re.search(r'at timestamp ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:\+[0-9]{2}:[0-9]{2}|Z|))', error_msg)
                        if timestamp_match:
                            error_timestamp_str = timestamp_match.group(1)
                            try:
                                # Handle different timestamp formats
                                if error_timestamp_str.endswith('Z'):
                                    error_timestamp_str = error_timestamp_str.replace('Z', '+00:00')
                                elif '+' not in error_timestamp_str and '-' not in error_timestamp_str[-6:]:
                                    # Assume UTC if no timezone
                                    error_timestamp_str = error_timestamp_str + '+00:00'
                                error_timestamp = datetime.fromisoformat(error_timestamp_str)
                                if error_timestamp.tzinfo is None:
                                    error_timestamp = error_timestamp.replace(tzinfo=timezone.utc)
                                logger.debug(
                                    "timestamp_extracted_from_error",
                                    original_error=error_msg,
                                    extracted_timestamp=error_timestamp.isoformat(),
                                )
                            except Exception as parse_error:
                                logger.warning(
                                    "failed_to_parse_timestamp_from_error",
                                    error_message=error_msg,
                                    timestamp_str=error_timestamp_str,
                                    parse_error=str(parse_error),
                                    fallback="using batch_start_ts",
                                )
                                error_timestamp = batch_start_ts  # Fallback to batch start
                        else:
                            logger.warning(
                                "timestamp_not_found_in_error",
                                error_message=error_msg,
                                fallback="using batch_start_ts",
                            )
                            error_timestamp = batch_start_ts  # Fallback to batch start
                        
                        # Determine missing data types from error message
                        missing_data_types = []
                        if "klines" in error_msg.lower():
                            missing_data_types.append("klines")
                        if "trades" in error_msg.lower():
                            missing_data_types.append("trades")
                        
                        if not missing_data_types:
                            missing_data_types = ["klines"]  # Default to klines
                        
                        logger.warning(
                            "missing_data_detected_attempting_backfill",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            error_timestamp=error_timestamp.isoformat(),
                            missing_data_types=missing_data_types,
                            error_message=error_msg,
                            retry_count=retry_count,
                        )
                        
                        # Calculate date range for backfill
                        max_lookback = requirements.max_lookback_minutes
                        buffer_minutes = 60  # 1 hour buffer
                        backfill_start = error_timestamp - timedelta(minutes=max_lookback + buffer_minutes)
                        backfill_end = error_timestamp + timedelta(minutes=buffer_minutes)
                        
                        backfill_start_date = backfill_start.date()
                        backfill_end_date = backfill_end.date()
                        
                        try:
                            # Start backfill job
                            job_id = await self.backfilling_service.backfill_historical(
                                symbol=symbol,
                                start_date=backfill_start_date,
                                end_date=backfill_end_date,
                                data_types=missing_data_types,
                            )
                            
                            # Wait for job to complete (with timeout)
                            max_wait_seconds = 300  # 5 minutes
                            wait_interval_seconds = 2
                            waited_seconds = 0
                            backfill_success = False
                            
                            while waited_seconds < max_wait_seconds:
                                status = self.backfilling_service.get_job_status(job_id)
                                if not status:
                                    logger.error(
                                        "backfill_job_not_found",
                                        job_id=job_id,
                                        symbol=symbol,
                                    )
                                    break
                                
                                if status["status"] == "completed":
                                    logger.info(
                                        "backfill_completed",
                                        job_id=job_id,
                                        symbol=symbol,
                                        completed_dates=status.get("completed_dates", []),
                                    )
                                    backfill_success = True
                                    break
                                elif status["status"] == "failed":
                                    logger.error(
                                        "backfill_failed",
                                        job_id=job_id,
                                        symbol=symbol,
                                        error_message=status.get("error_message"),
                                        failed_dates=status.get("failed_dates", []),
                                    )
                                    break
                                
                                # Job still in progress, wait a bit
                                await asyncio.sleep(wait_interval_seconds)
                                waited_seconds += wait_interval_seconds
                            
                            if not backfill_success:
                                logger.warning(
                                    "backfill_timeout_or_failed",
                                    job_id=job_id,
                                    symbol=symbol,
                                    waited_seconds=waited_seconds,
                                    max_wait_seconds=max_wait_seconds,
                                )
                                # Still try to reload data in case some data was backfilled
                            
                        except Exception as backfill_error:
                            logger.error(
                                "backfill_error",
                                symbol=symbol,
                                error_timestamp=error_timestamp.isoformat(),
                                error=str(backfill_error),
                                exc_info=True,
                            )
                            # Still try to reload data in case some data was backfilled
                        
                        # Reload data after backfill attempt
                        # Backfill was done, reload data and retry
                        logger.info(
                            "reloading_data_after_backfill",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            batch_start=batch_start,
                            batch_end=batch_end,
                            retry_count=retry_count,
                        )
                        
                        # Clear cache for affected dates to ensure fresh data is loaded
                        if cache:
                            # Clear local cache to force reload from parquet
                            cache.clear_local_cache()
                            logger.debug(
                                "cache_cleared_before_reload",
                                dataset_id=dataset_id,
                                symbol=symbol,
                                cache_type="local",
                            )
                        
                        # Reload data for the period
                        # Calculate data period needed for this batch
                        max_lookback = requirements.max_lookback_minutes
                        batch_data_start = batch_start_ts - timedelta(minutes=max_lookback + 20)
                        batch_data_end = batch_end_ts + timedelta(minutes=target_horizon_minutes + 20)
                        
                        logger.debug(
                            "data_reload_period_calculated",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            batch_data_start=batch_data_start.isoformat(),
                            batch_data_end=batch_data_end.isoformat(),
                            max_lookback_minutes=max_lookback,
                            error_timestamp=error_timestamp.isoformat(),
                        )
                        
                        # Reload klines and trades
                        batch_days_to_load = self._generate_days_list(batch_data_start, batch_data_end)
                        batch_all_klines = pd.DataFrame()
                        batch_all_trades = pd.DataFrame()
                        
                        for day_date in batch_days_to_load:
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
                                        if "timestamp" in day_klines.columns:
                                            day_klines = day_klines[
                                                (day_klines["timestamp"] >= batch_data_start) &
                                                (day_klines["timestamp"] <= batch_data_end)
                                            ]
                                        batch_all_klines = pd.concat([batch_all_klines, day_klines], ignore_index=True)
                                    
                                    if not day_trades.empty:
                                        if "timestamp" in day_trades.columns:
                                            day_trades = day_trades[
                                                (day_trades["timestamp"] >= batch_data_start) &
                                                (day_trades["timestamp"] <= batch_data_end)
                                            ]
                                        batch_all_trades = pd.concat([batch_all_trades, day_trades], ignore_index=True)
                            except Exception as day_error:
                                logger.warning(
                                    "day_data_reload_failed",
                                    dataset_id=dataset_id,
                                    day=day_date.isoformat(),
                                    error=str(day_error),
                                )
                        
                        # Update rolling window with reloaded data
                        if not batch_all_klines.empty or not batch_all_trades.empty:
                            # Sort and deduplicate
                            if not batch_all_klines.empty and "timestamp" in batch_all_klines.columns:
                                batch_all_klines = batch_all_klines.sort_values("timestamp").reset_index(drop=True)
                                batch_all_klines = batch_all_klines.drop_duplicates(subset=["timestamp"], keep="last")
                            
                            if not batch_all_trades.empty and "timestamp" in batch_all_trades.columns:
                                batch_all_trades = batch_all_trades.sort_values("timestamp").reset_index(drop=True)
                                batch_all_trades = batch_all_trades.drop_duplicates(subset=["timestamp"], keep="last")
                            
                            # Clear rolling window before adding reloaded data
                            rolling_window.clear()
                            
                            # Update rolling window with reloaded data
                            rolling_window.add_data(
                                timestamp=batch_data_end,
                                trades=batch_all_trades,
                                klines=batch_all_klines,
                                skip_trim=True,
                            )
                            
                            # Update all_klines and all_trades for this batch
                            all_klines = batch_all_klines
                            all_trades = batch_all_trades
                            
                            # Log detailed information about reloaded data
                            klines_timestamp_range = None
                            if not batch_all_klines.empty and "timestamp" in batch_all_klines.columns:
                                klines_timestamp_range = {
                                    "min": batch_all_klines["timestamp"].min().isoformat(),
                                    "max": batch_all_klines["timestamp"].max().isoformat(),
                                }
                            
                            # Check if we have data around the error timestamp
                            error_window_start = error_timestamp - timedelta(minutes=15)
                            error_window_end = error_timestamp
                            klines_in_error_window = 0
                            if not batch_all_klines.empty and "timestamp" in batch_all_klines.columns:
                                klines_in_error_window = len(
                                    batch_all_klines[
                                        (batch_all_klines["timestamp"] >= error_window_start) &
                                        (batch_all_klines["timestamp"] <= error_window_end)
                                    ]
                                )
                            
                            logger.info(
                                "data_reloaded_after_backfill",
                                dataset_id=dataset_id,
                                symbol=symbol,
                                klines_count=len(batch_all_klines),
                                trades_count=len(batch_all_trades),
                                klines_timestamp_range=klines_timestamp_range,
                                error_timestamp=error_timestamp.isoformat(),
                                klines_in_error_window=klines_in_error_window,
                                error_window_start=error_window_start.isoformat(),
                                error_window_end=error_window_end.isoformat(),
                            )
                        
                        retry_count += 1
                        continue  # Retry computation with reloaded data
                    else:
                        # Log why backfill was not attempted
                        logger.warning(
                            "missing_data_error_not_handled",
                            dataset_id=dataset_id,
                            symbol=symbol,
                            error_message=error_msg,
                            has_backfilling_service=self.backfilling_service is not None,
                            retry_count=retry_count,
                            max_retries=max_retries,
                            is_missing_data_error="Missing required data" in error_msg,
                        )
                        # Not a backfill-related error or max retries reached, re-raise
                        raise
            
            if batch_features is None:
                # All retries failed, raise the last error
                raise ValueError(
                    f"Failed to compute features for batch after {max_retries + 1} attempts. "
                    f"Last error should have been logged above."
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
        timestamp_interval_minutes: int = 1,
    ) -> pd.Series:
        """
        Generate timestamps for entire period with configurable step.
        
        Timestamps are aligned to interval boundaries:
        - For 1-minute intervals: 00:00, 01:00, 02:00, ...
        - For 5-minute intervals: 00:00, 05:00, 10:00, 15:00, ...
        - For 15-minute intervals: 00:00, 15:00, 30:00, 45:00, ...
        
        Args:
            start_date: Start date for dataset
            end_date: End date for dataset
            all_klines: All klines DataFrame (unused, kept for compatibility)
            all_trades: All trades DataFrame (unused, kept for compatibility)
            requirements: Data requirements (unused, kept for compatibility)
            timestamp_interval_minutes: Interval in minutes for timestamp generation (default: 1)
            
        Returns:
            Series of timestamps with specified interval, aligned to interval boundaries
        """
        timestamps = []
        current = start_date
        
        # Ensure timezone-aware
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        
        # Align start_date to interval boundary
        # For 1-minute: round to minute (00:00, 01:00, ...)
        # For 5-minute: round to 5-minute boundary (00:00, 05:00, 10:00, ...)
        # For 15-minute: round to 15-minute boundary (00:00, 15:00, 30:00, 45:00, ...)
        if timestamp_interval_minutes == 1:
            # Round to minute boundary
            current = current.replace(second=0, microsecond=0)
        else:
            # Round down to nearest interval boundary
            # Example: 16:42:30 with 5-minute interval -> 16:40:00
            # Example: 16:47:30 with 5-minute interval -> 16:45:00
            minutes_aligned = (current.minute // timestamp_interval_minutes) * timestamp_interval_minutes
            current = current.replace(minute=minutes_aligned, second=0, microsecond=0)
        
        # Generate timestamps with specified interval
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(minutes=timestamp_interval_minutes)
        
        if not timestamps:
            return pd.Series(dtype="datetime64[ns, UTC]")
        
        logger.debug(
            "timestamps_generated_for_period",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval_minutes=timestamp_interval_minutes,
            count=len(timestamps),
            first_timestamp=timestamps[0].isoformat() if timestamps else None,
            last_timestamp=timestamps[-1].isoformat() if timestamps else None,
        )
        
        # Convert to Series
        return pd.Series(timestamps, name="timestamp", dtype="datetime64[ns, UTC]")
    
    def _filter_problematic_timestamps(
        self,
        timestamps: pd.Series,
        klines: pd.DataFrame,
        dataset_id: str,
        symbol: str,
        horizon_seconds: int = 3600,
    ) -> tuple[pd.Series, List[Dict[str, Any]]]:
        """
        Filter out timestamps that fall into problematic periods with identical OHLC.
        
        A timestamp is considered problematic if:
        1. The kline at that timestamp has identical OHLC values
        2. The kline 1 hour ahead (for target computation) also has identical OHLC
        3. Both prices are the same (would result in zero target)
        
        Args:
            timestamps: Series of timestamps to filter
            klines: DataFrame with klines data
            dataset_id: Dataset ID for logging
            symbol: Symbol for logging
            horizon_seconds: Target horizon in seconds (default: 3600 for 1 hour)
            
        Returns:
            Tuple of (filtered_timestamps, problematic_periods_list)
        """
        if klines.empty or timestamps.empty:
            return timestamps, []
        
        # Ensure klines are sorted by timestamp
        klines_sorted = klines.sort_values("timestamp").reset_index(drop=True)
        
        # Create a set of problematic timestamps to exclude
        problematic_timestamps = set()
        problematic_periods = []
        
        # Check each timestamp
        for ts in timestamps:
            # Find kline at this timestamp (or nearest)
            current_klines = klines_sorted[klines_sorted["timestamp"] >= ts]
            if current_klines.empty:
                continue
            
            current_kline = current_klines.iloc[0]
            
            # Check if current kline has identical OHLC
            current_identical = (
                abs(current_kline["open"] - current_kline["high"]) < 1e-6 and
                abs(current_kline["high"] - current_kline["low"]) < 1e-6 and
                abs(current_kline["low"] - current_kline["close"]) < 1e-6
            )
            
            if not current_identical:
                continue  # Current kline is OK, skip
            
            # Check future kline (for target computation)
            future_ts = ts + timedelta(seconds=horizon_seconds)
            future_klines = klines_sorted[klines_sorted["timestamp"] >= future_ts]
            if future_klines.empty:
                continue
            
            future_kline = future_klines.iloc[0]
            
            # Check if future kline also has identical OHLC
            future_identical = (
                abs(future_kline["open"] - future_kline["high"]) < 1e-6 and
                abs(future_kline["high"] - future_kline["low"]) < 1e-6 and
                abs(future_kline["low"] - future_kline["close"]) < 1e-6
            )
            
            if not future_identical:
                continue  # Future kline is OK, skip
            
            # Check if prices are the same (would result in zero target)
            price_same = abs(current_kline["close"] - future_kline["close"]) < 1e-6
            
            if price_same:
                # This timestamp is problematic - exclude it
                problematic_timestamps.add(ts)
                
                # Track problematic period
                problematic_periods.append({
                    "start": ts.isoformat(),
                    "end": future_ts.isoformat(),
                    "current_price": float(current_kline["close"]),
                    "future_price": float(future_kline["close"]),
                    "current_volume": float(current_kline.get("volume", 0)),
                    "future_volume": float(future_kline.get("volume", 0)),
                })
        
        # Filter out problematic timestamps
        if problematic_timestamps:
            filtered_timestamps = timestamps[~timestamps.isin(problematic_timestamps)]
            
            logger.info(
                "problematic_timestamps_filtered",
                dataset_id=dataset_id,
                symbol=symbol,
                original_count=len(timestamps),
                filtered_count=len(filtered_timestamps),
                excluded_count=len(problematic_timestamps),
                excluded_percentage=len(problematic_timestamps) / len(timestamps) * 100,
            )
            
            return filtered_timestamps, problematic_periods
        else:
            return timestamps, []
    
    def get_problematic_periods(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get problematic periods that were excluded for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of problematic periods dictionaries
        """
        return self._last_problematic_periods.get(dataset_id, [])
    
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

