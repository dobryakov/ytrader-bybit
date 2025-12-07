"""
Backfilling service for fetching historical market data from Bybit REST API.

Fetches historical data when insufficient data is available for model training,
enabling immediate model training without waiting for data accumulation.
"""
import asyncio
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, Optional, List
from uuid import uuid4
from pathlib import Path
import pandas as pd
import structlog

from src.utils.bybit_client import BybitClient, BybitAPIError
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.config import config
from src.logging import get_logger

logger = structlog.get_logger(__name__)


class BackfillingJob:
    """Represents a backfilling job."""
    
    def __init__(self, job_id: str, symbol: str, start_date: date, end_date: date, data_types: List[str]):
        self.job_id = job_id
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_types = data_types
        self.status = "pending"  # pending, in_progress, completed, failed
        self.progress = {
            "dates_completed": 0,
            "dates_total": 0,
            "current_date": None,
        }
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.completed_dates: List[date] = []
        self.failed_dates: List[date] = []


class BackfillingService:
    """Service for backfilling historical market data from Bybit REST API."""
    
    def __init__(
        self,
        parquet_storage: ParquetStorage,
        feature_registry_loader: Optional[FeatureRegistryLoader] = None,
        bybit_client: Optional[BybitClient] = None,
    ):
        """
        Initialize backfilling service.
        
        Args:
            parquet_storage: Parquet storage for saving backfilled data
            feature_registry_loader: Optional Feature Registry loader for determining data types
            bybit_client: Optional Bybit REST API client (creates default if not provided)
        """
        self._parquet_storage = parquet_storage
        self._feature_registry_loader = feature_registry_loader
        
        if bybit_client is None:
            self._bybit_client = BybitClient(
                api_key=config.bybit_api_key,
                api_secret=config.bybit_api_secret,
                base_url=config.bybit_rest_base_url,
                rate_limit_delay_ms=config.feature_service_backfill_rate_limit_delay_ms,
            )
        else:
            self._bybit_client = bybit_client
        
        self._jobs: Dict[str, BackfillingJob] = {}
        self._max_candles_per_request = 200  # Bybit API limit
    
    async def backfill_klines(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Backfill kline (candlestick) data from Bybit REST API.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_date: Start date for backfilling
            end_date: End date for backfilling
            interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
            
        Returns:
            List of kline data dictionaries in internal format
            
        Raises:
            BybitAPIError: If API request fails
        """
        logger.info(
            "backfilling_klines_start",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            interval=interval,
        )
        
        # Convert dates to timestamps (milliseconds)
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        
        while current_start < end_timestamp:
            # Calculate end timestamp for this chunk (max 200 candles)
            # For 1-minute interval: 200 minutes = 200 * 60 * 1000 ms
            chunk_duration_ms = self._max_candles_per_request * interval * 60 * 1000
            current_end = min(current_start + chunk_duration_ms, end_timestamp)
            
            try:
                # Make API request
                # Note: Bybit API returns data starting from 'start' timestamp
                # We use 'end' to limit the range, but API may return data up to 'end' inclusive
                response = await self._bybit_client.get(
                    endpoint="/v5/market/kline",
                    params={
                        "category": "spot",
                        "symbol": symbol,
                        "interval": str(interval),
                        "start": current_start,
                        "end": current_end,
                        "limit": self._max_candles_per_request,
                    },
                    authenticated=False,  # Public endpoint
                )
                
                # Parse response
                if "result" not in response or "list" not in response["result"]:
                    logger.warning(
                        "bybit_api_unexpected_response",
                        symbol=symbol,
                        response_keys=list(response.keys()),
                    )
                    break
                
                klines = response["result"]["list"]
                if not klines:
                    # No more data available, stop pagination
                    break
                
                # Log pagination details for debugging
                first_timestamp = int(klines[0][0]) if klines else None
                last_timestamp_in_response = int(klines[-1][0]) if klines else None
                logger.debug(
                    "backfilling_klines_api_response",
                    symbol=symbol,
                    request_start=current_start,
                    request_end=current_end,
                    response_count=len(klines),
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp_in_response,
                    first_datetime=datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).isoformat() if first_timestamp else None,
                    last_datetime=datetime.fromtimestamp(last_timestamp_in_response / 1000, tz=timezone.utc).isoformat() if last_timestamp_in_response else None,
                )
                
                # Convert to internal format
                for kline in klines:
                    # Bybit format: [startTime, open, high, low, close, volume, turnover]
                    # Internal format: timestamp, open, high, low, close, volume
                    internal_kline = {
                        "timestamp": datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "symbol": symbol,
                    }
                    all_klines.append(internal_kline)
                
                # Check for duplicates in current response
                timestamps_in_response = [int(k[0]) for k in klines]
                unique_timestamps = set(timestamps_in_response)
                if len(timestamps_in_response) != len(unique_timestamps):
                    duplicates_in_response = len(timestamps_in_response) - len(unique_timestamps)
                    logger.warning(
                        "backfilling_duplicates_in_api_response",
                        symbol=symbol,
                        request_start=current_start,
                        total_records=len(klines),
                        unique_records=len(unique_timestamps),
                        duplicates=duplicates_in_response,
                        message="API returned duplicate timestamps in single response",
                    )
                
                # Check for overlap with previously fetched data
                if all_klines:
                    existing_timestamps = {int(k["timestamp"].timestamp() * 1000) for k in all_klines}
                    overlapping = existing_timestamps.intersection(unique_timestamps)
                    if overlapping:
                        logger.warning(
                            "backfilling_pagination_overlap",
                            symbol=symbol,
                            request_start=current_start,
                            overlapping_count=len(overlapping),
                            overlapping_timestamps=sorted(list(overlapping))[:10],  # First 10 for logging
                            message="API returned data that overlaps with previously fetched data",
                        )
                        # Filter out overlapping data before adding to all_klines
                        # This prevents duplicates even if API returns overlapping data
                        new_klines = [k for k in klines if int(k[0]) not in existing_timestamps]
                        if len(new_klines) < len(klines):
                            logger.info(
                                "backfilling_filtered_overlapping",
                                symbol=symbol,
                                original_count=len(klines),
                                filtered_count=len(new_klines),
                                removed_count=len(klines) - len(new_klines),
                            )
                            klines = new_klines
                            # Recalculate unique timestamps after filtering
                            unique_timestamps = {int(k[0]) for k in klines}
                
                # Update current_start to last kline timestamp + interval duration
                # This ensures we don't request the same data twice
                # Use the last timestamp from filtered data (without overlaps)
                if klines:
                    last_timestamp = int(klines[-1][0])
                    next_start = last_timestamp + (interval * 60 * 1000)
                else:
                    # No new data in this response, advance by chunk duration to avoid infinite loop
                    next_start = current_end + (interval * 60 * 1000)
                
                # Safety check: if next_start doesn't advance, break to avoid infinite loop
                if next_start <= current_start:
                    logger.warning(
                        "backfilling_pagination_stuck",
                        symbol=symbol,
                        current_start=current_start,
                        next_start=next_start,
                        last_timestamp=last_timestamp,
                    )
                    break
                
                current_start = next_start
                
                # Safety check: if we've processed all requested data, break
                if current_start >= end_timestamp:
                    break
                
                logger.debug(
                    "backfilling_klines_chunk",
                    symbol=symbol,
                    chunk_size=len(klines),
                    total_klines=len(all_klines),
                )
                
            except BybitAPIError as e:
                logger.error(
                    "backfilling_klines_api_error",
                    symbol=symbol,
                    error=str(e),
                    current_start=current_start,
                )
                raise
        
        # Remove duplicates by timestamp (keep last occurrence)
        # This handles cases where API returns duplicate data
        if all_klines:
            # Convert to DataFrame for efficient deduplication
            df_temp = pd.DataFrame(all_klines)
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            initial_count = len(df_temp)
            df_temp = df_temp.drop_duplicates(subset=["timestamp"], keep="last")
            df_temp = df_temp.sort_values("timestamp")
            
            if len(df_temp) < initial_count:
                logger.warning(
                    "backfilling_klines_duplicates_removed",
                    symbol=symbol,
                    initial_count=initial_count,
                    final_count=len(df_temp),
                    duplicates_removed=initial_count - len(df_temp),
                )
            
            # Convert back to list of dicts
            all_klines = df_temp.to_dict("records")
        
        logger.info(
            "backfilling_klines_complete",
            symbol=symbol,
            total_klines=len(all_klines),
        )
        
        return all_klines
    
    async def _save_klines(
        self,
        symbol: str,
        date_str: str,
        klines: List[Dict[str, Any]],
    ) -> None:
        """
        Save klines to Parquet storage.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            klines: List of kline data dictionaries
        """
        if not klines:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(klines)
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Remove duplicates by timestamp (keep last occurrence)
        # This prevents duplicate data from being saved
        initial_count = len(df)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        if len(df) < initial_count:
            logger.warning(
                "klines_duplicates_removed",
                symbol=symbol,
                date=date_str,
                initial_count=initial_count,
                final_count=len(df),
                duplicates_removed=initial_count - len(df),
            )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # For backfilling, we should overwrite existing file, not append
        # Delete existing file if it exists to ensure clean data
        klines_path = self._parquet_storage._get_klines_path(symbol, date_str)
        if klines_path.exists():
            logger.debug(
                "klines_overwriting_existing",
                symbol=symbol,
                date=date_str,
                existing_file=str(klines_path),
            )
            klines_path.unlink()
        
        # Save to Parquet
        await self._parquet_storage.write_klines(symbol, date_str, df)
        
        logger.debug(
            "klines_saved",
            symbol=symbol,
            date=date_str,
            record_count=len(klines),
        )
    
    async def _validate_saved_data(
        self,
        symbol: str,
        date_str: str,
        expected_count: int,
        data_type: str,
    ) -> bool:
        """
        Validate that saved data is correct and readable.
        
        Args:
            symbol: Trading pair symbol
            date_str: Date string (YYYY-MM-DD)
            expected_count: Expected number of records
            data_type: Data type ("klines", "trades", etc.)
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Read data back from Parquet
            if data_type == "klines":
                read_data = await self._parquet_storage.read_klines(symbol, date_str)
            else:
                # Other data types not implemented yet
                return True
            
            # Verify record count
            actual_count = len(read_data)
            if actual_count != expected_count:
                logger.error(
                    "data_validation_failed_count_mismatch",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    expected_count=expected_count,
                    actual_count=actual_count,
                )
                return False
            
            # Verify data structure
            required_fields = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            missing_fields = [field for field in required_fields if field not in read_data.columns]
            if missing_fields:
                logger.error(
                    "data_validation_failed_missing_fields",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                    missing_fields=missing_fields,
                )
                return False
            
            # Verify data types
            if not pd.api.types.is_datetime64_any_dtype(read_data["timestamp"]):
                logger.error(
                    "data_validation_failed_timestamp_type",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                )
                return False
            
            # Verify data integrity: prices are positive, timestamps are valid
            if (read_data["open"] <= 0).any() or (read_data["high"] <= 0).any() or (read_data["low"] <= 0).any() or (read_data["close"] <= 0).any():
                logger.error(
                    "data_validation_failed_invalid_prices",
                    symbol=symbol,
                    date=date_str,
                    data_type=data_type,
                )
                return False
            
            logger.debug(
                "data_validation_passed",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
                record_count=actual_count,
            )
            
            return True
            
        except FileNotFoundError:
            logger.error(
                "data_validation_failed_file_not_found",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
            )
            return False
        except Exception as e:
            logger.error(
                "data_validation_failed_exception",
                symbol=symbol,
                date=date_str,
                data_type=data_type,
                error=str(e),
                exc_info=True,
            )
            return False
    
    async def _check_data_availability(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        data_types: List[str],
    ) -> Dict[date, List[str]]:
        """
        Check which dates need backfilling.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date
            end_date: End date
            data_types: List of data types to check
            
        Returns:
            Dict mapping date to list of missing data types
        """
        missing_data: Dict[date, List[str]] = {}
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.isoformat()
            missing_types = []
            
            for data_type in data_types:
                if data_type == "klines":
                    try:
                        # Try to read existing data
                        await self._parquet_storage.read_klines(symbol, date_str)
                    except FileNotFoundError:
                        missing_types.append(data_type)
                # Other data types not implemented yet
            
            if missing_types:
                missing_data[current_date] = missing_types
            
            current_date += timedelta(days=1)
        
        return missing_data
    
    async def backfill_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        data_types: Optional[List[str]] = None,
    ) -> str:
        """
        Backfill historical data for a symbol and date range.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for backfilling
            end_date: End date for backfilling
            data_types: Optional list of data types to backfill (if None, uses Feature Registry)
            
        Returns:
            Job ID for tracking backfilling progress
        """
        # Determine data types to backfill
        if data_types is None:
            if self._feature_registry_loader is not None:
                try:
                    required_types = self._feature_registry_loader.get_required_data_types()
                    data_type_mapping = self._feature_registry_loader.get_data_type_mapping()
                    # Map input sources to storage types
                    data_types = []
                    supported_types = {"klines"}  # Currently supported types for backfilling
                    
                    for input_source in required_types:
                        if input_source in data_type_mapping:
                            storage_types = data_type_mapping[input_source]
                            for storage_type in storage_types:
                                # Add all supported types from mapping
                                if storage_type in supported_types:
                                    if storage_type not in data_types:
                                        data_types.append(storage_type)
                                else:
                                    logger.warning(
                                        "backfilling_unsupported_data_type",
                                        symbol=symbol,
                                        input_source=input_source,
                                        storage_type=storage_type,
                                        message=f"Backfilling for {storage_type} is not yet implemented",
                                    )
                    
                    # If no supported types found, default to klines
                    if not data_types:
                        logger.warning(
                            "backfilling_no_supported_types",
                            symbol=symbol,
                            required_types=required_types,
                            data_type_mapping=data_type_mapping,
                            fallback="using klines",
                        )
                        data_types = ["klines"]
                    
                    logger.info(
                        "backfilling_data_types_from_registry",
                        symbol=symbol,
                        data_types=data_types,
                        required_input_sources=required_types,
                    )
                except Exception as e:
                    logger.warning(
                        "feature_registry_analysis_failed",
                        error=str(e),
                        fallback="backfilling_all_data_types",
                    )
                    data_types = ["klines"]  # Default to klines
            else:
                data_types = ["klines"]  # Default to klines
        
        # Check data availability
        missing_data = await self._check_data_availability(symbol, start_date, end_date, data_types)
        
        if not missing_data:
            logger.info(
                "backfilling_no_missing_data",
                symbol=symbol,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
            # Create a completed job
            job_id = str(uuid4())
            job = BackfillingJob(job_id, symbol, start_date, end_date, data_types)
            job.status = "completed"
            job.start_time = datetime.now(timezone.utc)
            job.end_time = datetime.now(timezone.utc)
            job.progress["dates_total"] = 0
            job.progress["dates_completed"] = 0
            self._jobs[job_id] = job
            return job_id
        
        # Create job
        job_id = str(uuid4())
        job = BackfillingJob(job_id, symbol, start_date, end_date, data_types)
        job.progress["dates_total"] = len(missing_data)
        self._jobs[job_id] = job
        
        # Start backfilling in background
        asyncio.create_task(self._backfill_job_task(job, missing_data))
        
        return job_id
    
    async def _backfill_job_task(
        self,
        job: BackfillingJob,
        missing_data: Dict[date, List[str]],
    ) -> None:
        """Background task for backfilling job."""
        job.status = "in_progress"
        job.start_time = datetime.now(timezone.utc)
        
        try:
            for date_obj, missing_types in missing_data.items():
                job.progress["current_date"] = date_obj.isoformat()
                
                for data_type in missing_types:
                    if data_type == "klines":
                        try:
                            # Backfill klines for this date
                            klines = await self.backfill_klines(
                                job.symbol,
                                date_obj,
                                date_obj,
                                interval=config.feature_service_backfill_default_interval,
                            )
                            
                            if klines:
                                # Save to Parquet
                                date_str = date_obj.isoformat()
                                
                                # Count unique records before saving (after deduplication in _save_klines)
                                # _save_klines will deduplicate, so we need to count unique timestamps
                                df_temp = pd.DataFrame(klines)
                                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
                                unique_count = df_temp["timestamp"].nunique()
                                
                                await self._save_klines(job.symbol, date_str, klines)
                                
                                # Validate saved data (compare with unique count, not total with duplicates)
                                validation_passed = await self._validate_saved_data(
                                    job.symbol,
                                    date_str,
                                    unique_count,
                                    "klines",
                                )
                                
                                if not validation_passed:
                                    # Delete corrupted file and mark date as failed
                                    try:
                                        klines_path = self._parquet_storage._get_klines_path(job.symbol, date_str)
                                        if klines_path.exists():
                                            klines_path.unlink()
                                        logger.warning(
                                            "backfilling_validation_failed_deleted",
                                            symbol=job.symbol,
                                            date=date_str,
                                        )
                                    except Exception as e:
                                        logger.error(
                                            "backfilling_validation_failed_delete_error",
                                            symbol=job.symbol,
                                            date=date_str,
                                            error=str(e),
                                        )
                                    
                                    job.failed_dates.append(date_obj)
                                else:
                                    job.completed_dates.append(date_obj)
                            else:
                                logger.warning(
                                    "backfilling_no_data",
                                    symbol=job.symbol,
                                    date=date_obj.isoformat(),
                                    data_type=data_type,
                                )
                                job.failed_dates.append(date_obj)
                        
                        except Exception as e:
                            logger.error(
                                "backfilling_date_failed",
                                symbol=job.symbol,
                                date=date_obj.isoformat(),
                                data_type=data_type,
                                error=str(e),
                                exc_info=True,
                            )
                            job.failed_dates.append(date_obj)
                
                job.progress["dates_completed"] += 1
            
            # Determine final job status
            job.end_time = datetime.now(timezone.utc)
            
            if job.failed_dates:
                # Some dates failed - mark as failed
                job.status = "failed"
                job.error_message = f"Failed to backfill {len(job.failed_dates)} date(s): {[d.isoformat() for d in job.failed_dates]}"
                logger.warning(
                    "backfilling_job_completed_with_failures",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    total_dates=job.progress["dates_total"],
                    completed_dates=len(job.completed_dates),
                    failed_dates=len(job.failed_dates),
                    failed_date_list=[d.isoformat() for d in job.failed_dates],
                )
            else:
                # All dates completed successfully
                job.status = "completed"
                logger.info(
                    "backfilling_job_completed",
                    job_id=job.job_id,
                    symbol=job.symbol,
                    total_dates=job.progress["dates_total"],
                    completed_dates=len(job.completed_dates),
                    failed_dates=len(job.failed_dates),
                )
        
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now(timezone.utc)
            job.error_message = str(e)
            
            logger.error(
                "backfilling_job_failed",
                job_id=job.job_id,
                symbol=job.symbol,
                error=str(e),
                exc_info=True,
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get backfilling job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary or None if job not found
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        
        return {
            "job_id": job.job_id,
            "symbol": job.symbol,
            "start_date": job.start_date.isoformat(),
            "end_date": job.end_date.isoformat(),
            "data_types": job.data_types,
            "status": job.status,
            "progress": job.progress,
            "start_time": job.start_time.isoformat() if job.start_time else None,
            "end_time": job.end_time.isoformat() if job.end_time else None,
            "error_message": job.error_message,
            "completed_dates": [d.isoformat() for d in job.completed_dates],
            "failed_dates": [d.isoformat() for d in job.failed_dates],
        }
    
    async def close(self) -> None:
        """Close Bybit client."""
        await self._bybit_client.close()

