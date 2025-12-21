"""
Data loading and availability checking for target computation.

This module provides functions for finding available data ranges and loading
historical data for target computation, with fallback logic for data delays.
"""
from datetime import datetime, timedelta, timezone, date
from typing import Optional, Dict, Any
import pandas as pd
from src.storage.parquet_storage import ParquetStorage
from src.logging import get_logger

logger = get_logger(__name__)


async def find_available_data_range(
    parquet_storage: ParquetStorage,
    symbol: str,
    target_timestamp: datetime,
    max_lookback_seconds: int = 300,
    max_expected_delay_seconds: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Найти диапазон доступных данных и адаптировать target_timestamp.
    
    Это ДОПОЛНИТЕЛЬНЫЙ уровень fallback перед вызовом TargetComputationEngine.
    
    Алгоритм:
    1. Определить дату для target_timestamp
    2. Попытаться прочитать klines за эту дату (и возможно предыдущую)
    3. Найти последний доступный timestamp в данных
    4. Если target_timestamp <= latest_available_timestamp:
       - Данные доступны, использовать target_timestamp как есть
    5. Если target_timestamp > latest_available_timestamp:
       - Вычислить разрыв: gap = target_timestamp - latest_available_timestamp
       - Если gap <= max_lookback_seconds:
         - Адаптировать target_timestamp = latest_available_timestamp
         - Вернуть метаданные об адаптации
       - Если gap > max_lookback_seconds:
         - Вернуть None (данные слишком старые)
    
    Args:
        parquet_storage: ParquetStorage instance
        symbol: Trading pair symbol
        target_timestamp: Requested target timestamp
        max_lookback_seconds: Maximum lookback window for data availability
        max_expected_delay_seconds: Expected maximum delay for data ingestion
    
    Returns:
        Dict with adjusted timestamp and metadata, or None if data unavailable:
        {
            "adjusted_target_timestamp": datetime,
            "latest_available_timestamp": datetime,
            "timestamp_adjusted": bool,
            "lookback_seconds_used": int,
            "historical_data": pd.DataFrame
        }
    """
    # Normalize target_timestamp to UTC timezone-aware
    if target_timestamp.tzinfo is None:
        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
    else:
        target_timestamp = target_timestamp.astimezone(timezone.utc)
    
    logger.info(
        "find_available_data_range_start",
        symbol=symbol,
        target_timestamp=target_timestamp.isoformat(),
        max_lookback_seconds=max_lookback_seconds,
        max_expected_delay_seconds=max_expected_delay_seconds,
    )
    
    # Determine date range to check
    target_date = target_timestamp.date()
    previous_date = target_date - timedelta(days=1)
    
    logger.debug(
        "find_available_data_range_dates",
        symbol=symbol,
        target_date=target_date.isoformat(),
        previous_date=previous_date.isoformat(),
    )
    
    # Try to load data for target_date and previous_date
    all_data = []
    dates_checked = []
    dates_found = []
    dates_not_found = []
    
    for check_date in [target_date, previous_date]:
        date_str = check_date.strftime("%Y-%m-%d")
        dates_checked.append(date_str)
        try:
            klines_df = await parquet_storage.read_klines(symbol, date_str)
            if not klines_df.empty and "timestamp" in klines_df.columns:
                all_data.append(klines_df)
                dates_found.append(date_str)
                logger.debug(
                    "find_available_data_range_date_found",
                    symbol=symbol,
                    date=date_str,
                    rows=len(klines_df),
                    timestamp_min=klines_df["timestamp"].min().isoformat() if "timestamp" in klines_df.columns else None,
                    timestamp_max=klines_df["timestamp"].max().isoformat() if "timestamp" in klines_df.columns else None,
                )
            else:
                dates_not_found.append(date_str)
                logger.debug(
                    "find_available_data_range_date_empty",
                    symbol=symbol,
                    date=date_str,
                    empty=klines_df.empty,
                    has_timestamp="timestamp" in klines_df.columns if not klines_df.empty else False,
                )
        except FileNotFoundError as e:
            dates_not_found.append(date_str)
            logger.debug(
                "no_klines_data_for_date",
                symbol=symbol,
                date=date_str,
                error=str(e),
            )
            continue
        except Exception as e:
            dates_not_found.append(date_str)
            logger.warning(
                "find_available_data_range_date_error",
                symbol=symbol,
                date=date_str,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            continue
    
    logger.info(
        "find_available_data_range_dates_summary",
        symbol=symbol,
        dates_checked=dates_checked,
        dates_found=dates_found,
        dates_not_found=dates_not_found,
        data_frames_count=len(all_data),
    )
    
    if not all_data:
        logger.warning(
            "no_data_available_for_target_computation",
            symbol=symbol,
            target_timestamp=target_timestamp.isoformat(),
            target_date=target_date.isoformat(),
            previous_date=previous_date.isoformat(),
            dates_checked=dates_checked,
        )
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.debug(
        "find_available_data_range_combined",
        symbol=symbol,
        combined_rows=len(combined_df),
        columns=list(combined_df.columns) if not combined_df.empty else [],
    )
    
    # Normalize timestamps
    if "timestamp" in combined_df.columns:
        if combined_df["timestamp"].dtype.tz is None:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"]).dt.tz_localize(timezone.utc)
        else:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"]).dt.tz_convert(timezone.utc)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        
        # Find latest available timestamp
        latest_available_timestamp = combined_df["timestamp"].max()
        earliest_available_timestamp = combined_df["timestamp"].min()
        
        logger.info(
            "find_available_data_range_timestamps",
            symbol=symbol,
            earliest_available=earliest_available_timestamp.isoformat(),
            latest_available=latest_available_timestamp.isoformat(),
            target_timestamp=target_timestamp.isoformat(),
            total_rows=len(combined_df),
        )
        
        # Check if data is too old (beyond expected delay)
        now_utc = datetime.now(timezone.utc)
        data_age_seconds = (now_utc - latest_available_timestamp).total_seconds()
        max_allowed_age = max_expected_delay_seconds + max_lookback_seconds
        
        logger.debug(
            "find_available_data_range_age_check",
            symbol=symbol,
            now_utc=now_utc.isoformat(),
            latest_available=latest_available_timestamp.isoformat(),
            data_age_seconds=data_age_seconds,
            max_expected_delay_seconds=max_expected_delay_seconds,
            max_lookback_seconds=max_lookback_seconds,
            max_allowed_age=max_allowed_age,
        )
        
        if data_age_seconds > max_allowed_age:
            logger.warning(
                "data_too_old_for_target_computation",
                symbol=symbol,
                target_timestamp=target_timestamp.isoformat(),
                latest_available=latest_available_timestamp.isoformat(),
                data_age_seconds=data_age_seconds,
                max_allowed_age=max_allowed_age,
                max_lookback=max_lookback_seconds,
                max_expected_delay=max_expected_delay_seconds,
            )
            return None
        
        # Check if target_timestamp is available or needs adjustment
        target_vs_latest_diff = (target_timestamp - latest_available_timestamp).total_seconds()
        
        logger.debug(
            "find_available_data_range_timestamp_comparison",
            symbol=symbol,
            target_timestamp=target_timestamp.isoformat(),
            latest_available_timestamp=latest_available_timestamp.isoformat(),
            target_vs_latest_diff_seconds=target_vs_latest_diff,
            target_is_before_latest=target_timestamp <= latest_available_timestamp,
        )
        
        if target_timestamp <= latest_available_timestamp:
            # Data is available, use as-is
            timestamp_adjusted = False
            adjusted_target_timestamp = target_timestamp
            lookback_seconds_used = 0
            
            logger.info(
                "find_available_data_range_success_no_adjustment",
                symbol=symbol,
                target_timestamp=target_timestamp.isoformat(),
                latest_available=latest_available_timestamp.isoformat(),
            )
        else:
            # Data is not available, check if we can adjust
            gap_seconds = (target_timestamp - latest_available_timestamp).total_seconds()
            
            logger.debug(
                "find_available_data_range_gap_check",
                symbol=symbol,
                target_timestamp=target_timestamp.isoformat(),
                latest_available=latest_available_timestamp.isoformat(),
                gap_seconds=gap_seconds,
                max_lookback_seconds=max_lookback_seconds,
                gap_exceeds_max=gap_seconds > max_lookback_seconds,
            )
            
            if gap_seconds > max_lookback_seconds:
                logger.warning(
                    "target_timestamp_too_far_ahead",
                    symbol=symbol,
                    target_timestamp=target_timestamp.isoformat(),
                    latest_available=latest_available_timestamp.isoformat(),
                    gap_seconds=gap_seconds,
                    max_lookback=max_lookback_seconds,
                    gap_exceeds_max_by=gap_seconds - max_lookback_seconds,
                )
                return None
            
            # Adjust target_timestamp to latest available
            timestamp_adjusted = True
            adjusted_target_timestamp = latest_available_timestamp
            lookback_seconds_used = int(gap_seconds)
            
            logger.info(
                "target_timestamp_adjusted_for_data_availability",
                symbol=symbol,
                requested_timestamp=target_timestamp.isoformat(),
                adjusted_timestamp=adjusted_target_timestamp.isoformat(),
                lookback_seconds=lookback_seconds_used,
            )
        
        result = {
            "adjusted_target_timestamp": adjusted_target_timestamp,
            "latest_available_timestamp": latest_available_timestamp,
            "timestamp_adjusted": timestamp_adjusted,
            "lookback_seconds_used": lookback_seconds_used,
            "historical_data": combined_df,
        }
        
        logger.info(
            "find_available_data_range_success",
            symbol=symbol,
            adjusted_target_timestamp=adjusted_target_timestamp.isoformat(),
            latest_available_timestamp=latest_available_timestamp.isoformat(),
            timestamp_adjusted=timestamp_adjusted,
            lookback_seconds_used=lookback_seconds_used,
            historical_data_rows=len(combined_df),
        )
        
        return result
    
    logger.warning(
        "find_available_data_range_no_timestamp_column",
        symbol=symbol,
        target_timestamp=target_timestamp.isoformat(),
        columns=list(combined_df.columns) if not combined_df.empty else [],
    )
    return None


async def load_historical_data_for_target_computation(
    parquet_storage: ParquetStorage,
    symbol: str,
    prediction_timestamp: datetime,
    target_timestamp: datetime,
    buffer_seconds: int = 60,
) -> pd.DataFrame:
    """
    Загрузить исторические данные для вычисления таргета.
    
    Аналогично optimized_builder._compute_targets(), но для одного timestamp.
    
    Загружает данные в окне:
    - Начало: prediction_timestamp - buffer_seconds
    - Конец: target_timestamp + buffer_seconds
    
    Использует ParquetStorage.read_klines_range() (как в dataset builder).
    
    Args:
        parquet_storage: ParquetStorage instance
        symbol: Trading pair symbol
        prediction_timestamp: Timestamp when prediction was made
        target_timestamp: Timestamp for target computation
        buffer_seconds: Buffer window in seconds
    
    Returns:
        DataFrame with historical price data
    """
    # Normalize timestamps to UTC timezone-aware
    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc)
    else:
        prediction_timestamp = prediction_timestamp.astimezone(timezone.utc)
    
    if target_timestamp.tzinfo is None:
        target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
    else:
        target_timestamp = target_timestamp.astimezone(timezone.utc)
    
    # Calculate date range
    start_date = (prediction_timestamp - timedelta(seconds=buffer_seconds)).date()
    end_date = (target_timestamp + timedelta(seconds=buffer_seconds)).date()
    
    # Use the same logic as dataset builder
    price_df = await parquet_storage.read_klines_range(
        symbol, start_date, end_date
    )
    
    if not price_df.empty and "timestamp" in price_df.columns:
        # Normalize timestamps
        if price_df["timestamp"].dtype.tz is None:
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.tz_localize(timezone.utc)
        else:
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.tz_convert(timezone.utc)
        
        # Sort by timestamp
        price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    
    return price_df

