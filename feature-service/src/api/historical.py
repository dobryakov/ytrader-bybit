"""
Historical price API.

Минимальная реализация для соответствия спецификации trading-chain.md:
- эндпоинт GET /api/v1/historical/price
- пока возвращает 501 Not Implemented, чтобы не ломать существующий функционал.

Дальше можно доработать, используя Parquet-хранилище и DataStorageService.
"""

from datetime import datetime, date, timedelta
from typing import Optional, List
import pandas as pd

from fastapi import APIRouter, HTTPException, Query, Depends
import structlog

from src.storage.parquet_storage import ParquetStorage
from src.config import config
from src.api.middleware.auth import verify_api_key

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/historical", tags=["historical"])

# Global ParquetStorage instance
_parquet_storage: Optional[ParquetStorage] = None


def get_parquet_storage() -> ParquetStorage:
    """Get or create ParquetStorage instance."""
    global _parquet_storage
    if _parquet_storage is None:
        _parquet_storage = ParquetStorage(base_path=config.feature_service_raw_data_path)
    return _parquet_storage


@router.get("/klines")
async def get_historical_klines(
    symbol: str = Query(..., description="Trading symbol, e.g. BTCUSDT"),
    start_time: datetime = Query(..., description="Start timestamp (UTC)"),
    end_time: datetime = Query(..., description="End timestamp (UTC)"),
    interval: int = Query(1, description="Kline interval in minutes (default: 1)"),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Get historical kline (candlestick) data from Parquet storage.
    
    Returns klines for the specified time range. Data is read from Parquet files
    organized by date (YYYY-MM-DD).
    """
    try:
        parquet_storage = get_parquet_storage()
        
        # Convert timestamps to dates
        start_date = start_time.date()
        end_date = end_time.date()
        
        # Collect klines from all dates in range
        all_klines = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.isoformat()
            try:
                df = await parquet_storage.read_klines(symbol, date_str)
                
                if not df.empty:
                    # Ensure timestamp column exists and is datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    elif 'start' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['start'])
                    
                    # Filter by time range
                    mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
                    filtered_df = df[mask]
                    
                    if not filtered_df.empty:
                        all_klines.append(filtered_df)
            except FileNotFoundError:
                # Date file doesn't exist, skip
                logger.debug(f"Klines file not found for {symbol} on {date_str}")
            except Exception as e:
                logger.warning(f"Error reading klines for {symbol} on {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        if not all_klines:
            return {
                "symbol": symbol,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval": interval,
                "klines": [],
                "count": 0
            }
        
        # Concatenate all dataframes
        combined_df = pd.concat(all_klines, ignore_index=True)
        
        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp')
        
        # Filter by interval if data has interval column
        if 'interval' in combined_df.columns:
            # Convert interval to match format (e.g., "1m" -> 1, "5m" -> 5)
            # Interval in data might be stored as "1m", "5m", etc. or as integer
            def normalize_interval(interval_val):
                if pd.isna(interval_val):
                    return None
                if isinstance(interval_val, (int, float)):
                    return int(interval_val)
                if isinstance(interval_val, str):
                    # Remove 'm' suffix if present
                    interval_str = interval_val.rstrip('m')
                    try:
                        return int(interval_str)
                    except ValueError:
                        return None
                return None
            
            combined_df['interval_normalized'] = combined_df['interval'].apply(normalize_interval)
            # Filter to requested interval
            combined_df = combined_df[combined_df['interval_normalized'] == interval]
            combined_df = combined_df.drop(columns=['interval_normalized'])
        
        # Remove duplicates by timestamp (keep first occurrence)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Convert to list of dicts
        klines_list = []
        for _, row in combined_df.iterrows():
            kline = {
                "timestamp": row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
                "open": float(row['open']) if 'open' in row and pd.notna(row['open']) else None,
                "high": float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
                "low": float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
                "close": float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                "volume": float(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None,
            }
            klines_list.append(kline)
        
        return {
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "interval": interval,
            "klines": klines_list,
            "count": len(klines_list)
        }
        
    except Exception as e:
        logger.error(f"Error getting historical klines: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve historical klines: {str(e)}")


@router.get("/price")
async def get_historical_price(
    symbol: str = Query(..., description="Trading symbol, e.g. BTCUSDT"),
    timestamp: datetime = Query(..., description="Target timestamp (UTC)"),
    lookback_seconds: int = Query(
        60,
        description="Lookback window in seconds for searching closest price",
        gt=0,
    ),
) -> dict:
    """
    Placeholder endpoint for historical price lookup.

    TODO:
    - Использовать DataStorageService/ParquetStorage для поиска цены.
    - Вернуть структуру с ценой(ами), на основе которых можно считать actual targets.
    """
    logger.info(
        "historical_price_not_implemented_yet",
        symbol=symbol,
        timestamp=timestamp.isoformat(),
        lookback_seconds=lookback_seconds,
    )
    raise HTTPException(
        status_code=501,
        detail="Historical price API not implemented yet",
    )


