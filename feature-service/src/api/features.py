"""
Features API endpoints.
"""
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import structlog

from src.models.feature_vector import FeatureVector
from src.services.feature_computer import FeatureComputer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/features", tags=["features"])

# Global feature computer instance (will be set during app startup)
_feature_computer: Optional[FeatureComputer] = None


def set_feature_computer(computer: FeatureComputer) -> None:
    """Set global feature computer instance."""
    global _feature_computer
    _feature_computer = computer


@router.get("/latest")
async def get_latest_features(
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTCUSDT)"),
) -> FeatureVector:
    """
    Get latest computed features for a symbol.
    
    Returns 404 if features are not available for the symbol.
    """
    if _feature_computer is None:
        raise HTTPException(status_code=503, detail="Feature computer not available")
    
    try:
        feature_vector = _feature_computer.compute_features(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )
        
        if feature_vector is None:
            raise HTTPException(
                status_code=404,
                detail=f"Features not available for symbol: {symbol}",
            )
        
        return feature_vector
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_latest_features_error",
            symbol=symbol,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/debug/rolling-windows")
async def get_rolling_windows_debug(
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTCUSDT)"),
) -> dict:
    """
    Get debug information about RollingWindows state for a symbol.
    """
    if _feature_computer is None:
        raise HTTPException(status_code=503, detail="Feature computer not available")
    
    try:
        rolling_windows = _feature_computer.get_rolling_windows(symbol)
        klines_df = rolling_windows.get_window_data("1m")
        
        # Get first and last timestamps
        first_ts = None
        last_ts = None
        klines_count = len(klines_df)
        
        if klines_count > 0 and "timestamp" in klines_df.columns:
            try:
                sorted_klines = klines_df.sort_values("timestamp")
                first_ts = str(sorted_klines.iloc[0]["timestamp"]) if len(sorted_klines) > 0 else None
                last_ts = str(sorted_klines.iloc[-1]["timestamp"]) if len(sorted_klines) > 0 else None
            except Exception as e:
                logger.warning("failed_to_get_timestamps", error=str(e))
        
        return {
            "symbol": symbol,
            "klines_count": klines_count,
            "first_kline_timestamp": first_ts,
            "last_kline_timestamp": last_ts,
            "last_update": rolling_windows.last_update.isoformat() if isinstance(rolling_windows.last_update, datetime) else str(rolling_windows.last_update),
            "max_lookback_minutes_1m": rolling_windows.max_lookback_minutes_1m,
            "window_intervals": list(rolling_windows.window_intervals) if rolling_windows.window_intervals else None,
        }
    
    except Exception as e:
        logger.error(
            "get_rolling_windows_debug_error",
            symbol=symbol,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

