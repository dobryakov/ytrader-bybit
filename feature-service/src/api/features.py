"""
Features API endpoints.
"""
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Query
import structlog

from src.models.feature_vector import FeatureVector
from src.models.features_response import FeaturesResponse
from src.services.feature_computer import FeatureComputer

if TYPE_CHECKING:
    from src.services.feature_computer_manager import FeatureComputerManager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/features", tags=["features"])

# Global feature computer instance (will be set during app startup)
_feature_computer: Optional[FeatureComputer] = None
# Global feature computer manager (for versioned feature computation)
_feature_computer_manager: Optional["FeatureComputerManager"] = None


def set_feature_computer(computer: FeatureComputer) -> None:
    """Set global feature computer instance."""
    global _feature_computer
    _feature_computer = computer


def set_feature_computer_manager(manager: "FeatureComputerManager") -> None:
    """Set global feature computer manager instance."""
    global _feature_computer_manager
    _feature_computer_manager = manager


@router.get("/latest")
async def get_latest_features(
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTCUSDT)"),
    feature_registry_version: Optional[str] = Query(None, description="Feature Registry version (default: active version)"),
) -> FeaturesResponse:
    """
    Get latest computed features for a symbol.
    
    If feature_registry_version is provided, computes features using that version.
    Otherwise uses active Feature Registry version.
    
    Returns 404 if features are not available for the symbol.
    """
    # Prefer manager if available (new architecture with versioning)
    if _feature_computer_manager is not None:
        try:
            # Get FeatureComputer for specified version (or active if None)
            feature_computer = await _feature_computer_manager.get_or_create_computer(
                feature_registry_version=feature_registry_version
            )
            
            feature_vector, market_data = feature_computer.compute_features(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
            )
            
            return FeaturesResponse(
                feature_vector=feature_vector,
                market_data=market_data,
            )
        
        except ValueError as e:
            # Version not found
            logger.error(
                "get_latest_features_version_not_found",
                symbol=symbol,
                feature_registry_version=feature_registry_version,
                error=str(e),
            )
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry version not found: {feature_registry_version}",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "get_latest_features_error",
                symbol=symbol,
                feature_registry_version=feature_registry_version,
                error=str(e),
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail="Internal server error")
    
    # Fallback to legacy feature_computer (backward compatibility)
    if _feature_computer is None:
        raise HTTPException(status_code=503, detail="Feature computer not available")
    
    # If version specified but no manager, log warning
    if feature_registry_version is not None:
        logger.warning(
            "get_latest_features_version_ignored",
            symbol=symbol,
            feature_registry_version=feature_registry_version,
            message="Feature registry versioning not available (using legacy mode)",
        )
    
    try:
        feature_vector, market_data = _feature_computer.compute_features(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )
        
        return FeaturesResponse(
            feature_vector=feature_vector,
            market_data=market_data,
        )
    
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

