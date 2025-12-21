"""
Target computation API endpoints.

Provides endpoint for computing actual target values for model predictions.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import structlog

from src.services.target_registry_version_manager import TargetRegistryVersionManager
from src.storage.parquet_storage import ParquetStorage
from src.services.target_computation import TargetComputationEngine, TargetComputationPresets
from src.services.target_computation_data import (
    find_available_data_range,
    load_historical_data_for_target_computation,
)
from src.config import config
from .middleware.auth import verify_api_key

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/targets", tags=["targets"])

# Global instances (will be set during app startup)
_target_registry_version_manager: Optional[TargetRegistryVersionManager] = None
_parquet_storage: Optional[ParquetStorage] = None


def set_target_registry_version_manager(manager: TargetRegistryVersionManager) -> None:
    """Set global target registry version manager instance."""
    global _target_registry_version_manager
    _target_registry_version_manager = manager


def set_parquet_storage(storage: ParquetStorage) -> None:
    """Set global parquet storage instance."""
    global _parquet_storage
    _parquet_storage = storage


# Request/Response models
class TargetComputeRequest(BaseModel):
    """Request model for target computation."""
    symbol: str = Field(description="Trading pair symbol, e.g. BTCUSDT")
    prediction_timestamp: datetime = Field(description="Timestamp when prediction was made (UTC)")
    target_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp for target computation (UTC). If None, computed from prediction_timestamp + horizon_seconds"
    )
    target_registry_version: str = Field(description="Target Registry version")
    horizon_seconds: Optional[int] = Field(
        default=None,
        description="Optional horizon override (if None, uses horizon from registry config)"
    )
    max_lookback_seconds: int = Field(
        default=300,
        description="Maximum lookback window for data availability fallback (seconds)"
    )


class TargetComputeResponse(BaseModel):
    """Response model for target computation."""
    target_type: str = Field(description="Target type: regression, classification, or risk_adjusted")
    preset: str = Field(description="Computation preset: returns, next_candle_direction, sharpe_ratio, etc.")
    prediction_timestamp_used: datetime = Field(description="Actual prediction timestamp used")
    target_timestamp_used: datetime = Field(description="Actual target timestamp used")
    data_available: bool = Field(description="Whether data was available")
    timestamp_adjusted: bool = Field(description="Whether target_timestamp was adjusted due to data delays")
    lookback_seconds_used: int = Field(description="How many seconds we had to look back")
    computation_timestamp: datetime = Field(description="When computation was performed")
    
    # Dynamic fields based on target_type and preset
    # For regression (returns):
    target_value: Optional[float] = Field(default=None, description="Computed target value (for regression)")
    price_at_prediction: Optional[float] = Field(default=None, description="Price at prediction timestamp")
    price_at_target: Optional[float] = Field(default=None, description="Price at target timestamp")
    
    # For classification (next_candle_direction):
    direction: Optional[str] = Field(default=None, description="Direction: green or red (for classification)")
    candle_open: Optional[float] = Field(default=None, description="Candle open price (for classification)")
    candle_close: Optional[float] = Field(default=None, description="Candle close price (for classification)")
    return_value: Optional[float] = Field(default=None, description="Return value for compatibility (for classification)")
    
    # For risk_adjusted (sharpe_ratio):
    sharpe_value: Optional[float] = Field(default=None, description="Sharpe ratio value (for risk_adjusted)")
    returns_series: Optional[list] = Field(default=None, description="Returns series for calculation (for risk_adjusted)")
    volatility: Optional[float] = Field(default=None, description="Volatility (for risk_adjusted)")


class TargetComputeErrorResponse(BaseModel):
    """Error response model for target computation."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    requested_target_timestamp: Optional[datetime] = Field(default=None)
    latest_available_timestamp: Optional[datetime] = Field(default=None)
    max_lookback_seconds: Optional[int] = Field(default=None)
    data_gap_seconds: Optional[int] = Field(default=None)


@router.post("/compute", response_model=TargetComputeResponse)
async def compute_target(
    request: TargetComputeRequest,
    _: None = Depends(verify_api_key),
) -> TargetComputeResponse:
    """
    Compute actual target value for a prediction.
    
    This endpoint computes the actual target value that occurred in the market,
    allowing model-service to compare predictions with actual outcomes.
    
    The response structure dynamically adapts based on target_type and preset
    from the Target Registry configuration.
    """
    if _target_registry_version_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Target Registry Version Manager not initialized"
        )
    
    if _parquet_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Parquet Storage not initialized"
        )
    
    # Normalize timestamps to UTC
    prediction_timestamp = request.prediction_timestamp
    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc)
    else:
        prediction_timestamp = prediction_timestamp.astimezone(timezone.utc)
    
    # Load target_config from registry
    try:
        target_config_dict = await _target_registry_version_manager.get_version(request.target_registry_version)
        if not target_config_dict:
            raise HTTPException(
                status_code=404,
                detail=f"Target Registry version not found: {request.target_registry_version}"
            )
    except Exception as e:
        logger.error(
            "failed_to_load_target_config",
            version=request.target_registry_version,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load target config: {str(e)}"
        )
    
    # Extract target_type and computation config
    target_type = target_config_dict.get("type", "regression")
    computation_config_dict = target_config_dict.get("computation", {})
    preset = computation_config_dict.get("preset", "returns") if computation_config_dict else "returns"
    
    # Determine target_timestamp first
    if request.target_timestamp is not None:
        target_timestamp = request.target_timestamp
        if target_timestamp.tzinfo is None:
            target_timestamp = target_timestamp.replace(tzinfo=timezone.utc)
        else:
            target_timestamp = target_timestamp.astimezone(timezone.utc)
    else:
        # Compute from prediction_timestamp + horizon
        if request.horizon_seconds is not None:
            horizon_seconds = request.horizon_seconds
        else:
            horizon_seconds = target_config_dict.get("horizon", 60)
        target_timestamp = prediction_timestamp.replace(tzinfo=timezone.utc) + timedelta(seconds=horizon_seconds)
    
    # Determine effective horizon_seconds (actual time difference between prediction and target)
    # This is what TargetComputationEngine needs, not the config horizon
    effective_horizon_seconds = int((target_timestamp - prediction_timestamp).total_seconds())
    if effective_horizon_seconds <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"target_timestamp must be after prediction_timestamp. Got: prediction={prediction_timestamp.isoformat()}, target={target_timestamp.isoformat()}"
        )
    
    # Step 1: Find available data range (NEW level fallback)
    max_expected_delay = getattr(config, "target_computation_max_expected_delay_seconds", 30)
    data_range = await find_available_data_range(
        parquet_storage=_parquet_storage,
        symbol=request.symbol,
        target_timestamp=target_timestamp,
        max_lookback_seconds=request.max_lookback_seconds,
        max_expected_delay_seconds=max_expected_delay,
    )
    
    if not data_range:
        # Data unavailable
        raise HTTPException(
            status_code=404,
            detail={
                "error": "data_unavailable",
                "message": "No data available within max_lookback_seconds",
                "requested_target_timestamp": target_timestamp.isoformat(),
                "max_lookback_seconds": request.max_lookback_seconds,
            }
        )
    
    adjusted_target_timestamp = data_range["adjusted_target_timestamp"]
    historical_data = data_range["historical_data"]
    timestamp_adjusted = data_range["timestamp_adjusted"]
    lookback_seconds_used = data_range["lookback_seconds_used"]
    
    logger.info(
        "target_computation_data_range_found",
        symbol=request.symbol,
        prediction_timestamp=prediction_timestamp.isoformat(),
        target_timestamp=target_timestamp.isoformat(),
        adjusted_target_timestamp=adjusted_target_timestamp.isoformat(),
        timestamp_adjusted=timestamp_adjusted,
        lookback_seconds_used=lookback_seconds_used,
        historical_data_rows=len(historical_data),
    )
    
    # Step 2: Create prediction DataFrame (single timestamp)
    # Find price at prediction_timestamp from historical_data
    import pandas as pd
    import numpy as np
    
    # Find the closest price to prediction_timestamp in historical_data
    # For klines (1-minute candles), find the candle closest to prediction_timestamp
    price_at_prediction = None
    if not historical_data.empty and "close" in historical_data.columns:
        # Ensure timestamps are timezone-aware
        if historical_data["timestamp"].dtype == "object":
            historical_data["timestamp"] = pd.to_datetime(historical_data["timestamp"], utc=True)
        
        # Find the closest timestamp (can be before or after, but prefer before)
        time_diffs = (historical_data["timestamp"] - prediction_timestamp).abs()
        closest_idx = time_diffs.idxmin()
        closest_timestamp = historical_data.loc[closest_idx, "timestamp"]
        time_diff = time_diffs[closest_idx]
        
        logger.debug(
            "target_computation_price_search",
            symbol=request.symbol,
            prediction_timestamp=prediction_timestamp.isoformat(),
            closest_timestamp=closest_timestamp.isoformat() if hasattr(closest_timestamp, 'isoformat') else str(closest_timestamp),
            time_diff_seconds=time_diff.total_seconds(),
            time_diff_minutes=time_diff.total_seconds() / 60,
        )
        
        # Allow up to 10 minutes tolerance for klines (to handle data gaps)
        if time_diff <= pd.Timedelta(minutes=10):
            price_at_prediction = float(historical_data.loc[closest_idx, "close"])
            time_diff_seconds = time_diff.total_seconds()
            
            logger.info(
                "target_computation_price_found",
                symbol=request.symbol,
                prediction_timestamp=prediction_timestamp.isoformat(),
                price_timestamp=closest_timestamp.isoformat() if hasattr(closest_timestamp, 'isoformat') else str(closest_timestamp),
                time_diff_seconds=time_diff_seconds,
                price_at_prediction=price_at_prediction,
            )
        else:
            logger.warning(
                "target_computation_price_too_far",
                symbol=request.symbol,
                prediction_timestamp=prediction_timestamp.isoformat(),
                closest_timestamp=closest_timestamp.isoformat() if hasattr(closest_timestamp, 'isoformat') else str(closest_timestamp),
                time_diff_minutes=time_diff.total_seconds() / 60,
                max_tolerance_minutes=10,
            )
    
    if price_at_prediction is None:
        logger.warning(
            "target_computation_no_price_at_prediction",
            symbol=request.symbol,
            prediction_timestamp=prediction_timestamp.isoformat(),
            historical_data_timestamps_min=historical_data["timestamp"].min().isoformat() if not historical_data.empty else None,
            historical_data_timestamps_max=historical_data["timestamp"].max().isoformat() if not historical_data.empty else None,
            historical_data_rows=len(historical_data),
        )
        raise HTTPException(
            status_code=404,
            detail=f"Could not find price data at prediction_timestamp {prediction_timestamp.isoformat()}"
        )
    
    prediction_df = pd.DataFrame({
        "timestamp": [prediction_timestamp],
        "price": [price_at_prediction],
        "close": [price_at_prediction],  # Also add as 'close' for compatibility
    })
    
    # Ensure timestamp is timezone-aware
    if prediction_df["timestamp"].dtype == "object":
        prediction_df["timestamp"] = pd.to_datetime(prediction_df["timestamp"], utc=True)
    
    logger.info(
        "target_computation_prediction_df_created",
        symbol=request.symbol,
        prediction_timestamp=prediction_timestamp.isoformat(),
        price_at_prediction=price_at_prediction,
        prediction_df_columns=list(prediction_df.columns),
    )
    
    # Step 3: Get computation configuration
    from src.models.dataset import TargetComputationConfig
    computation_config = TargetComputationPresets.get_computation_config(
        TargetComputationConfig(**computation_config_dict) if computation_config_dict else None
    )
    
    logger.info(
        "target_computation_config_loaded",
        symbol=request.symbol,
        target_type=target_type,
        preset=preset,
        effective_horizon_seconds=effective_horizon_seconds,
        config_horizon_seconds=target_config_dict.get("horizon"),
        request_horizon_seconds=request.horizon_seconds,
        computation_config_keys=list(computation_config.keys()) if computation_config else [],
    )
    
    # Step 4: Find future price at target_timestamp
    # For single prediction point, we can find the price directly instead of using merge_asof
    price_at_target = None
    if not historical_data.empty and "close" in historical_data.columns:
        # Find price at or after target_timestamp (adjusted)
        target_data = historical_data[historical_data["timestamp"] >= adjusted_target_timestamp]
        
        if not target_data.empty:
            # Use the first available price at or after target_timestamp
            price_at_target = float(target_data.iloc[0]["close"])
            target_price_timestamp = target_data.iloc[0]["timestamp"]
        else:
            # If no data at target_timestamp, find closest
            time_diffs = (historical_data["timestamp"] - adjusted_target_timestamp).abs()
            closest_idx = time_diffs.idxmin()
            time_diff = time_diffs[closest_idx]
            
            if time_diff <= pd.Timedelta(minutes=5):
                price_at_target = float(historical_data.loc[closest_idx, "close"])
                target_price_timestamp = historical_data.loc[closest_idx, "timestamp"]
        
        if price_at_target is None:
            logger.warning(
                "target_computation_no_price_at_target",
                symbol=request.symbol,
                target_timestamp=adjusted_target_timestamp.isoformat(),
                historical_data_timestamps_min=historical_data["timestamp"].min().isoformat() if not historical_data.empty else None,
                historical_data_timestamps_max=historical_data["timestamp"].max().isoformat() if not historical_data.empty else None,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Could not find price data at target_timestamp {adjusted_target_timestamp.isoformat()}"
            )
        
        logger.info(
            "target_computation_future_price_found",
            symbol=request.symbol,
            target_timestamp=adjusted_target_timestamp.isoformat(),
            price_timestamp=target_price_timestamp.isoformat() if hasattr(target_price_timestamp, 'isoformat') else str(target_price_timestamp),
            price_at_target=price_at_target,
            price_at_prediction=price_at_prediction,
        )
    
    # Step 5: Compute target value directly
    # For returns: (future_price - price) / price
    # For classification (next_candle_direction): same as returns, but will be mapped to classes
    formula = computation_config.get("formula", "returns")
    
    if formula == "returns":
        target_value = (price_at_target - price_at_prediction) / price_at_prediction
    elif formula == "log_returns":
        target_value = np.log(price_at_target / price_at_prediction)
    elif formula == "price_change":
        target_value = price_at_target - price_at_prediction
    else:
        # For other formulas (sharpe, volatility_normalized), use TargetComputationEngine
        # But for now, fallback to returns
        logger.warning(
            "target_computation_unsupported_formula_fallback",
            symbol=request.symbol,
            formula=formula,
            falling_back_to="returns",
        )
        target_value = (price_at_target - price_at_prediction) / price_at_prediction
    
    # Create targets_df manually for compatibility
    targets_df = pd.DataFrame({
        "timestamp": [prediction_timestamp],
        "target": [target_value],
    })
    
    logger.info(
        "target_computation_target_computed",
        symbol=request.symbol,
        prediction_timestamp=prediction_timestamp.isoformat(),
        target_timestamp=adjusted_target_timestamp.isoformat(),
        target_value=target_value,
        formula=formula,
    )
    
    # target_value, price_at_prediction, and price_at_target are already computed above
    
    # Step 6: Format response based on target_type and preset
    computation_timestamp = datetime.now(timezone.utc)
    
    response_data = {
        "target_type": target_type,
        "preset": preset,
        "prediction_timestamp_used": prediction_timestamp,
        "target_timestamp_used": adjusted_target_timestamp,
        "data_available": True,
        "timestamp_adjusted": timestamp_adjusted,
        "lookback_seconds_used": lookback_seconds_used,
        "computation_timestamp": computation_timestamp,
    }
    
    if target_type == "regression" and preset == "returns":
        response_data.update({
            "target_value": target_value,
            "price_at_prediction": price_at_prediction,
            "price_at_target": price_at_target,
        })
    elif target_type == "classification" and preset == "next_candle_direction":
        # Determine direction from target_value (return)
        direction = "green" if target_value > 0 else "red"
        response_data.update({
            "direction": direction,
            "candle_open": price_at_prediction,
            "candle_close": price_at_target,
            "return_value": target_value,  # For compatibility
            "target_value": target_value,  # Also include for consistency
            "price_at_prediction": price_at_prediction,  # Also include for consistency
            "price_at_target": price_at_target,  # Also include for consistency
        })
    elif target_type == "risk_adjusted" and preset == "sharpe_ratio":
        # For sharpe_ratio, we need additional data
        # This is a simplified version - full implementation would extract returns_series
        response_data.update({
            "sharpe_value": target_value,
            # returns_series and volatility would need to be extracted from computation
            # For now, we return basic structure
        })
    else:
        # Fallback: return basic structure
        response_data.update({
            "target_value": target_value,
        })
    
    return TargetComputeResponse(**response_data)

