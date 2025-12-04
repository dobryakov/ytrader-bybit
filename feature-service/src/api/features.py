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

