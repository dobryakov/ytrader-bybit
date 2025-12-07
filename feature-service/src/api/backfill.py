"""
Backfilling API endpoints for historical data fetching.
"""
from datetime import date, datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.middleware.auth import verify_api_key
from src.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/backfill", tags=["backfill"])

# Global backfilling service instance (set by main.py)
_backfilling_service = None


def set_backfilling_service(service):
    """Set backfilling service instance."""
    global _backfilling_service
    _backfilling_service = service


class BackfillHistoricalRequest(BaseModel):
    """Request model for historical backfilling."""
    
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    start_date: date = Field(..., description="Start date for backfilling (YYYY-MM-DD)")
    end_date: date = Field(..., description="End date for backfilling (YYYY-MM-DD)")
    data_types: Optional[List[str]] = Field(
        default=None,
        description="Optional list of data types to backfill (if not provided, uses Feature Registry)"
    )


class BackfillAutoRequest(BaseModel):
    """Request model for automatic backfilling."""
    
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    max_days: Optional[int] = Field(
        default=None,
        description="Maximum days to backfill (uses config default if not provided)"
    )


@router.post("/historical", status_code=202)
async def backfill_historical(
    request: BackfillHistoricalRequest,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Backfill historical data for a symbol and date range.
    
    Returns:
        Backfill job ID for tracking progress
    """
    if _backfilling_service is None:
        raise HTTPException(status_code=503, detail="Backfilling service not initialized")
    
    try:
        job_id = await _backfilling_service.backfill_historical(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            data_types=request.data_types,
        )
        
        logger.info(
            "backfill_historical_requested",
            job_id=job_id,
            symbol=request.symbol,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            data_types=request.data_types,
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Backfilling job started",
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("backfill_historical_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/auto", status_code=202)
async def backfill_auto(
    request: BackfillAutoRequest,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Automatically backfill missing data for a symbol up to configured maximum days.
    
    Uses Feature Registry to determine which data types to backfill.
    
    Returns:
        Backfill job ID for tracking progress
    """
    if _backfilling_service is None:
        raise HTTPException(status_code=503, detail="Backfilling service not initialized")
    
    from src.config import config
    
    # Calculate date range
    end_date = date.today()
    max_days = request.max_days or config.feature_service_backfill_max_days
    start_date = end_date - timedelta(days=max_days)
    
    try:
        job_id = await _backfilling_service.backfill_historical(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            data_types=None,  # Use Feature Registry to determine
        )
        
        logger.info(
            "backfill_auto_requested",
            job_id=job_id,
            symbol=request.symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            max_days=max_days,
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Automatic backfilling job started",
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("backfill_auto_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}")
async def get_backfill_status(
    job_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Get backfilling job status.
    
    Returns:
        Job status with progress information
    """
    if _backfilling_service is None:
        raise HTTPException(status_code=503, detail="Backfilling service not initialized")
    
    status = _backfilling_service.get_job_status(job_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return status

