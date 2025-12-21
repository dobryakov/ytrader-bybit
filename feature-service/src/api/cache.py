"""
Cache management API endpoints.
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from .middleware.auth import verify_api_key
from src.services.cache_service import CacheService
from src.services.cache_invalidation import CacheInvalidationService

router = APIRouter(prefix="/cache", tags=["cache"])

# Global cache service instance
_cache_service: Optional[CacheService] = None
_cache_invalidation_service: Optional[CacheInvalidationService] = None


def set_cache_service(cache_service: CacheService) -> None:
    """Set cache service instance."""
    global _cache_service
    _cache_service = cache_service
    
    # Initialize cache invalidation service
    global _cache_invalidation_service
    _cache_invalidation_service = CacheInvalidationService(cache_service=cache_service)


class CacheInvalidateRequest(BaseModel):
    """Request model for cache invalidation."""
    pattern: Optional[str] = None
    symbol: Optional[str] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


@router.get("/statistics")
async def get_cache_statistics(
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Get cache statistics.
    
    Returns cache hit rate, size, eviction count, and other metrics.
    """
    if not _cache_service:
        raise HTTPException(
            status_code=503,
            detail="Cache service not available",
        )
    
    try:
        stats = await _cache_service.get_statistics()
        return {
            "status": "success",
            "statistics": stats,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}",
        )


@router.post("/invalidate")
async def invalidate_cache(
    request: CacheInvalidateRequest,
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Manually invalidate cache entries.
    
    Parameters:
    - pattern: Cache key pattern to match (e.g., "historical_data:*" or "features:*")
    - symbol: Symbol to invalidate (invalidates all entries for this symbol)
    - date_range_start: Start date of date range to invalidate
    - date_range_end: End date of date range to invalidate
    
    If no parameters provided, invalidates all cache entries.
    """
    if not _cache_invalidation_service:
        raise HTTPException(
            status_code=503,
            detail="Cache invalidation service not available",
        )
    
    try:
        date_range = None
        if request.date_range_start and request.date_range_end:
            date_range = (request.date_range_start, request.date_range_end)
        
        count = await _cache_invalidation_service.invalidate_manual(
            pattern=request.pattern,
            symbol=request.symbol,
            date_range=date_range,
        )
        
        return {
            "status": "success",
            "invalidated_count": count,
            "message": f"Invalidated {count} cache entries",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to invalidate cache: {str(e)}",
        )


@router.post("/warm")
async def warm_cache(
    symbol: str = Query(..., description="Symbol to warm cache for"),
    start_date: datetime = Query(..., description="Start date for cache warming"),
    end_date: datetime = Query(..., description="End date for cache warming"),
    api_key: str = Depends(verify_api_key),
) -> dict:
    """
    Pre-populate (warm) cache for frequently used periods.
    
    This endpoint triggers dataset building for the specified period,
    which will populate the cache with historical data and computed features.
    """
    # Cache warming is implemented by triggering a dataset build
    # The dataset builder will populate the cache automatically
    # This is a placeholder - actual implementation would trigger dataset build
    return {
        "status": "success",
        "message": "Cache warming triggered (placeholder - requires dataset builder integration)",
        "symbol": symbol,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

