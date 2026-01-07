"""Historical data routes - proxy to Feature Service."""

import httpx
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from ...config.settings import settings
from ...config.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/historical/klines")
async def get_historical_klines(
    symbol: str = Query(..., description="Trading symbol, e.g. BTCUSDT"),
    start_time: datetime = Query(..., description="Start timestamp (UTC)"),
    end_time: datetime = Query(..., description="End timestamp (UTC)"),
    interval: int = Query(1, description="Kline interval in minutes (default: 1)"),
):
    """
    Get historical kline (candlestick) data from Feature Service.
    
    This endpoint proxies requests to Feature Service historical API.
    """
    try:
        # Build Feature Service URL
        feature_service_url = f"http://{settings.feature_service_host}:{settings.feature_service_port}"
        url = f"{feature_service_url}/api/v1/historical/klines"
        
        # Prepare query parameters
        params = {
            "symbol": symbol,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "interval": interval,
        }
        
        # Make request to Feature Service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                url,
                params=params,
                headers={"X-API-Key": settings.feature_service_api_key},
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        logger.error(
            "feature_service_request_failed",
            status_code=e.response.status_code,
            response_text=e.response.text,
        )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Feature Service error: {e.response.text}",
        )
    except httpx.RequestError as e:
        logger.error("feature_service_connection_error", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Failed to connect to Feature Service",
        )
    except Exception as e:
        logger.error("unexpected_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

