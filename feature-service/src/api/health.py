"""
Health check endpoint.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from typing import Dict, Any
from .middleware.auth import verify_api_key
from src.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(
    # Note: Health endpoint does not require authentication
    # verify_api_key: None = Depends(verify_api_key)  # Commented out for health endpoint
) -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict with health status and service information
    """
    return {
        "status": "healthy",
        "service": "feature-service",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

