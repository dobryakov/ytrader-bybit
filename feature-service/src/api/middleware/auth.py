"""
API authentication middleware.
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from ...config import config
from ...logging import get_logger

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def verify_api_key(request: Request) -> None:
    """
    Verify API key from request headers.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Allow health endpoint without authentication
    if request.url.path == "/health" or request.url.path == "/":
        return
    
    # Get API key from headers
    api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        logger.warning("API key missing", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if api_key != config.feature_service_api_key:
        logger.warning("Invalid API key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


def get_api_key_dependency():
    """
    Get FastAPI dependency for API key verification.
    
    Returns:
        Callable: Dependency function
    """
    return verify_api_key

