"""
Security middleware for path traversal protection and request validation.
"""
from fastapi import Request, HTTPException, status
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import Callable

from ...config.logging import get_logger

logger = get_logger(__name__)


def is_path_traversal_attempt(path: str) -> bool:
    """
    Check if path contains path traversal sequences.
    
    Args:
        path: Path to check
        
    Returns:
        True if path traversal detected, False otherwise
    """
    if not path:
        return False
    
    # Normalize path and check for traversal
    try:
        normalized = Path(path).as_posix()
    except (OSError, ValueError):
        # Invalid path
        return True
    
    # Check for common path traversal patterns
    dangerous_patterns = [
        "..",
        "//",
        "~",
        "\x00",  # Null byte
    ]
    
    # Check for patterns
    for pattern in dangerous_patterns:
        if pattern in normalized:
            return True
    
    return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware to protect against path traversal and other attacks."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security checks.
        
        Checks:
        - Path traversal attempts in URL path
        - Invalid characters in path
        """
        path = request.url.path
        raw_path = request.url.raw_path.decode('utf-8') if hasattr(request.url, 'raw_path') else path
        
        # Log suspicious paths for debugging
        if ".." in raw_path or ".." in path:
            logger.warning(
                "Suspicious path detected",
                path=path,
                raw_path=raw_path,
                client_ip=request.client.host if request.client else None,
            )
        
        # Check for path traversal in URL path (check both normalized and raw)
        if is_path_traversal_attempt(path) or is_path_traversal_attempt(raw_path):
            logger.warning(
                "Path traversal attempt detected",
                path=path,
                client_ip=request.client.host if request.client else None,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path"
            )
        
        # Check for null bytes
        if "\x00" in path:
            logger.warning(
                "Null byte detected in path",
                path=path,
                client_ip=request.client.host if request.client else None,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path"
            )
        
        response = await call_next(request)
        return response

