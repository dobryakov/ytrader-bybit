"""
Security middleware for path traversal protection and request validation.
"""
from fastapi import Request, HTTPException, status
from pathlib import Path
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


def validate_path_safe(base_path: Path, file_path: Path) -> bool:
    """
    Validate that file_path is within base_path (prevents path traversal).
    
    Args:
        base_path: Base directory that file_path must be within
        file_path: Path to validate
        
    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Resolve both paths to absolute
        base_resolved = base_path.resolve()
        file_resolved = file_path.resolve()
        
        # Check if file_path is within base_path
        # Use try_relative_to to check if file_path is a subpath of base_path
        try:
            file_resolved.relative_to(base_resolved)
            return True
        except ValueError:
            # Path is outside base_path
            return False
    except (OSError, ValueError) as e:
        logger.warning("Path validation error", base_path=str(base_path), file_path=str(file_path), error=str(e))
        return False


def validate_version_string(version: str) -> bool:
    """
    Validate model version string to prevent path traversal.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if version is safe, False otherwise
    """
    if not version:
        return False
    
    # Check for path traversal patterns
    if is_path_traversal_attempt(version):
        return False
    
    # Check for other dangerous characters
    dangerous_chars = ["/", "\\", "\x00"]
    for char in dangerous_chars:
        if char in version:
            return False
    
    return True


async def security_middleware(request: Request, call_next):
    """
    Security middleware to protect against path traversal and other attacks.
    
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

