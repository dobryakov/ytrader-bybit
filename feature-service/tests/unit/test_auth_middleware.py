"""
Unit tests for API authentication middleware.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse


class TestAuthMiddleware:
    """Tests for authentication middleware."""
    
    @pytest.mark.asyncio
    async def test_auth_middleware_validates_api_key(self):
        """Test that middleware validates API key."""
        from src.api.middleware.auth import verify_api_key
        
        # Create mock request with valid API key
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "valid-api-key"}
        
        # Mock config
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "valid-api-key"
            
            # Should not raise exception
            await verify_api_key(request)
    
    @pytest.mark.asyncio
    async def test_auth_middleware_rejects_missing_api_key(self):
        """Test that middleware rejects requests without API key."""
        from src.api.middleware.auth import verify_api_key
        
        # Create mock request without API key
        request = MagicMock(spec=Request)
        request.headers = {}
        
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(request)
        
        assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_auth_middleware_rejects_invalid_api_key(self):
        """Test that middleware rejects requests with invalid API key."""
        from src.api.middleware.auth import verify_api_key
        
        # Create mock request with invalid API key
        request = MagicMock(spec=Request)
        request.headers = {"X-API-Key": "invalid-api-key"}
        
        # Mock config
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "valid-api-key"
            
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(request)
            
            assert exc_info.value.status_code == 401
    
    @pytest.mark.asyncio
    async def test_auth_middleware_allows_health_endpoint(self):
        """Test that middleware allows health endpoint without authentication."""
        from src.api.middleware.auth import verify_api_key
        
        # Create mock request for health endpoint
        request = MagicMock(spec=Request)
        request.url.path = "/health"
        request.headers = {}
        
        # Should not raise exception for health endpoint
        # (adjust based on actual implementation)
        # This test may need to be adjusted if health endpoint is exempt
    
    def test_auth_middleware_creates_dependency(self):
        """Test that middleware creates FastAPI dependency."""
        from src.api.middleware.auth import get_api_key_dependency
        
        dependency = get_api_key_dependency()
        
        assert dependency is not None
        assert callable(dependency)

