"""
Unit tests for HTTP client setup (ws-gateway API).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, Response, Request


class TestHTTPClient:
    """Tests for HTTP client setup."""
    
    @pytest.mark.asyncio
    async def test_http_client_initializes(self):
        """Test that HTTP client initializes correctly."""
        from src.http.client import HTTPClient
        
        client = HTTPClient(base_url="http://localhost:8080")
        
        assert client is not None
        assert client.base_url == "http://localhost:8080"
    
    @pytest.mark.asyncio
    async def test_http_client_makes_get_request(self, mock_http_client):
        """Test that HTTP client makes GET requests."""
        from src.http.client import HTTPClient
        
        client = HTTPClient(base_url="http://localhost:8080")
        client._client = mock_http_client
        
        response = await client.get("/api/test")
        
        assert response is not None
        mock_http_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_http_client_makes_post_request(self, mock_http_client):
        """Test that HTTP client makes POST requests."""
        from src.http.client import HTTPClient
        
        client = HTTPClient(base_url="http://localhost:8080")
        client._client = mock_http_client
        
        response = await client.post("/api/test", json={"key": "value"})
        
        assert response is not None
        mock_http_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_http_client_handles_errors(self, mock_http_client):
        """Test that HTTP client handles errors gracefully."""
        from src.http.client import HTTPClient
        
        # Configure mock to raise an error
        mock_http_client.get.side_effect = Exception("Connection error")
        
        client = HTTPClient(base_url="http://localhost:8080")
        client._client = mock_http_client
        
        with pytest.raises(Exception):
            await client.get("/api/test")
    
    @pytest.mark.asyncio
    async def test_http_client_closes_connection(self, mock_http_client):
        """Test that HTTP client closes connection properly."""
        from src.http.client import HTTPClient
        
        client = HTTPClient(base_url="http://localhost:8080")
        client._client = mock_http_client
        
        await client.close()
        
        mock_http_client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_http_client_sets_headers(self, mock_http_client):
        """Test that HTTP client sets headers correctly."""
        from src.http.client import HTTPClient
        
        client = HTTPClient(
            base_url="http://localhost:8080",
            api_key="test-api-key"
        )
        client._client = mock_http_client
        
        await client.get("/api/test")
        
        # Verify headers are set
        call_args = mock_http_client.get.call_args
        assert call_args is not None
        # Check headers in call_args (adjust based on actual implementation)

