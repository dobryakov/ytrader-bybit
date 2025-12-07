"""
Unit tests for Bybit REST API client.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from datetime import datetime

from src.utils.bybit_client import BybitClient, BybitAPIError


class TestBybitClient:
    """Tests for Bybit REST API client."""
    
    @pytest.fixture
    def bybit_client(self):
        """Create Bybit client for testing."""
        return BybitClient(
            api_key=None,
            api_secret=None,
            base_url="https://api-testnet.bybit.com",
            rate_limit_delay_ms=0,  # No delay for tests
        )
    
    @pytest.fixture
    def authenticated_client(self):
        """Create authenticated Bybit client for testing."""
        return BybitClient(
            api_key="test_api_key",
            api_secret="test_api_secret",
            base_url="https://api-testnet.bybit.com",
            rate_limit_delay_ms=0,
        )
    
    @pytest.mark.asyncio
    async def test_client_init(self, bybit_client):
        """Test initializing Bybit client."""
        assert bybit_client.base_url == "https://api-testnet.bybit.com"
        assert bybit_client.api_key is None
        assert bybit_client.api_secret is None
    
    @pytest.mark.asyncio
    async def test_public_endpoint_request(self, bybit_client):
        """Test public endpoint request (no authentication)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        }
        
        with patch.object(bybit_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            
            result = await bybit_client.get("/v5/market/kline", params={"symbol": "BTCUSDT"})
            
            assert result["retCode"] == 0
            mock_client.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_authenticated_endpoint_request(self, authenticated_client):
        """Test authenticated endpoint request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {},
        }
        
        with patch.object(authenticated_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            
            result = await authenticated_client.get("/v5/account/wallet-balance", authenticated=True)
            
            assert result["retCode"] == 0
            # Verify authentication headers were added
            call_kwargs = mock_client.request.call_args[1]
            assert "headers" in call_kwargs
            assert "X-BAPI-API-KEY" in call_kwargs["headers"]
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, bybit_client):
        """Test retry logic for 429 rate limit errors."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {},
        }
        
        with patch.object(bybit_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=[mock_response_429, mock_response_200])
            mock_get_client.return_value = mock_client
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await bybit_client.get("/v5/market/kline")
                
                assert result["retCode"] == 0
                assert mock_client.request.call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limit_max_retries(self, bybit_client):
        """Test that rate limit error is raised after max retries."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        
        with patch.object(bybit_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response_429)
            mock_get_client.return_value = mock_client
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(BybitAPIError, match="Rate limit exceeded"):
                    await bybit_client.get("/v5/market/kline")
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, bybit_client):
        """Test timeout handling."""
        with patch.object(bybit_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("Request timeout"))
            mock_get_client.return_value = mock_client
            
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(BybitAPIError, match="Request failed"):
                    await bybit_client.get("/v5/market/kline")
    
    @pytest.mark.asyncio
    async def test_error_parsing(self, bybit_client):
        """Test error parsing from Bybit API responses."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "retCode": 10001,
            "retMsg": "Invalid symbol",
            "result": None,
        }
        
        with patch.object(bybit_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client
            
            with pytest.raises(BybitAPIError, match="Invalid symbol"):
                await bybit_client.get("/v5/market/kline")
    
    @pytest.mark.asyncio
    async def test_signature_generation(self, authenticated_client):
        """Test HMAC-SHA256 signature generation."""
        timestamp = 1234567890000
        recv_window = "5000"
        params = {"symbol": "BTCUSDT", "interval": "1"}
        
        signature = authenticated_client._generate_signature_for_get(params, timestamp, recv_window)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length
    
    @pytest.mark.asyncio
    async def test_close_client(self, bybit_client):
        """Test closing HTTP client."""
        mock_client = AsyncMock()
        bybit_client._client = mock_client  # Set client directly
        await bybit_client.close()
        mock_client.aclose.assert_called_once()
        assert bybit_client._client is None

