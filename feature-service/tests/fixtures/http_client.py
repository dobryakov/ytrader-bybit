"""
Test fixtures for HTTP client mocking (ws-gateway API).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, Response, Request
from typing import Optional


@pytest.fixture
def mock_http_client():
    """Mock httpx AsyncClient for ws-gateway API."""
    client = AsyncMock(spec=AsyncClient)
    
    # Default successful response
    default_response = Response(
        status_code=200,
        json={"status": "ok"},
        request=Request("GET", "http://localhost")
    )
    
    client.get = AsyncMock(return_value=default_response)
    client.post = AsyncMock(return_value=default_response)
    client.put = AsyncMock(return_value=default_response)
    client.delete = AsyncMock(return_value=default_response)
    client.request = AsyncMock(return_value=default_response)
    client.aclose = AsyncMock()
    
    return client


@pytest.fixture
def mock_ws_gateway_response():
    """Mock ws-gateway API response."""
    def _create_response(
        status_code: int = 200,
        json_data: Optional[dict] = None,
        text: Optional[str] = None
    ):
        response = Response(
            status_code=status_code,
            json=json_data or {"status": "ok"},
            text=text,
            request=Request("GET", "http://localhost")
        )
        return response
    
    return _create_response


@pytest.fixture
def mock_ws_gateway_subscription_response():
    """Mock ws-gateway subscription API response."""
    return {
        "subscription_id": "test-subscription-123",
        "channels": ["orderbook", "trades"],
        "symbols": ["BTCUSDT"],
        "status": "active"
    }


@pytest.fixture
def mock_ws_gateway_error_response():
    """Mock ws-gateway error response."""
    def _create_error(status_code: int = 500, message: str = "Internal server error"):
        response = Response(
            status_code=status_code,
            json={"error": message},
            request=Request("GET", "http://localhost")
        )
        return response
    
    return _create_error

