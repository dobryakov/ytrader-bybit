"""Integration tests for Position Manager REST API client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime

from src.services.position_manager_client import PositionManagerClient, PortfolioExposure
from src.models.position import Position
from src.exceptions import OrderExecutionError


@pytest.mark.asyncio
async def test_get_position_success():
    """Test successful position retrieval from Position Manager."""
    client = PositionManagerClient()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "size": "1.5",
        "average_entry_price": "50000.0",
        "unrealized_pnl": "500.0",
        "realized_pnl": "0.0",
        "mode": "one-way",
        "long_size": None,
        "short_size": None,
        "long_avg_price": None,
        "short_avg_price": None,
        "last_updated": "2025-01-27T10:00:00Z",
        "last_snapshot_at": None,
    }
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        position = await client.get_position("BTCUSDT", mode="one-way", trace_id="test-trace")
    
    assert position is not None
    assert position.asset == "BTCUSDT"
    assert position.size == Decimal("1.5")
    assert position.average_entry_price == Decimal("50000.0")
    assert position.mode == "one-way"
    await client.close()


@pytest.mark.asyncio
async def test_get_position_not_found():
    """Test position retrieval when position does not exist."""
    client = PositionManagerClient()
    
    # Mock 404 response
    mock_response = MagicMock()
    mock_response.status_code = 404
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        position = await client.get_position("BTCUSDT", mode="one-way", trace_id="test-trace")
    
    assert position is None
    await client.close()


@pytest.mark.asyncio
async def test_get_position_api_error():
    """Test position retrieval with API error."""
    client = PositionManagerClient()
    
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        with pytest.raises(OrderExecutionError, match="Failed to get position"):
            await client.get_position("BTCUSDT", mode="one-way", trace_id="test-trace")
    
    await client.close()


@pytest.mark.asyncio
async def test_get_all_positions_success():
    """Test successful retrieval of all positions."""
    client = PositionManagerClient()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "positions": [
            {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "asset": "BTCUSDT",
                "size": "1.5",
                "average_entry_price": "50000.0",
                "unrealized_pnl": "500.0",
                "realized_pnl": "0.0",
                "mode": "one-way",
                "long_size": None,
                "short_size": None,
                "long_avg_price": None,
                "short_avg_price": None,
                "last_updated": "2025-01-27T10:00:00Z",
                "last_snapshot_at": None,
            },
            {
                "id": "223e4567-e89b-12d3-a456-426614174001",
                "asset": "ETHUSDT",
                "size": "10.0",
                "average_entry_price": "3000.0",
                "unrealized_pnl": "100.0",
                "realized_pnl": "0.0",
                "mode": "one-way",
                "long_size": None,
                "short_size": None,
                "long_avg_price": None,
                "short_avg_price": None,
                "last_updated": "2025-01-27T10:00:00Z",
                "last_snapshot_at": None,
            },
        ],
        "count": 2,
    }
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        positions = await client.get_all_positions(asset=None, mode=None, trace_id="test-trace")
    
    assert len(positions) == 2
    assert positions[0].asset == "BTCUSDT"
    assert positions[1].asset == "ETHUSDT"
    await client.close()


@pytest.mark.asyncio
async def test_get_portfolio_exposure_success():
    """Test successful portfolio exposure retrieval."""
    client = PositionManagerClient()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "total_exposure_usdt": "50000.0",
        "calculated_at": "2025-01-27T10:00:00Z",
    }
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        exposure = await client.get_portfolio_exposure(trace_id="test-trace")
    
    assert exposure is not None
    assert exposure.total_exposure_usdt == Decimal("50000.0")
    assert isinstance(exposure.calculated_at, datetime)
    await client.close()


@pytest.mark.asyncio
async def test_get_portfolio_exposure_unavailable():
    """Test portfolio exposure retrieval when service is unavailable."""
    client = PositionManagerClient()
    
    # Mock 503 response (service unavailable)
    mock_response = MagicMock()
    mock_response.status_code = 503
    
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        with pytest.raises(OrderExecutionError, match="temporarily unavailable"):
            await client.get_portfolio_exposure(trace_id="test-trace")
    
    await client.close()

