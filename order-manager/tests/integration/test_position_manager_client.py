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


@pytest.mark.asyncio
async def test_get_position_from_bybit_success():
    """Test successful position retrieval directly from Bybit API."""
    client = PositionManagerClient()
    
    # Mock Bybit client response
    mock_bybit_response = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "size": "1.5",
                    "side": "Buy",
                    "avgPrice": "50000.0",
                    "unrealisedPnl": "500.0",
                }
            ]
        }
    }
    
    mock_bybit_client = AsyncMock()
    mock_bybit_client.get = AsyncMock(return_value=mock_bybit_response)
    
    with patch("src.utils.bybit_client.get_bybit_client", return_value=mock_bybit_client):
        position = await client.get_position_from_bybit("BTCUSDT", trace_id="test-trace")
    
    assert position is not None
    assert position.asset == "BTCUSDT"
    assert position.size == Decimal("1.5")
    assert position.average_entry_price == Decimal("50000.0")
    assert position.unrealized_pnl == Decimal("500.0")
    assert position.mode == "one-way"
    await client.close()


@pytest.mark.asyncio
async def test_get_position_from_bybit_not_found():
    """Test position retrieval from Bybit when position does not exist."""
    client = PositionManagerClient()
    
    # Mock Bybit client response - position not found
    mock_bybit_response = {
        "retCode": 10001,
        "retMsg": "Position not found",
        "result": {
            "list": []
        }
    }
    
    mock_bybit_client = AsyncMock()
    mock_bybit_client.get = AsyncMock(return_value=mock_bybit_response)
    
    with patch("src.utils.bybit_client.get_bybit_client", return_value=mock_bybit_client):
        position = await client.get_position_from_bybit("BTCUSDT", trace_id="test-trace")
    
    assert position is None
    await client.close()


@pytest.mark.asyncio
async def test_get_position_from_bybit_zero_size():
    """Test position retrieval from Bybit when position size is zero."""
    client = PositionManagerClient()
    
    # Mock Bybit client response - zero size
    mock_bybit_response = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "size": "0",
                    "side": "Buy",
                    "avgPrice": "0",
                    "unrealisedPnl": "0",
                }
            ]
        }
    }
    
    mock_bybit_client = AsyncMock()
    mock_bybit_client.get = AsyncMock(return_value=mock_bybit_response)
    
    with patch("src.utils.bybit_client.get_bybit_client", return_value=mock_bybit_client):
        position = await client.get_position_from_bybit("BTCUSDT", trace_id="test-trace")
    
    assert position is None
    await client.close()


@pytest.mark.asyncio
async def test_trigger_bybit_sync_async_success():
    """Test successful async trigger of Bybit sync (fire-and-forget)."""
    import asyncio
    client = PositionManagerClient()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "bybit_positions_count": 1,
        "local_positions_count": 1,
        "updated": [{"asset": "BTCUSDT", "action": "updated"}],
        "created": [],
        "errors": [],
    }
    
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    
    with patch.object(client, "_get_client", return_value=mock_client):
        # Call async method (fire-and-forget)
        await client.trigger_bybit_sync_async(asset="BTCUSDT", force=True, trace_id="test-trace")
        
        # Give task time to execute
        await asyncio.sleep(0.1)
    
    # Verify POST was called
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "/api/v1/positions/sync-bybit"
    assert "force" in call_args[1]["params"]
    assert call_args[1]["params"]["asset"] == "BTCUSDT"
    
    await client.close()


@pytest.mark.asyncio
async def test_trigger_bybit_sync_async_timeout():
    """Test async trigger of Bybit sync with timeout (should not raise)."""
    import asyncio
    import httpx
    client = PositionManagerClient()
    
    # Mock timeout exception
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    
    with patch.object(client, "_get_client", return_value=mock_client):
        # Call async method (fire-and-forget) - should not raise
        await client.trigger_bybit_sync_async(asset="BTCUSDT", force=True, trace_id="test-trace")
        
        # Give task time to execute
        await asyncio.sleep(0.1)
    
    # Should complete without raising
    await client.close()

