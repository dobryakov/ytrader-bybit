"""Tests for instrument_info_client service."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.instrument_info_client import InstrumentInfoClient, instrument_info_client


@pytest.mark.asyncio
async def test_get_min_order_value_from_db():
    """Test getting min_order_value from database."""
    client = InstrumentInfoClient()
    
    # Mock database pool and fetchrow
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda self, key: Decimal("10.5") if key == "min_order_value" else None
    
    with patch("src.services.instrument_info_client.db_pool") as mock_db_pool:
        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=mock_row)
        mock_db_pool.get_pool = AsyncMock(return_value=mock_pool)
        
        result = await client.get_min_order_value("ETHUSDT")
        
        assert result == Decimal("10.5")
        mock_pool.fetchrow.assert_called_once()
        assert "ETHUSDT" in client._cache
        assert client._cache["ETHUSDT"] == Decimal("10.5")


@pytest.mark.asyncio
async def test_get_min_order_value_not_found():
    """Test getting min_order_value when symbol not in database."""
    client = InstrumentInfoClient()
    
    with patch("src.services.instrument_info_client.db_pool") as mock_db_pool:
        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=None)
        mock_db_pool.get_pool = AsyncMock(return_value=mock_pool)
        
        result = await client.get_min_order_value("UNKNOWNUSDT")
        
        assert result is None
        assert "UNKNOWNUSDT" in client._cache
        assert client._cache["UNKNOWNUSDT"] is None


@pytest.mark.asyncio
async def test_get_min_order_value_cached():
    """Test that cached values are returned without database query."""
    client = InstrumentInfoClient()
    
    # Pre-populate cache
    client._cache["ETHUSDT"] = Decimal("15.0")
    
    with patch("src.services.instrument_info_client.db_pool") as mock_db_pool:
        result = await client.get_min_order_value("ETHUSDT")
        
        assert result == Decimal("15.0")
        # Should not call database
        mock_db_pool.get_pool.assert_not_called()


@pytest.mark.asyncio
async def test_get_min_order_value_db_error():
    """Test handling of database errors."""
    client = InstrumentInfoClient()
    
    with patch("src.services.instrument_info_client.db_pool") as mock_db_pool:
        mock_db_pool.get_pool = AsyncMock(side_effect=Exception("Database error"))
        
        result = await client.get_min_order_value("ETHUSDT")
        
        assert result is None
        # Should cache None to avoid repeated failed queries
        assert client._cache["ETHUSDT"] is None


@pytest.mark.asyncio
async def test_get_min_order_value_null_in_db():
    """Test handling when min_order_value is NULL in database."""
    client = InstrumentInfoClient()
    
    mock_row = MagicMock()
    mock_row.__getitem__ = lambda self, key: None if key == "min_order_value" else None
    
    with patch("src.services.instrument_info_client.db_pool") as mock_db_pool:
        mock_pool = AsyncMock()
        mock_pool.fetchrow = AsyncMock(return_value=mock_row)
        mock_db_pool.get_pool = AsyncMock(return_value=mock_pool)
        
        result = await client.get_min_order_value("ETHUSDT")
        
        assert result is None
        assert client._cache["ETHUSDT"] is None


@pytest.mark.asyncio
async def test_clear_cache_specific_symbol():
    """Test clearing cache for a specific symbol."""
    client = InstrumentInfoClient()
    
    client._cache["ETHUSDT"] = Decimal("10.0")
    client._cache["BTCUSDT"] = Decimal("20.0")
    
    await client.clear_cache("ETHUSDT")
    
    assert "ETHUSDT" not in client._cache
    assert "BTCUSDT" in client._cache
    assert client._cache["BTCUSDT"] == Decimal("20.0")


@pytest.mark.asyncio
async def test_clear_cache_all():
    """Test clearing all cache."""
    client = InstrumentInfoClient()
    
    client._cache["ETHUSDT"] = Decimal("10.0")
    client._cache["BTCUSDT"] = Decimal("20.0")
    
    await client.clear_cache()
    
    assert len(client._cache) == 0


@pytest.mark.asyncio
async def test_global_instance():
    """Test that global instance exists and works."""
    assert instrument_info_client is not None
    assert isinstance(instrument_info_client, InstrumentInfoClient)

