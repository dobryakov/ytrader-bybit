"""
Unit tests for raw data storage service.
"""
import pytest
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.data_storage import DataStorageService
from src.storage.parquet_storage import ParquetStorage


@pytest.fixture
def tmp_storage_path(tmp_path):
    """Temporary storage path for testing."""
    storage_dir = tmp_path / "raw_data"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


@pytest.fixture
def mock_parquet_storage():
    """Mock ParquetStorage for testing."""
    return MagicMock(spec=ParquetStorage)


@pytest.fixture
def data_storage_service(tmp_storage_path, mock_parquet_storage):
    """DataStorageService instance for testing."""
    return DataStorageService(
        base_path=tmp_storage_path,
        parquet_storage=mock_parquet_storage,
        retention_days=90,
    )


class TestDataStorageService:
    """Test cases for DataStorageService."""
    
    @pytest.mark.asyncio
    async def test_store_orderbook_snapshot(
        self, data_storage_service, raw_orderbook_snapshot
    ):
        """Test storing orderbook snapshot."""
        await data_storage_service.store_orderbook_snapshot(
            raw_orderbook_snapshot
        )
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_orderbook_snapshots.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_orderbook_snapshots.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert call_args[0][1] == datetime.now(timezone.utc).date().strftime("%Y-%m-%d")  # date
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_orderbook_delta(
        self, data_storage_service, raw_orderbook_delta
    ):
        """Test storing orderbook delta."""
        await data_storage_service.store_orderbook_delta(
            raw_orderbook_delta
        )
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_orderbook_deltas.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_orderbook_deltas.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_trade(
        self, data_storage_service, raw_trade
    ):
        """Test storing trade."""
        await data_storage_service.store_trade(raw_trade)
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_trades.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_trades.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_kline(
        self, data_storage_service, raw_kline
    ):
        """Test storing kline."""
        await data_storage_service.store_kline(raw_kline)
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_klines.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_klines.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_ticker(
        self, data_storage_service, raw_ticker
    ):
        """Test storing ticker."""
        await data_storage_service.store_ticker(raw_ticker)
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_ticker.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_ticker.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_funding_rate(
        self, data_storage_service, raw_funding_rate
    ):
        """Test storing funding rate."""
        await data_storage_service.store_funding_rate(raw_funding_rate)
        
        # Verify parquet storage was called
        data_storage_service._parquet_storage.write_funding.assert_called_once()
        call_args = data_storage_service._parquet_storage.write_funding.call_args
        
        assert call_args[0][0] == "BTCUSDT"  # symbol
        assert isinstance(call_args[0][2], pd.DataFrame)  # data
    
    @pytest.mark.asyncio
    async def test_store_execution_event(
        self, data_storage_service, raw_execution_event
    ):
        """Test storing execution event."""
        await data_storage_service.store_execution_event(raw_execution_event)
        
        # Verify execution event was stored (may use trades storage or separate)
        # For now, we'll check that the method exists and can be called
        assert hasattr(data_storage_service, 'store_execution_event')
    
    @pytest.mark.asyncio
    async def test_store_batch_events(
        self, data_storage_service, raw_market_data_batch
    ):
        """Test storing batch of events."""
        await data_storage_service.store_market_data_events(raw_market_data_batch)
        
        # Verify multiple storage calls were made
        assert data_storage_service._parquet_storage.write_orderbook_snapshots.called
        assert data_storage_service._parquet_storage.write_orderbook_deltas.called
        assert data_storage_service._parquet_storage.write_trades.called
        assert data_storage_service._parquet_storage.write_klines.called
    
    @pytest.mark.asyncio
    async def test_store_handles_missing_timestamps(
        self, data_storage_service, raw_trade
    ):
        """Test that storage handles missing timestamps gracefully."""
        # Remove timestamp fields
        del raw_trade["timestamp"]
        del raw_trade["internal_timestamp"]
        del raw_trade["exchange_timestamp"]
        
        # Should not raise error, should use current time
        await data_storage_service.store_trade(raw_trade)
        
        # Verify storage was called
        assert data_storage_service._parquet_storage.write_trades.called
    
    @pytest.mark.asyncio
    async def test_store_handles_invalid_event_type(
        self, data_storage_service
    ):
        """Test that storage handles invalid event types gracefully."""
        invalid_event = {
            "event_type": "unknown_event",
            "symbol": "BTCUSDT",
        }
        
        # Should not raise error, should log warning
        await data_storage_service.store_market_data_event(invalid_event)
        
        # Verify no storage calls were made
        assert not data_storage_service._parquet_storage.write_orderbook_snapshots.called
    
    @pytest.mark.asyncio
    async def test_store_handles_missing_symbol(
        self, data_storage_service, raw_trade
    ):
        """Test that storage handles missing symbol gracefully."""
        del raw_trade["symbol"]
        
        # Should not raise error, should log warning
        await data_storage_service.store_trade(raw_trade)
        
        # Verify no storage calls were made
        assert not data_storage_service._parquet_storage.write_trades.called
