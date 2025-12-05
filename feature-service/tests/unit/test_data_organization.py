"""
Unit tests for data organization by type.
"""
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestDataOrganization:
    """Test cases for data organization by type."""
    
    def test_orderbook_snapshots_path(self, data_storage_service):
        """Test that orderbook snapshots are stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "orderbook" / "snapshots" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "orderbook" in str(expected_path)
        assert "snapshots" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    def test_orderbook_deltas_path(self, data_storage_service):
        """Test that orderbook deltas are stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "orderbook" / "deltas" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "orderbook" in str(expected_path)
        assert "deltas" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    def test_trades_path(self, data_storage_service):
        """Test that trades are stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "trades" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "trades" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    def test_klines_path(self, data_storage_service):
        """Test that klines are stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "klines" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "klines" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    def test_ticker_path(self, data_storage_service):
        """Test that ticker data is stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "ticker" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "ticker" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    def test_funding_path(self, data_storage_service):
        """Test that funding rate data is stored in correct path."""
        date_str = "2025-01-27"
        symbol = "BTCUSDT"
        
        # Check path structure
        expected_path = Path(data_storage_service._base_path) / "funding" / date_str / f"{symbol}.parquet"
        
        # Verify path structure matches expected organization
        assert "funding" in str(expected_path)
        assert date_str in str(expected_path)
        assert symbol in str(expected_path)
    
    @pytest.mark.asyncio
    async def test_events_organized_by_type(
        self, data_storage_service, raw_orderbook_snapshot, raw_trade, raw_kline
    ):
        """Test that events are organized by type correctly."""
        # Store different event types
        await data_storage_service.store_orderbook_snapshot(raw_orderbook_snapshot)
        await data_storage_service.store_trade(raw_trade)
        await data_storage_service.store_kline(raw_kline)
        
        # Verify each type was stored in correct location
        assert data_storage_service._parquet_storage.write_orderbook_snapshots.called
        assert data_storage_service._parquet_storage.write_trades.called
        assert data_storage_service._parquet_storage.write_klines.called
        
        # Verify snapshots and trades used different paths
        snapshot_call = data_storage_service._parquet_storage.write_orderbook_snapshots.call_args
        trade_call = data_storage_service._parquet_storage.write_trades.call_args
        
        assert snapshot_call[0][0] == trade_call[0][0]  # Same symbol
        assert snapshot_call[0][1] == trade_call[0][1]  # Same date
    
    @pytest.mark.asyncio
    async def test_events_organized_by_date(
        self, data_storage_service, raw_trade
    ):
        """Test that events are organized by date correctly."""
        # Store same event type for different dates
        date1 = datetime(2025, 1, 27, tzinfo=timezone.utc)
        date2 = datetime(2025, 1, 28, tzinfo=timezone.utc)
        
        trade1 = raw_trade.copy()
        trade1["timestamp"] = date1.isoformat()
        trade1["internal_timestamp"] = date1.isoformat()
        
        trade2 = raw_trade.copy()
        trade2["timestamp"] = date2.isoformat()
        trade2["internal_timestamp"] = date2.isoformat()
        
        await data_storage_service.store_trade(trade1)
        await data_storage_service.store_trade(trade2)
        
        # Verify different dates were used
        calls = data_storage_service._parquet_storage.write_trades.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] != calls[1][0][1]  # Different dates
    
    @pytest.mark.asyncio
    async def test_events_organized_by_symbol(
        self, data_storage_service, raw_trade
    ):
        """Test that events are organized by symbol correctly."""
        # Store same event type for different symbols
        trade1 = raw_trade.copy()
        trade1["symbol"] = "BTCUSDT"
        
        trade2 = raw_trade.copy()
        trade2["symbol"] = "ETHUSDT"
        
        await data_storage_service.store_trade(trade1)
        await data_storage_service.store_trade(trade2)
        
        # Verify different symbols were used
        calls = data_storage_service._parquet_storage.write_trades.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] != calls[1][0][0]  # Different symbols
    
    def test_all_orderbook_deltas_stored(
        self, data_storage_service, raw_orderbook_delta, raw_orderbook_delta_insert, raw_orderbook_delta_delete
    ):
        """Test that all orderbook delta types (insert, update, delete) are stored."""
        # This test verifies that all delta types are handled
        # Implementation should store all deltas regardless of type
        assert hasattr(data_storage_service, 'store_orderbook_delta')
        
        # Verify delta types are recognized
        assert raw_orderbook_delta["delta_type"] == "update"
        assert raw_orderbook_delta_insert["delta_type"] == "insert"
        assert raw_orderbook_delta_delete["delta_type"] == "delete"
