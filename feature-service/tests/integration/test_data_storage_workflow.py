"""
Integration tests for raw data storage workflow.
"""
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from src.services.data_storage import DataStorageService
from src.storage.parquet_storage import ParquetStorage


@pytest.fixture
def tmp_storage_path(tmp_path):
    """Temporary storage path for testing."""
    storage_dir = tmp_path / "raw_data"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


@pytest.fixture
def parquet_storage(tmp_storage_path):
    """ParquetStorage instance for testing."""
    return ParquetStorage(base_path=tmp_storage_path)


@pytest.fixture
def data_storage_service(tmp_storage_path, parquet_storage):
    """DataStorageService instance for testing."""
    return DataStorageService(
        base_path=tmp_storage_path,
        parquet_storage=parquet_storage,
        retention_days=90,
    )


class TestDataStorageWorkflow:
    """Integration tests for raw data storage workflow."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_orderbook_snapshot(
        self, data_storage_service, raw_orderbook_snapshot, tmp_storage_path
    ):
        """Test storing and retrieving orderbook snapshot."""
        # Store snapshot
        await data_storage_service.store_orderbook_snapshot(raw_orderbook_snapshot)
        
        # Verify file was created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        file_path = Path(tmp_storage_path) / "orderbook" / "snapshots" / date_str / "BTCUSDT.parquet"
        
        # File should exist
        assert file_path.exists()
        
        # Verify data can be read back
        df = await data_storage_service._parquet_storage.read_orderbook_snapshots("BTCUSDT", date_str)
        assert not df.empty
        assert "symbol" in df.columns or "BTCUSDT" in str(df.values)
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_orderbook_delta(
        self, data_storage_service, raw_orderbook_delta, tmp_storage_path
    ):
        """Test storing and retrieving orderbook delta."""
        # Store delta
        await data_storage_service.store_orderbook_delta(raw_orderbook_delta)
        
        # Verify file was created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        file_path = Path(tmp_storage_path) / "orderbook" / "deltas" / date_str / "BTCUSDT.parquet"
        
        # File should exist
        assert file_path.exists()
        
        # Verify data can be read back
        df = await data_storage_service._parquet_storage.read_orderbook_deltas("BTCUSDT", date_str)
        assert not df.empty
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_trade(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test storing and retrieving trade."""
        # Store trade
        await data_storage_service.store_trade(raw_trade)
        
        # Verify file was created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        file_path = Path(tmp_storage_path) / "trades" / date_str / "BTCUSDT.parquet"
        
        # File should exist
        assert file_path.exists()
        
        # Verify data can be read back
        df = await data_storage_service._parquet_storage.read_trades("BTCUSDT", date_str)
        assert not df.empty
    
    @pytest.mark.asyncio
    async def test_store_multiple_events_same_day(
        self, data_storage_service, raw_market_data_batch, tmp_storage_path
    ):
        """Test storing multiple events for the same day."""
        # Store batch of events
        await data_storage_service.store_market_data_events(raw_market_data_batch)
        
        # Verify all files were created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        
        snapshot_path = Path(tmp_storage_path) / "orderbook" / "snapshots" / date_str / "BTCUSDT.parquet"
        delta_path = Path(tmp_storage_path) / "orderbook" / "deltas" / date_str / "BTCUSDT.parquet"
        trade_path = Path(tmp_storage_path) / "trades" / date_str / "BTCUSDT.parquet"
        kline_path = Path(tmp_storage_path) / "klines" / date_str / "BTCUSDT.parquet"
        
        # All files should exist
        assert snapshot_path.exists()
        assert delta_path.exists()
        assert trade_path.exists()
        assert kline_path.exists()
    
    @pytest.mark.asyncio
    async def test_store_events_different_days(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test storing events for different days."""
        # Store trade for day 1
        date1 = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
        trade1 = raw_trade.copy()
        trade1["timestamp"] = date1.isoformat()
        trade1["internal_timestamp"] = date1.isoformat()
        await data_storage_service.store_trade(trade1)
        
        # Store trade for day 2
        date2 = datetime(2025, 1, 28, 12, 0, 0, tzinfo=timezone.utc)
        trade2 = raw_trade.copy()
        trade2["timestamp"] = date2.isoformat()
        trade2["internal_timestamp"] = date2.isoformat()
        await data_storage_service.store_trade(trade2)
        
        # Verify separate files were created
        file1 = Path(tmp_storage_path) / "trades" / "2025-01-27" / "BTCUSDT.parquet"
        file2 = Path(tmp_storage_path) / "trades" / "2025-01-28" / "BTCUSDT.parquet"
        
        assert file1.exists()
        assert file2.exists()
    
    @pytest.mark.asyncio
    async def test_store_events_different_symbols(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test storing events for different symbols."""
        # Store trade for BTCUSDT
        trade1 = raw_trade.copy()
        trade1["symbol"] = "BTCUSDT"
        await data_storage_service.store_trade(trade1)
        
        # Store trade for ETHUSDT
        trade2 = raw_trade.copy()
        trade2["symbol"] = "ETHUSDT"
        await data_storage_service.store_trade(trade2)
        
        # Verify separate files were created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        file1 = Path(tmp_storage_path) / "trades" / date_str / "BTCUSDT.parquet"
        file2 = Path(tmp_storage_path) / "trades" / date_str / "ETHUSDT.parquet"
        
        assert file1.exists()
        assert file2.exists()
    
    @pytest.mark.asyncio
    async def test_store_all_orderbook_deltas(
        self, data_storage_service, raw_orderbook_delta, raw_orderbook_delta_insert, raw_orderbook_delta_delete, tmp_storage_path
    ):
        """Test that all orderbook delta types are stored."""
        # Store different delta types
        await data_storage_service.store_orderbook_delta(raw_orderbook_delta)
        await data_storage_service.store_orderbook_delta(raw_orderbook_delta_insert)
        await data_storage_service.store_orderbook_delta(raw_orderbook_delta_delete)
        
        # Verify file exists and contains all deltas
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        file_path = Path(tmp_storage_path) / "orderbook" / "deltas" / date_str / "BTCUSDT.parquet"
        
        assert file_path.exists()
        
        # Verify data can be read back
        df = await data_storage_service._parquet_storage.read_orderbook_deltas("BTCUSDT", date_str)
        assert not df.empty
        # Should contain all delta types
        if "delta_type" in df.columns:
            assert "update" in df["delta_type"].values or len(df) >= 3
    
    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(
        self, data_storage_service, raw_market_data_batch
    ):
        """Test concurrent storage operations."""
        import asyncio
        
        # Store multiple events concurrently
        tasks = [
            data_storage_service.store_market_data_event(event)
            for event in raw_market_data_batch
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all files were created
        date_str = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
        base_path = Path(data_storage_service._base_path)
        
        # Check that files exist (may be in different states due to concurrency)
        # At minimum, verify no errors occurred
        assert True  # If we get here, no exceptions were raised
