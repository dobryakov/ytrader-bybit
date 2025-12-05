"""
Integration tests for data retrieval for dataset rebuilding.
"""
import pytest
from datetime import datetime, timezone, timedelta, date
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


class TestDataRetrieval:
    """Integration tests for data retrieval for dataset rebuilding."""
    
    @pytest.mark.asyncio
    async def test_retrieve_trades_for_date_range(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test retrieving trades for a date range."""
        # Store trades for multiple days
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 29)
        
        for i in range(3):
            current_date = start_date + timedelta(days=i)
            trade = raw_trade.copy()
            trade["timestamp"] = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
            trade["internal_timestamp"] = trade["timestamp"]
            await data_storage_service.store_trade(trade)
        
        # Retrieve trades for date range
        df = await data_storage_service._parquet_storage.read_trades_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify data was retrieved
        assert not df.empty
        assert len(df) >= 3  # At least 3 trades
    
    @pytest.mark.asyncio
    async def test_retrieve_orderbook_snapshots_for_date_range(
        self, data_storage_service, raw_orderbook_snapshot, tmp_storage_path
    ):
        """Test retrieving orderbook snapshots for a date range."""
        # Store snapshots for multiple days
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 29)
        
        for i in range(3):
            current_date = start_date + timedelta(days=i)
            snapshot = raw_orderbook_snapshot.copy()
            snapshot["timestamp"] = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
            snapshot["internal_timestamp"] = snapshot["timestamp"]
            await data_storage_service.store_orderbook_snapshot(snapshot)
        
        # Retrieve snapshots for date range
        df = await data_storage_service._parquet_storage.read_orderbook_snapshots_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify data was retrieved
        assert not df.empty
        assert len(df) >= 3  # At least 3 snapshots
    
    @pytest.mark.asyncio
    async def test_retrieve_orderbook_deltas_for_date_range(
        self, data_storage_service, raw_orderbook_deltas_sequence, tmp_storage_path
    ):
        """Test retrieving orderbook deltas for a date range."""
        # Store deltas for multiple days
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 29)
        
        # Store deltas across multiple days
        for i, delta in enumerate(raw_orderbook_deltas_sequence):
            current_date = start_date + timedelta(days=i % 3)
            delta["timestamp"] = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
            delta["internal_timestamp"] = delta["timestamp"]
            await data_storage_service.store_orderbook_delta(delta)
        
        # Retrieve deltas for date range
        df = await data_storage_service._parquet_storage.read_orderbook_deltas_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify data was retrieved
        assert not df.empty
        assert len(df) >= len(raw_orderbook_deltas_sequence)  # At least all deltas
    
    @pytest.mark.asyncio
    async def test_retrieve_klines_for_date_range(
        self, data_storage_service, raw_kline, tmp_storage_path
    ):
        """Test retrieving klines for a date range."""
        # Store klines for multiple days
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 29)
        
        for i in range(3):
            current_date = start_date + timedelta(days=i)
            kline = raw_kline.copy()
            kline["timestamp"] = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
            kline["internal_timestamp"] = kline["timestamp"]
            await data_storage_service.store_kline(kline)
        
        # Retrieve klines for date range
        df = await data_storage_service._parquet_storage.read_klines_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify data was retrieved
        assert not df.empty
        assert len(df) >= 3  # At least 3 klines
    
    @pytest.mark.asyncio
    async def test_retrieve_data_for_offline_reconstruction(
        self, data_storage_service, raw_orderbook_snapshot, raw_orderbook_deltas_sequence, tmp_storage_path
    ):
        """Test retrieving data for offline orderbook reconstruction."""
        # Store snapshot
        snapshot_date = date(2025, 1, 27)
        snapshot = raw_orderbook_snapshot.copy()
        snapshot["timestamp"] = datetime.combine(snapshot_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
        snapshot["internal_timestamp"] = snapshot["timestamp"]
        await data_storage_service.store_orderbook_snapshot(snapshot)
        
        # Store deltas after snapshot
        for i, delta in enumerate(raw_orderbook_deltas_sequence[:5]):
            delta["timestamp"] = (datetime.combine(snapshot_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(seconds=i)).isoformat()
            delta["internal_timestamp"] = delta["timestamp"]
            delta["sequence"] = snapshot["sequence"] + i + 1
            await data_storage_service.store_orderbook_delta(delta)
        
        # Retrieve snapshot
        snapshot_df = await data_storage_service._parquet_storage.read_orderbook_snapshots(
            "BTCUSDT", snapshot_date.strftime("%Y-%m-%d")
        )
        
        # Retrieve deltas
        deltas_df = await data_storage_service._parquet_storage.read_orderbook_deltas_range(
            "BTCUSDT", snapshot_date, snapshot_date
        )
        
        # Verify both snapshot and deltas are available
        assert not snapshot_df.empty
        assert not deltas_df.empty
        assert len(deltas_df) >= 5  # At least 5 deltas
    
    @pytest.mark.asyncio
    async def test_retrieve_data_with_missing_days(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test retrieving data when some days are missing."""
        # Store trades for day 1 and day 3 (skip day 2)
        trade1 = raw_trade.copy()
        trade1["timestamp"] = datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        trade1["internal_timestamp"] = trade1["timestamp"]
        await data_storage_service.store_trade(trade1)
        
        trade3 = raw_trade.copy()
        trade3["timestamp"] = datetime(2025, 1, 29, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        trade3["internal_timestamp"] = trade3["timestamp"]
        await data_storage_service.store_trade(trade3)
        
        # Retrieve for date range including missing day
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 29)
        
        df = await data_storage_service._parquet_storage.read_trades_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Should return data from available days only
        assert not df.empty
        assert len(df) >= 2  # At least 2 trades
    
    @pytest.mark.asyncio
    async def test_retrieve_data_sorted_by_timestamp(
        self, data_storage_service, raw_trade, tmp_storage_path
    ):
        """Test that retrieved data is sorted by timestamp."""
        # Store trades in reverse order
        timestamps = [
            datetime(2025, 1, 27, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 27, 18, 0, 0, tzinfo=timezone.utc),
        ]
        
        for ts in timestamps:
            trade = raw_trade.copy()
            trade["timestamp"] = ts.isoformat()
            trade["internal_timestamp"] = ts.isoformat()
            await data_storage_service.store_trade(trade)
        
        # Retrieve trades
        start_date = date(2025, 1, 27)
        end_date = date(2025, 1, 27)
        
        df = await data_storage_service._parquet_storage.read_trades_range(
            "BTCUSDT", start_date, end_date
        )
        
        # Verify data is sorted by timestamp
        assert not df.empty
        if "timestamp" in df.columns:
            timestamps_sorted = df["timestamp"].tolist()
            assert timestamps_sorted == sorted(timestamps_sorted)
