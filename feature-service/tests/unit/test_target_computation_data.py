"""
Unit tests for target computation data loading functions.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.target_computation_data import (
    find_available_data_range,
    load_historical_data_for_target_computation,
)
from src.storage.parquet_storage import ParquetStorage


@pytest.fixture
def mock_parquet_storage():
    """Create a mock ParquetStorage instance."""
    storage = MagicMock(spec=ParquetStorage)
    storage.read_klines = AsyncMock()
    storage.read_klines_range = AsyncMock()
    return storage


@pytest.fixture
def sample_klines_data():
    """Create sample klines data for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    timestamps = [base_time + timedelta(seconds=i * 60) for i in range(100)]
    prices = [50000.0 + i * 0.1 for i in range(100)]
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [100.0] * 100,
    })


@pytest.mark.asyncio
async def test_find_available_data_range_data_available(mock_parquet_storage, sample_klines_data):
    """Test find_available_data_range when data is available."""
    target_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    target_date = target_timestamp.date()
    target_date_str = target_date.strftime("%Y-%m-%d")
    previous_date_str = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Mock read_klines to return data for target_date, None for previous_date
    def read_klines_side_effect(symbol, date_str):
        if date_str == target_date_str:
            return sample_klines_data
        elif date_str == previous_date_str:
            raise FileNotFoundError("No data")
        else:
            raise FileNotFoundError("No data")
    
    mock_parquet_storage.read_klines.side_effect = read_klines_side_effect
    
    result = await find_available_data_range(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        target_timestamp=target_timestamp,
        max_lookback_seconds=300,
    )
    
    assert result is not None
    assert result["timestamp_adjusted"] is False
    assert result["lookback_seconds_used"] == 0
    assert not result["historical_data"].empty
    # Function checks both target_date and previous_date, so should be called twice
    assert mock_parquet_storage.read_klines.call_count == 2


@pytest.mark.asyncio
async def test_find_available_data_range_data_delayed_within_lookback(mock_parquet_storage, sample_klines_data):
    """Test find_available_data_range when data is delayed but within lookback window."""
    # Target timestamp is in the future (data not yet available)
    target_timestamp = datetime.now(timezone.utc) + timedelta(minutes=1)
    target_date = target_timestamp.date()
    date_str = target_date.strftime("%Y-%m-%d")
    
    # But latest data is only 2 minutes ago (within 5 minute lookback)
    latest_timestamp = datetime.now(timezone.utc) - timedelta(minutes=2)
    delayed_data = sample_klines_data.copy()
    delayed_data["timestamp"] = pd.to_datetime([latest_timestamp] * len(delayed_data))
    
    mock_parquet_storage.read_klines.return_value = delayed_data
    
    result = await find_available_data_range(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        target_timestamp=target_timestamp,
        max_lookback_seconds=300,  # 5 minutes
    )
    
    assert result is not None
    assert result["timestamp_adjusted"] is True
    assert result["lookback_seconds_used"] > 0
    assert result["lookback_seconds_used"] <= 300
    assert not result["historical_data"].empty


@pytest.mark.asyncio
async def test_find_available_data_range_data_too_old(mock_parquet_storage, sample_klines_data):
    """Test find_available_data_range when data is too old (beyond lookback)."""
    # Target timestamp is in the future
    target_timestamp = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    # Latest data is 10 minutes ago (beyond 5 minute lookback)
    latest_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
    old_data = sample_klines_data.copy()
    old_data["timestamp"] = pd.to_datetime([latest_timestamp] * len(old_data))
    
    mock_parquet_storage.read_klines.return_value = old_data
    
    result = await find_available_data_range(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        target_timestamp=target_timestamp,
        max_lookback_seconds=300,  # 5 minutes
    )
    
    # Should return None because gap is too large
    assert result is None


@pytest.mark.asyncio
async def test_find_available_data_range_no_data(mock_parquet_storage):
    """Test find_available_data_range when no data is available."""
    target_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    
    # Mock read_klines to raise FileNotFoundError
    mock_parquet_storage.read_klines.side_effect = FileNotFoundError("No data")
    
    result = await find_available_data_range(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        target_timestamp=target_timestamp,
        max_lookback_seconds=300,
    )
    
    # Should return None when no data available
    assert result is None


@pytest.mark.asyncio
async def test_find_available_data_range_fallback_to_previous_day(mock_parquet_storage, sample_klines_data):
    """Test find_available_data_range falls back to previous day when today's data is missing."""
    target_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    target_date = target_timestamp.date()
    target_date_str = target_date.strftime("%Y-%m-%d")
    previous_date = target_date - timedelta(days=1)
    previous_date_str = previous_date.strftime("%Y-%m-%d")
    
    # Today's data not available, but yesterday's is
    def read_klines_side_effect(symbol, date_str):
        if date_str == target_date_str:
            raise FileNotFoundError("No data today")
        elif date_str == previous_date_str:
            return sample_klines_data
        else:
            raise FileNotFoundError("No data")
    
    mock_parquet_storage.read_klines.side_effect = read_klines_side_effect
    
    result = await find_available_data_range(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        target_timestamp=target_timestamp,
        max_lookback_seconds=300,
    )
    
    # Should find data from previous day
    assert result is not None
    assert not result["historical_data"].empty


@pytest.mark.asyncio
async def test_load_historical_data_for_target_computation(mock_parquet_storage, sample_klines_data):
    """Test load_historical_data_for_target_computation."""
    prediction_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
    target_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    
    mock_parquet_storage.read_klines_range.return_value = sample_klines_data
    
    result = await load_historical_data_for_target_computation(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        prediction_timestamp=prediction_timestamp,
        target_timestamp=target_timestamp,
        buffer_seconds=60,
    )
    
    assert not result.empty
    assert "timestamp" in result.columns
    assert "close" in result.columns
    
    # Verify read_klines_range was called with correct date range
    mock_parquet_storage.read_klines_range.assert_called_once()
    call_args = mock_parquet_storage.read_klines_range.call_args
    assert call_args[0][0] == "BTCUSDT"  # symbol
    assert isinstance(call_args[0][1], type(prediction_timestamp.date()))  # start_date
    assert isinstance(call_args[0][2], type(target_timestamp.date()))  # end_date


@pytest.mark.asyncio
async def test_load_historical_data_for_target_computation_empty_result(mock_parquet_storage):
    """Test load_historical_data_for_target_computation when no data is available."""
    prediction_timestamp = datetime.now(timezone.utc) - timedelta(minutes=10)
    target_timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)
    
    mock_parquet_storage.read_klines_range.return_value = pd.DataFrame()
    
    result = await load_historical_data_for_target_computation(
        parquet_storage=mock_parquet_storage,
        symbol="BTCUSDT",
        prediction_timestamp=prediction_timestamp,
        target_timestamp=target_timestamp,
        buffer_seconds=60,
    )
    
    assert result.empty

