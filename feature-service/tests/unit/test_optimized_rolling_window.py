"""
Unit tests for OptimizedRollingWindow.
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.services.optimized_dataset.rolling_window import OptimizedRollingWindow
from src.models.rolling_windows import RollingWindows


@pytest.fixture
def rolling_window():
    """Create OptimizedRollingWindow instance."""
    return OptimizedRollingWindow(max_lookback_minutes=30, symbol="BTCUSDT")


@pytest.fixture
def sample_trades():
    """Create sample trades DataFrame."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i) for i in range(10)],
        "price": [50000.0 + i * 10 for i in range(10)],
        "volume": [1.0] * 10,
        "side": ["Buy"] * 10,
    })


@pytest.fixture
def sample_klines():
    """Create sample klines DataFrame."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(5)],
        "open": [50000.0 + i * 100 for i in range(5)],
        "high": [50100.0 + i * 100 for i in range(5)],
        "low": [49900.0 + i * 100 for i in range(5)],
        "close": [50050.0 + i * 100 for i in range(5)],
        "volume": [100.0] * 5,
    })


def test_initialization(rolling_window):
    """Test rolling window initialization."""
    assert rolling_window.symbol == "BTCUSDT"
    assert rolling_window.max_lookback == timedelta(minutes=35)  # 30 + 5 buffer
    assert rolling_window.trades_buffer.empty
    assert rolling_window.klines_buffer.empty
    assert rolling_window._last_timestamp is None


def test_add_trades(rolling_window, sample_trades):
    """Test adding trades to rolling window."""
    current_time = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
    rolling_window.add_data(current_time, trades=sample_trades)
    
    assert not rolling_window.trades_buffer.empty
    assert len(rolling_window.trades_buffer) == 10
    assert rolling_window._last_timestamp == current_time


def test_add_klines(rolling_window, sample_klines):
    """Test adding klines to rolling window."""
    current_time = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    rolling_window.add_data(current_time, klines=sample_klines)
    
    assert not rolling_window.klines_buffer.empty
    assert len(rolling_window.klines_buffer) == 5
    assert rolling_window._last_timestamp == current_time


def test_trim_old_data(rolling_window, sample_trades):
    """Test trimming old data outside lookback window."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Add old trades (outside lookback)
    old_trades = pd.DataFrame({
        "timestamp": [base_time - timedelta(minutes=40)],
        "price": [49000.0],
        "volume": [1.0],
        "side": ["Buy"],
    })
    
    # Add recent trades (within lookback)
    recent_trades = pd.DataFrame({
        "timestamp": [base_time],
        "price": [50000.0],
        "volume": [1.0],
        "side": ["Buy"],
    })
    
    current_time = base_time + timedelta(minutes=5)
    rolling_window.add_data(current_time, trades=old_trades)
    rolling_window.add_data(current_time, trades=recent_trades)
    
    # Old trades should be trimmed
    assert len(rolling_window.trades_buffer) == 1
    assert rolling_window.trades_buffer.iloc[0]["timestamp"] == base_time


def test_get_window(rolling_window, sample_trades, sample_klines):
    """Test getting RollingWindows for specific timestamp."""
    current_time = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    rolling_window.add_data(current_time, trades=sample_trades, klines=sample_klines)
    
    window = rolling_window.get_window(current_time)
    
    assert isinstance(window, RollingWindows)
    assert window.symbol == "BTCUSDT"
    assert window.last_update == current_time
    assert "1s" in window.windows
    assert "3s" in window.windows
    assert "15s" in window.windows
    assert "1m" in window.windows


def test_get_window_filters_by_lookback(rolling_window, sample_trades):
    """Test that get_window filters data by lookback window."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Add trades spanning 1 hour
    trades = pd.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(60)],
        "price": [50000.0] * 60,
        "volume": [1.0] * 60,
        "side": ["Buy"] * 60,
    })
    
    current_time = base_time + timedelta(minutes=60)
    rolling_window.add_data(current_time, trades=trades)
    
    # Get window - should only include last 35 minutes (max_lookback)
    window = rolling_window.get_window(current_time)
    
    # Check that 1m window has data within lookback
    if not window.windows["1m"].empty:
        window_timestamps = window.windows["1m"]["timestamp"]
        cutoff = current_time - rolling_window.max_lookback
        assert window_timestamps.min() >= cutoff


def test_clear(rolling_window, sample_trades, sample_klines):
    """Test clearing rolling window."""
    current_time = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    rolling_window.add_data(current_time, trades=sample_trades, klines=sample_klines)
    
    assert not rolling_window.trades_buffer.empty
    assert not rolling_window.klines_buffer.empty
    
    rolling_window.clear()
    
    assert rolling_window.trades_buffer.empty
    assert rolling_window.klines_buffer.empty
    assert rolling_window._last_timestamp is None


def test_get_buffer_stats(rolling_window, sample_trades):
    """Test getting buffer statistics."""
    current_time = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)
    rolling_window.add_data(current_time, trades=sample_trades)
    
    stats = rolling_window.get_buffer_stats()
    
    assert stats["trades_count"] == 10
    assert stats["klines_count"] == 0
    assert stats["last_timestamp"] == current_time.isoformat()
    assert stats["max_lookback_minutes"] == 35


def test_incremental_updates(rolling_window):
    """Test incremental updates to rolling window."""
    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # Add first batch
    trades1 = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i) for i in range(5)],
        "price": [50000.0] * 5,
        "volume": [1.0] * 5,
        "side": ["Buy"] * 5,
    })
    rolling_window.add_data(base_time + timedelta(seconds=5), trades=trades1)
    
    assert len(rolling_window.trades_buffer) == 5
    
    # Add second batch
    trades2 = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=i) for i in range(5, 10)],
        "price": [50000.0] * 5,
        "volume": [1.0] * 5,
        "side": ["Buy"] * 5,
    })
    rolling_window.add_data(base_time + timedelta(seconds=10), trades=trades2)
    
    # Should have all 10 trades
    assert len(rolling_window.trades_buffer) == 10

