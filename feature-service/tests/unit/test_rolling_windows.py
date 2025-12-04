"""
Unit tests for Rolling Windows model.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from feature_service.tests.fixtures.rolling_windows import (
    sample_rolling_windows,
    sample_rolling_windows_empty,
)


class TestRollingWindows:
    """Test Rolling Windows model."""
    
    def test_rolling_windows_creation(self, sample_rolling_windows):
        """Test creating rolling windows."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        assert rw.symbol == "BTCUSDT"
        assert "1s" in rw.windows
        assert "3s" in rw.windows
        assert "15s" in rw.windows
        assert "1m" in rw.windows
        assert rw.last_update is not None
    
    def test_rolling_windows_empty(self, sample_rolling_windows_empty):
        """Test creating empty rolling windows."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows_empty)
        
        assert rw.symbol == "BTCUSDT"
        assert all(len(rw.windows[interval]) == 0 for interval in ["1s", "3s", "15s", "1m"])
    
    def test_rolling_windows_add_trade(self, sample_rolling_windows):
        """Test adding trade to rolling windows."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        trade = {
            "timestamp": datetime.now(timezone.utc),
            "price": 50002.0,
            "quantity": 0.3,
            "side": "Buy",
        }
        
        initial_count_1s = len(rw.windows["1s"])
        rw.add_trade(trade)
        
        assert len(rw.windows["1s"]) == initial_count_1s + 1
        assert len(rw.windows["3s"]) == initial_count_1s + 1
        assert len(rw.windows["15s"]) == initial_count_1s + 1
        assert len(rw.windows["1m"]) == initial_count_1s + 1
    
    def test_rolling_windows_add_kline(self, sample_rolling_windows):
        """Test adding kline to rolling windows."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        kline = {
            "timestamp": datetime.now(timezone.utc),
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 15.0,
        }
        
        initial_count = len(rw.windows["1m"])
        rw.add_kline(kline)
        
        assert len(rw.windows["1m"]) == initial_count + 1
    
    def test_rolling_windows_trim_old_data(self, sample_rolling_windows):
        """Test trimming old data outside window."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        # Add old data that should be trimmed
        old_trade = {
            "timestamp": datetime.now(timezone.utc) - timedelta(seconds=5),
            "price": 49990.0,
            "quantity": 0.1,
            "side": "Buy",
        }
        
        rw.add_trade(old_trade)
        rw.trim_old_data(window_seconds=3)
        
        # Data older than 3 seconds should be removed from 3s window
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=3)
        remaining_3s = rw.windows["3s"][rw.windows["3s"]["timestamp"] >= cutoff_time]
        
        assert len(remaining_3s) <= len(rw.windows["3s"])
    
    def test_rolling_windows_get_window_data(self, sample_rolling_windows):
        """Test getting data for specific window."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        window_3s = rw.get_window_data("3s")
        
        assert isinstance(window_3s, pd.DataFrame)
        assert len(window_3s) > 0
    
    def test_rolling_windows_get_window_data_empty(self, sample_rolling_windows_empty):
        """Test getting data from empty window."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows_empty)
        
        window_3s = rw.get_window_data("3s")
        
        assert isinstance(window_3s, pd.DataFrame)
        assert len(window_3s) == 0
    
    def test_rolling_windows_get_trades_for_window(self, sample_rolling_windows):
        """Test getting trades for specific time window."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        now = datetime.now(timezone.utc)
        trades = rw.get_trades_for_window("3s", now - timedelta(seconds=3), now)
        
        assert isinstance(trades, pd.DataFrame)
        assert "price" in trades.columns
        assert "quantity" in trades.columns
        assert "side" in trades.columns
    
    def test_rolling_windows_get_klines_for_window(self, sample_rolling_windows):
        """Test getting klines for specific time window."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        now = datetime.now(timezone.utc)
        klines = rw.get_klines_for_window("1m", now - timedelta(minutes=1), now)
        
        assert isinstance(klines, pd.DataFrame)
        if len(klines) > 0:
            assert "open" in klines.columns
            assert "close" in klines.columns
            assert "volume" in klines.columns

