"""
Unit tests for RollingWindows trim_old_data with feature lookback requirements.

Tests that trim_old_data preserves enough historical data for features
that require lookback periods (e.g., ema_21 needs 26 minutes, rsi_14 needs 19 minutes).
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.models.rolling_windows import RollingWindows
from src.features.technical_indicators import compute_ema_21, compute_rsi_14
from src.features.price_features import compute_volume_ratio_20


class TestRollingWindowsTrimFeatureLookback:
    """Test that trim_old_data preserves enough data for feature computation."""
    
    def test_trim_old_data_preserves_data_for_ema_21(self):
        """Test that trim_old_data preserves at least 26 minutes of data for ema_21."""
        # Create rolling windows with 30 minutes of kline data
        now = datetime.now(timezone.utc)
        klines_data = []
        for i in range(30):  # 30 minutes of data
            kline_time = now - timedelta(minutes=30-i)
            klines_data.append({
                "timestamp": kline_time,
                "open": 50000.0 + i * 0.1,
                "high": 50010.0 + i * 0.1,
                "low": 49990.0 + i * 0.1,
                "close": 50005.0 + i * 0.1,
                "volume": 100.0 + i * 0.1,
            })
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(klines_data),
            },
            last_update=now,
        )
        
        # Verify we have 30 klines
        assert len(rw.windows["1m"]) == 30
        
        # Call trim_old_data with max_lookback_minutes_1m=26 (for ema_21)
        # In production, this value is computed from Feature Registry
        rw.trim_old_data(max_lookback_minutes_1m=26)
        
        # After trimming, we should have at least 26 minutes of data (for ema_21)
        # With buffer (26 + 5 = 31 minutes), we should have 31 minutes
        remaining_klines = rw.windows["1m"]
        assert len(remaining_klines) >= 26, f"Expected at least 26 klines after trim, got {len(remaining_klines)}"
        
        # Verify ema_21 can be computed (needs 21 minutes + 5 minute buffer = 26 minutes)
        ema_21 = compute_ema_21(rw)
        assert ema_21 is not None, "ema_21 should be computable after trim_old_data"
        assert isinstance(ema_21, float)
    
    def test_trim_old_data_preserves_data_for_rsi_14(self):
        """Test that trim_old_data preserves at least 19 minutes of data for rsi_14."""
        # Create rolling windows with 25 minutes of kline data
        now = datetime.now(timezone.utc)
        klines_data = []
        for i in range(25):  # 25 minutes of data
            kline_time = now - timedelta(minutes=25-i)
            # Add some price variation for RSI computation
            price_change = (i % 3) - 1  # -1, 0, or 1
            klines_data.append({
                "timestamp": kline_time,
                "open": 50000.0 + i * 0.1 + price_change * 0.5,
                "high": 50010.0 + i * 0.1 + price_change * 0.5,
                "low": 49990.0 + i * 0.1 + price_change * 0.5,
                "close": 50005.0 + i * 0.1 + price_change * 0.5,
                "volume": 100.0 + i * 0.1,
            })
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(klines_data),
            },
            last_update=now,
        )
        
        # Call trim_old_data with max_lookback_minutes_1m=19 (for rsi_14)
        # In production, this value is computed from Feature Registry
        rw.trim_old_data(max_lookback_minutes_1m=19)
        
        # After trimming, we should have at least 19 minutes of data (for rsi_14)
        remaining_klines = rw.windows["1m"]
        assert len(remaining_klines) >= 19, f"Expected at least 19 klines after trim, got {len(remaining_klines)}"
        
        # Verify rsi_14 can be computed (needs 14 minutes + 5 minute buffer = 19 minutes)
        rsi_14 = compute_rsi_14(rw)
        assert rsi_14 is not None, "rsi_14 should be computable after trim_old_data"
        assert isinstance(rsi_14, float)
        assert 0 <= rsi_14 <= 100, f"RSI should be between 0 and 100, got {rsi_14}"
    
    def test_trim_old_data_preserves_data_for_volume_ratio_20(self):
        """Test that trim_old_data preserves at least 20 minutes of data for volume_ratio_20."""
        # Create rolling windows with 25 minutes of kline data
        now = datetime.now(timezone.utc)
        klines_data = []
        for i in range(25):  # 25 minutes of data
            kline_time = now - timedelta(minutes=25-i)
            klines_data.append({
                "timestamp": kline_time,
                "open": 50000.0 + i * 0.1,
                "high": 50010.0 + i * 0.1,
                "low": 49990.0 + i * 0.1,
                "close": 50005.0 + i * 0.1,
                "volume": 100.0 + i * 0.1,  # Varying volume
            })
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(klines_data),
            },
            last_update=now,
        )
        
        # Call trim_old_data with max_lookback_minutes_1m=20 (for volume_ratio_20)
        # In production, this value is computed from Feature Registry
        rw.trim_old_data(max_lookback_minutes_1m=20)
        
        # After trimming, we should have at least 20 minutes of data (for volume_ratio_20)
        remaining_klines = rw.windows["1m"]
        assert len(remaining_klines) >= 20, f"Expected at least 20 klines after trim, got {len(remaining_klines)}"
        
        # Get current volume from latest kline
        if len(remaining_klines) > 0:
            latest_kline = remaining_klines.iloc[-1]
            current_volume = float(latest_kline["volume"])
            
            # Verify volume_ratio_20 can be computed (needs 20 minutes)
            volume_ratio = compute_volume_ratio_20(rw, current_volume)
            assert volume_ratio is not None, "volume_ratio_20 should be computable after trim_old_data"
            assert isinstance(volume_ratio, float)
            assert volume_ratio > 0, f"Volume ratio should be positive, got {volume_ratio}"
    
    def test_trim_old_data_removes_very_old_data(self):
        """Test that trim_old_data still removes data older than 30 minutes."""
        # Create rolling windows with 40 minutes of kline data
        now = datetime.now(timezone.utc)
        klines_data = []
        for i in range(40):  # 40 minutes of data
            kline_time = now - timedelta(minutes=40-i)
            klines_data.append({
                "timestamp": kline_time,
                "open": 50000.0 + i * 0.1,
                "high": 50010.0 + i * 0.1,
                "low": 49990.0 + i * 0.1,
                "close": 50005.0 + i * 0.1,
                "volume": 100.0 + i * 0.1,
            })
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(klines_data),
            },
            last_update=now,
        )
        
        # Verify we have 40 klines initially
        assert len(rw.windows["1m"]) == 40
        
        # Call trim_old_data with max_lookback_minutes_1m=26 (should keep 31 minutes with buffer, remove 9 minutes)
        # In production, this value is computed from Feature Registry
        rw.trim_old_data(max_lookback_minutes_1m=26)
        
        # After trimming, we should have at most 31 minutes of data (26 + 5 buffer)
        remaining_klines = rw.windows["1m"]
        assert len(remaining_klines) <= 31, f"Expected at most 31 klines after trim, got {len(remaining_klines)}"
        assert len(remaining_klines) >= 26, f"Expected at least 26 klines after trim, got {len(remaining_klines)}"
        
        # Verify all remaining timestamps are within 31 minutes (26 + 5 buffer)
        if len(remaining_klines) > 0:
            cutoff_time = now - timedelta(minutes=31)
            oldest_timestamp = remaining_klines["timestamp"].min()
            assert oldest_timestamp >= cutoff_time, f"Oldest timestamp {oldest_timestamp} should be >= {cutoff_time}"
    
    def test_trim_old_data_uses_computed_max_lookback(self):
        """Test that trim_old_data uses computed max_lookback from Feature Registry."""
        # Create rolling windows with 35 minutes of kline data
        now = datetime.now(timezone.utc)
        klines_data = []
        for i in range(35):  # 35 minutes of data
            kline_time = now - timedelta(minutes=35-i)
            klines_data.append({
                "timestamp": kline_time,
                "open": 50000.0 + i * 0.1,
                "high": 50010.0 + i * 0.1,
                "low": 49990.0 + i * 0.1,
                "close": 50005.0 + i * 0.1,
                "volume": 100.0 + i * 0.1,
            })
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(klines_data),
            },
            last_update=now,
        )
        
        # Test with different max_lookback values
        # Test 1: max_lookback = 20 minutes (should preserve 25 minutes with buffer)
        rw_test1 = RollingWindows(
            symbol="BTCUSDT",
            windows={"1m": pd.DataFrame(klines_data)},
            last_update=now,
        )
        rw_test1.trim_old_data(max_lookback_minutes_1m=20)
        remaining_1 = rw_test1.windows["1m"]
        # Should preserve 20 + 5 = 25 minutes
        assert len(remaining_1) >= 20, f"Expected at least 20 klines, got {len(remaining_1)}"
        assert len(remaining_1) <= 25, f"Expected at most 25 klines, got {len(remaining_1)}"
        
        # Test 2: max_lookback = 26 minutes (should preserve 31 minutes with buffer)
        rw_test2 = RollingWindows(
            symbol="BTCUSDT",
            windows={"1m": pd.DataFrame(klines_data)},
            last_update=now,
        )
        rw_test2.trim_old_data(max_lookback_minutes_1m=26)
        remaining_2 = rw_test2.windows["1m"]
        # Should preserve 26 + 5 = 31 minutes
        assert len(remaining_2) >= 26, f"Expected at least 26 klines, got {len(remaining_2)}"
        assert len(remaining_2) <= 31, f"Expected at most 31 klines, got {len(remaining_2)}"
        
        # Test 3: max_lookback = None (should use default 30 minutes = 1800 seconds)
        rw_test3 = RollingWindows(
            symbol="BTCUSDT",
            windows={"1m": pd.DataFrame(klines_data)},
            last_update=now,
        )
        rw_test3.trim_old_data()
        remaining_3 = rw_test3.windows["1m"]
        # Should preserve 30 minutes (default)
        assert len(remaining_3) >= 30, f"Expected at least 30 klines (default), got {len(remaining_3)}"
        assert len(remaining_3) <= 30, f"Expected at most 30 klines (default), got {len(remaining_3)}"

