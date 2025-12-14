"""
Unit tests for candlestick pattern features computation.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.rolling_windows import RollingWindows
from src.features.candle_patterns import compute_all_candle_patterns_3m


@pytest.fixture
def sample_rolling_windows_3_klines():
    """Rolling windows with exactly 3 klines for pattern testing."""
    base_time = datetime.now(timezone.utc)
    
    # Create 3 klines: oldest (0), middle (1), newest/current (2)
    # Pattern: green (large) → small red → red (large) - should trigger some patterns
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=3),
            "open": 50000.0,
            "high": 50100.0,
            "low": 49950.0,
            "close": 50050.0,  # Green candle (close > open)
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=2),
            "open": 50050.0,
            "high": 50060.0,
            "low": 50040.0,
            "close": 50045.0,  # Small red candle (close < open)
            "volume": 50.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50045.0,
            "high": 50055.0,
            "low": 49900.0,
            "close": 49950.0,  # Red candle (close < open)
            "volume": 150.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_all_green():
    """Rolling windows with 3 green candles (Three White Soldiers pattern)."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=3),
            "open": 50000.0,
            "high": 50050.0,
            "low": 49950.0,
            "close": 50030.0,  # Green
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=2),
            "open": 50030.0,
            "high": 50080.0,
            "low": 50020.0,
            "close": 50060.0,  # Green
            "volume": 120.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50060.0,
            "high": 50100.0,
            "low": 50050.0,
            "close": 50090.0,  # Green
            "volume": 140.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_doji():
    """Rolling windows with doji pattern."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=3),
            "open": 50000.0,
            "high": 50100.0,
            "low": 49950.0,
            "close": 50050.0,  # Large green
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=2),
            "open": 50050.0,
            "high": 50060.0,
            "low": 50040.0,
            "close": 50050.1,  # Doji (very small body)
            "volume": 50.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50050.0,
            "high": 50200.0,
            "low": 49900.0,
            "close": 50080.0,  # Large green
            "volume": 150.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_hammer():
    """Rolling windows with hammer pattern."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=3),
            "open": 50000.0,
            "high": 50050.0,
            "low": 49950.0,
            "close": 50030.0,
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=2),
            "open": 50030.0,
            "high": 50040.0,
            "low": 49900.0,  # Long lower shadow
            "close": 50035.0,  # Small body
            "volume": 120.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50035.0,
            "high": 50080.0,
            "low": 50020.0,
            "close": 50070.0,
            "volume": 140.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_engulfing():
    """Rolling windows with engulfing pattern."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=3),
            "open": 50000.0,
            "high": 50050.0,
            "low": 49950.0,
            "close": 50030.0,
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=2),
            "open": 50030.0,
            "high": 50040.0,
            "low": 50020.0,
            "close": 50025.0,  # Small red
            "volume": 50.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50020.0,  # Opens below previous close
            "high": 50080.0,
            "low": 50015.0,
            "close": 50075.0,  # Large green that engulfs previous
            "volume": 150.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_insufficient():
    """Rolling windows with less than 3 klines."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 50000.0,
            "high": 50050.0,
            "low": 49950.0,
            "close": 50030.0,
            "volume": 100.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1m": df,
        },
        "last_update": base_time,
    }


class TestCandlePatterns:
    """Test candlestick pattern features computation."""
    
    def test_compute_all_candle_patterns_3m_basic(self, sample_rolling_windows_3_klines):
        """Test basic pattern computation with 3 klines."""
        rw = RollingWindows(**sample_rolling_windows_3_klines)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Should return dictionary with all 77 features
        assert isinstance(features, dict)
        assert len(features) == 77
        
        # Check that all features are present
        assert "candle_0_is_green" in features
        assert "candle_2_is_red" in features
        assert "pattern_all_green" in features
        assert "pattern_evening_star" in features
        
        # Verify binary features are 0.0 or 1.0
        for key, value in features.items():
            if value is not None:
                assert value == 0.0 or value == 1.0, f"Feature {key} has invalid value: {value}"
    
    def test_compute_all_candle_patterns_3m_all_green(self, sample_rolling_windows_all_green):
        """Test pattern_all_green detection."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # All candles should be green
        assert features["candle_0_is_green"] == 1.0
        assert features["candle_1_is_green"] == 1.0
        assert features["candle_2_is_green"] == 1.0
        assert features["candle_0_is_red"] == 0.0
        assert features["candle_1_is_red"] == 0.0
        assert features["candle_2_is_red"] == 0.0
        
        # Pattern all green should be detected
        assert features["pattern_all_green"] == 1.0
        assert features["pattern_all_red"] == 0.0
    
    def test_compute_all_candle_patterns_3m_doji(self, sample_rolling_windows_doji):
        """Test doji pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_doji)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Middle candle should be detected as doji
        assert features["candle_1_is_doji"] == 1.0
        
        # Doji star pattern should be detected
        assert features["pattern_doji_star"] == 1.0
    
    def test_compute_all_candle_patterns_3m_hammer(self, sample_rolling_windows_hammer):
        """Test hammer pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_hammer)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Middle candle should be detected as hammer
        assert features["candle_1_is_hammer"] == 1.0
    
    def test_compute_all_candle_patterns_3m_engulfing(self, sample_rolling_windows_engulfing):
        """Test engulfing pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_engulfing)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Bullish engulfing should be detected (green candle 2 engulfs red candle 1)
        assert features["candle_1_is_red"] == 1.0
        assert features["candle_2_is_green"] == 1.0
        assert features["pattern_bullish_engulfing"] == 1.0
    
    def test_compute_all_candle_patterns_3m_insufficient_data(self, sample_rolling_windows_insufficient):
        """Test pattern computation with insufficient data (less than 3 klines)."""
        rw = RollingWindows(**sample_rolling_windows_insufficient)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Should return all features as None
        assert isinstance(features, dict)
        assert len(features) == 77
        for key, value in features.items():
            assert value is None, f"Feature {key} should be None with insufficient data, got {value}"
    
    def test_compute_all_candle_patterns_3m_body_trends(self, sample_rolling_windows_all_green):
        """Test body trend patterns."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Should detect body trends
        assert "pattern_body_increasing" in features
        assert "pattern_body_decreasing" in features
        assert "pattern_body_middle_peak" in features
        assert "pattern_body_middle_low" in features
    
    def test_compute_all_candle_patterns_3m_volume_patterns(self, sample_rolling_windows_all_green):
        """Test volume pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Should detect volume trends
        assert "pattern_volume_increasing" in features
        assert "pattern_volume_decreasing" in features
        assert "pattern_volume_middle_peak" in features
    
    def test_compute_all_candle_patterns_3m_empty_klines(self):
        """Test pattern computation with empty klines."""
        base_time = datetime.now(timezone.utc)
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            },
            last_update=base_time,
        )
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Should return all features as None
        assert isinstance(features, dict)
        assert len(features) == 77
        for key, value in features.items():
            assert value is None, f"Feature {key} should be None with empty data, got {value}"
    
    def test_compute_all_candle_patterns_3m_relative_thresholds(self, sample_rolling_windows_3_klines):
        """Test that relative thresholds work correctly."""
        rw = RollingWindows(**sample_rolling_windows_3_klines)
        
        features = compute_all_candle_patterns_3m(rw)
        
        # Check that body size features use relative thresholds
        # If body is larger than average, body_large should be 1.0
        assert "candle_0_body_large" in features
        assert "candle_0_body_small" in features
        assert features["candle_0_body_large"] in [0.0, 1.0]
        assert features["candle_0_body_small"] in [0.0, 1.0]
        
        # Same for shadows and volumes
        assert "candle_0_upper_shadow_large" in features
        assert "candle_0_lower_shadow_large" in features
        assert "candle_0_volume_large" in features

