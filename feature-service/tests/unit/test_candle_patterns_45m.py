"""
Unit tests for candlestick pattern features computation for 45m version.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.rolling_windows import RollingWindows
from src.features.candle_patterns import compute_all_candle_patterns_45m


@pytest.fixture
def sample_rolling_windows_15m_klines():
    """Rolling windows with 15-minute klines for 45m pattern testing."""
    base_time = datetime.now(timezone.utc)
    
    # Create 3 klines of 15 minutes each (45 minutes total)
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=45),
            "open": 50000.0,
            "high": 50100.0,
            "low": 49950.0,
            "close": 50050.0,  # Green candle
            "volume": 100.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=30),
            "open": 50050.0,
            "high": 50060.0,
            "low": 50040.0,
            "close": 50045.0,  # Small red candle
            "volume": 50.0,
        },
        {
            "timestamp": base_time - timedelta(minutes=15),
            "open": 50045.0,
            "high": 50055.0,
            "low": 49900.0,
            "close": 49950.0,  # Red candle
            "volume": 150.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "15m": df,
        },
        "last_update": base_time,
    }


class TestCandlePatterns45m:
    """Test candlestick pattern features computation for 45m version (v1.7.3)."""
    
    def test_compute_all_candle_patterns_45m_basic(self, sample_rolling_windows_15m_klines):
        """Test basic pattern computation with 15-minute klines."""
        rw = RollingWindows(**sample_rolling_windows_15m_klines)
        
        features = compute_all_candle_patterns_45m(rw)
        
        # Should return dictionary with features
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check that all features are present
        assert "candle_0_is_green" in features
        assert "candle_0_body_ratio" in features
        assert "candle_0_upper_shadow_ratio" in features
        assert "candle_0_lower_shadow_ratio" in features
        assert "candle_0_is_doji" in features
        assert "candle_0_is_hammer" in features
        
        assert "pattern_all_green" in features
        assert "pattern_volume_increasing" in features
        assert "pattern_volume_decreasing" in features
        assert "pattern_green_large_volume" in features
        
        # Verify binary features are 0.0 or 1.0, ratio features are floats
        for key, value in features.items():
            if value is not None:
                if "ratio" in key:
                    assert isinstance(value, float), f"Feature {key} should be float, got {type(value)}"
                    assert 0.0 <= value <= 1.0, f"Feature {key} should be between 0 and 1, got {value}"
                else:
                    assert value == 0.0 or value == 1.0, f"Feature {key} has invalid value: {value}"
    
    def test_compute_all_candle_patterns_45m_insufficient_data(self):
        """Test pattern computation with insufficient data."""
        base_time = datetime.now(timezone.utc)
        
        # Only 2 klines (need 3)
        klines_data = [
            {
                "timestamp": base_time - timedelta(minutes=30),
                "open": 50000.0,
                "high": 50050.0,
                "low": 49950.0,
                "close": 50030.0,
                "volume": 100.0,
            },
            {
                "timestamp": base_time - timedelta(minutes=15),
                "open": 50030.0,
                "high": 50040.0,
                "low": 50020.0,
                "close": 50035.0,
                "volume": 50.0,
            },
        ]
        
        df = pd.DataFrame(klines_data)
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={"15m": df},
            last_update=base_time,
        )
        
        features = compute_all_candle_patterns_45m(rw)
        
        # Function uses approximation for missing candles, so features should be computed
        assert isinstance(features, dict)
        assert "candle_0_is_green" in features
        assert "candle_1_is_green" in features
        assert "candle_2_is_green" in features
    
    def test_compute_all_candle_patterns_45m_ratios(self, sample_rolling_windows_15m_klines):
        """Test that ratio features are computed correctly."""
        rw = RollingWindows(**sample_rolling_windows_15m_klines)
        
        features = compute_all_candle_patterns_45m(rw)
        
        # Check body ratios
        assert "candle_0_body_ratio" in features
        assert "candle_1_body_ratio" in features
        assert "candle_2_body_ratio" in features
        
        # Check shadow ratios
        assert "candle_0_upper_shadow_ratio" in features
        assert "candle_0_lower_shadow_ratio" in features
        
        # Ratios should be between 0 and 1
        for key in ["candle_0_body_ratio", "candle_1_body_ratio", "candle_2_body_ratio",
                    "candle_0_upper_shadow_ratio", "candle_0_lower_shadow_ratio"]:
            if features[key] is not None:
                assert 0.0 <= features[key] <= 1.0, f"{key} should be between 0 and 1"

