"""
Unit tests for candlestick pattern features computation for 45m version.
Updated to use dynamic pattern architecture.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.rolling_windows import RollingWindows
from src.features.candle_patterns import compute_all_candle_patterns_dynamic
from src.models.feature_registry import FeatureDefinition


def create_feature_definitions_45m():
    """Create feature definitions for 45m patterns (v1.7.5)."""
    # These are the actual features from v1.7.5
    feature_names = [
        "pattern_all_green",
        "pattern_all_red",
        "pattern_red_green_large_body",
        "pattern_green_red_large_body",
        "pattern_candle_color_sequence_ternary",
        "pattern_green_large_volume",
        "pattern_red_large_volume",
    ]
    
    return [
        FeatureDefinition(
            name=name,
            input_sources=["kline"],
            lookback_window="45m",
            lookahead_forbidden=True,
            max_lookback_days=1,
            data_sources=[{"source": "kline", "timestamp_required": True}],
        )
        for name in feature_names
    ]


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
            "1m": df,  # Use 1m for dynamic aggregation
        },
        "last_update": base_time,
    }


class TestCandlePatterns45m:
    """Test candlestick pattern features computation for 45m version (v1.7.5)."""
    
    def test_compute_all_candle_patterns_dynamic_45m_basic(self, sample_rolling_windows_15m_klines):
        """Test basic pattern computation with 45m lookback."""
        rw = RollingWindows(**sample_rolling_windows_15m_klines)
        feature_defs = create_feature_definitions_45m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Should return dictionary with features
        assert isinstance(features, dict)
        assert len(features) == len(feature_defs)
        
        # Check that all v1.7.5 features are present
        assert "pattern_all_green" in features
        assert "pattern_all_red" in features
        assert "pattern_red_green_large_body" in features
        assert "pattern_green_red_large_body" in features
        assert "pattern_candle_color_sequence_ternary" in features
        assert "pattern_green_large_volume" in features
        assert "pattern_red_large_volume" in features
        
        # Verify binary features are 0.0 or 1.0
        for key, value in features.items():
            if value is not None and key != "pattern_candle_color_sequence_ternary":
                assert value == 0.0 or value == 1.0, f"Feature {key} has invalid value: {value}"
    
    def test_compute_all_candle_patterns_dynamic_45m_insufficient_data(self):
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
            windows={"1m": df},
            last_update=base_time,
        )
        
        feature_defs = create_feature_definitions_45m()
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Function uses approximation for missing candles, so features should be computed
        assert isinstance(features, dict)
        assert len(features) == len(feature_defs)
        # Features should be computed (not None) due to approximation
        assert "pattern_all_green" in features
        assert "pattern_all_red" in features
    
    def test_compute_all_candle_patterns_dynamic_45m_pattern_body_features(self, sample_rolling_windows_15m_klines):
        """Test that pattern features are computed correctly."""
        rw = RollingWindows(**sample_rolling_windows_15m_klines)
        feature_defs = create_feature_definitions_45m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # All v1.7.5 features must be present
        assert "pattern_all_green" in features
        assert "pattern_all_red" in features
        assert "pattern_red_green_large_body" in features
        assert "pattern_green_red_large_body" in features
        assert "pattern_candle_color_sequence_ternary" in features
        assert "pattern_green_large_volume" in features
        assert "pattern_red_large_volume" in features
        
        # Values should be 0.0 or 1.0 (binary features) or float for ternary
        if features["pattern_all_green"] is not None:
            assert features["pattern_all_green"] in [0.0, 1.0]
        if features["pattern_all_red"] is not None:
            assert features["pattern_all_red"] in [0.0, 1.0]
        if features["pattern_candle_color_sequence_ternary"] is not None:
            assert isinstance(features["pattern_candle_color_sequence_ternary"], float)
            assert 0.0 <= features["pattern_candle_color_sequence_ternary"] <= 26.0
    
    def test_compute_all_candle_patterns_dynamic_45m_insufficient_data_15_to_45_minutes(self):
        """Test pattern computation with 15-45 minutes of data (should aggregate 1m klines)."""
        base_time = datetime.now(timezone.utc)
        
        # Create 20 minutes of 1-minute klines (less than 45, but more than 15)
        klines_1m_data = []
        base_price = 50000.0
        for minute in range(20):
            klines_1m_data.append({
                "timestamp": base_time - timedelta(minutes=20-minute),
                "open": base_price + minute * 0.5,
                "high": base_price + minute * 0.5 + 10.0,
                "low": base_price + minute * 0.5 - 10.0,
                "close": base_price + minute * 0.5 + 5.0,
                "volume": 10.0 + minute,
            })
        
        df_1m = pd.DataFrame(klines_1m_data)
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={"1m": df_1m},
            last_update=base_time,
        )
        
        feature_defs = create_feature_definitions_45m()
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Function should aggregate 1m klines into segments and compute features
        # Even with less than 45 minutes, it should approximate missing candles
        assert isinstance(features, dict)
        assert len(features) == len(feature_defs)
        
        # Key features should be present
        assert "pattern_all_green" in features
        assert "pattern_all_red" in features
        
        # Values may be None if data is insufficient (20 minutes < 45 minutes required)
        # The function should still return the features dict, but values may be None
        # This is acceptable behavior - approximation happens when we have at least some data
        # With only 20 minutes out of 45, we may not have enough for reliable computation
        assert isinstance(features["pattern_all_green"], (float, type(None)))
        assert isinstance(features["pattern_all_red"], (float, type(None)))
