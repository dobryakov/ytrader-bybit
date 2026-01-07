"""
Unit tests for candlestick pattern features computation.
Updated to use dynamic pattern architecture.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.rolling_windows import RollingWindows
from src.features.candle_patterns import compute_all_candle_patterns_dynamic
from src.models.feature_registry import FeatureDefinition


def create_feature_definitions_3m():
    """Create feature definitions for 3m patterns (all candle/pattern features)."""
    # List of all candle/pattern features that would be in a 3m registry
    feature_names = [
        "candle_0_is_green", "candle_1_is_green", "candle_2_is_green",
        "candle_0_is_red", "candle_1_is_red", "candle_2_is_red",
        "candle_0_body_large", "candle_0_body_small",
        "candle_1_body_large", "candle_1_body_small",
        "candle_2_body_large", "candle_2_body_small",
        "candle_0_upper_shadow_large", "candle_0_upper_shadow_small",
        "candle_1_upper_shadow_large", "candle_1_upper_shadow_small",
        "candle_2_upper_shadow_large", "candle_2_upper_shadow_small",
        "candle_0_lower_shadow_large", "candle_0_lower_shadow_small",
        "candle_1_lower_shadow_large", "candle_1_lower_shadow_small",
        "candle_2_lower_shadow_large", "candle_2_lower_shadow_small",
        "candle_0_volume_large", "candle_0_volume_small",
        "candle_1_volume_large", "candle_1_volume_small",
        "candle_2_volume_large", "candle_2_volume_small",
        "candle_0_is_doji", "candle_1_is_doji", "candle_2_is_doji",
        "candle_0_is_hammer", "candle_1_is_hammer", "candle_2_is_hammer",
        "pattern_all_green", "pattern_all_red",
        "pattern_green_red_green", "pattern_red_green_red",
        "pattern_body_increasing", "pattern_body_decreasing",
        "pattern_volume_increasing", "pattern_volume_decreasing",
        "pattern_green_large_volume", "pattern_red_large_volume",
        "pattern_bullish_engulfing", "pattern_bearish_engulfing",
        "pattern_red_green_large_body", "pattern_green_red_large_body",
        "pattern_candle_color_sequence_ternary",
        "pattern_upper_shadows_increasing", "pattern_lower_shadows_increasing",
        "pattern_evening_star", "pattern_morning_star", "pattern_doji_star",
        "pattern_bullish_harami", "pattern_bearish_harami",
        "pattern_rising_three_methods", "pattern_falling_three_methods",
        "pattern_inside_bar", "pattern_inside_bar_bullish", "pattern_inside_bar_bearish",
        "pattern_hanging_man", "pattern_tweezers_top", "pattern_tweezers_bottom",
    ]
    
    return [
        FeatureDefinition(
            name=name,
            input_sources=["kline"],
            lookback_window="3m",
            lookahead_forbidden=True,
            max_lookback_days=1,
            data_sources=[{"source": "kline", "timestamp_required": True}],
        )
        for name in feature_names
    ]


def create_feature_definitions_15m():
    """Create feature definitions for 15m patterns (compact format with ratios)."""
    feature_names = [
        "candle_0_is_green", "candle_1_is_green", "candle_2_is_green",
        "candle_0_body_ratio", "candle_1_body_ratio", "candle_2_body_ratio",
        "candle_0_upper_shadow_ratio", "candle_1_upper_shadow_ratio", "candle_2_upper_shadow_ratio",
        "candle_0_lower_shadow_ratio", "candle_1_lower_shadow_ratio", "candle_2_lower_shadow_ratio",
        "candle_0_is_doji", "candle_1_is_doji", "candle_2_is_doji",
        "candle_0_is_hammer", "candle_1_is_hammer", "candle_2_is_hammer",
        "pattern_all_green", "pattern_all_red",
        "pattern_body_increasing", "pattern_body_decreasing",
        "pattern_volume_increasing", "pattern_volume_decreasing",
        "pattern_green_large_volume", "pattern_red_large_volume",
        "pattern_bullish_engulfing", "pattern_bearish_engulfing",
        "pattern_red_green_large_body", "pattern_green_red_large_body",
        "pattern_candle_color_sequence_ternary",
    ]
    
    return [
        FeatureDefinition(
            name=name,
            input_sources=["kline"],
            lookback_window="15m",
            lookahead_forbidden=True,
            max_lookback_days=1,
            data_sources=[{"source": "kline", "timestamp_required": True}],
        )
        for name in feature_names
    ]


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
    
    def test_compute_all_candle_patterns_dynamic_basic(self, sample_rolling_windows_3_klines):
        """Test basic pattern computation with 3 klines."""
        rw = RollingWindows(**sample_rolling_windows_3_klines)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Should return dictionary with features from registry
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check that all requested features are present
        assert "candle_0_is_green" in features
        assert "candle_2_is_green" in features
        assert "pattern_all_green" in features
        assert "pattern_evening_star" in features
        
        # Verify binary features are 0.0 or 1.0 (except ternary which is 0-26)
        for key, value in features.items():
            if value is not None:
                if key == "pattern_candle_color_sequence_ternary":
                    assert isinstance(value, float) and 0.0 <= value <= 26.0, f"Feature {key} should be 0-26, got {value}"
                else:
                    assert value == 0.0 or value == 1.0, f"Feature {key} has invalid value: {value}"
    
    def test_compute_all_candle_patterns_dynamic_all_green(self, sample_rolling_windows_all_green):
        """Test pattern_all_green detection."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # All candles should be green
        assert features["candle_0_is_green"] == 1.0
        assert features["candle_1_is_green"] == 1.0
        assert features["candle_2_is_green"] == 1.0
        
        # Pattern all green should be detected
        assert features["pattern_all_green"] == 1.0
        assert features["pattern_all_red"] == 0.0
    
    def test_compute_all_candle_patterns_dynamic_doji(self, sample_rolling_windows_doji):
        """Test doji pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_doji)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Middle candle should be detected as doji (if conditions are met)
        assert "candle_1_is_doji" in features
        assert features["candle_1_is_doji"] in [0.0, 1.0]
        
        # Doji star pattern should be detected (if doji is in middle)
        if features["candle_1_is_doji"] == 1.0:
            assert features["pattern_doji_star"] == 1.0
    
    def test_compute_all_candle_patterns_dynamic_hammer(self, sample_rolling_windows_hammer):
        """Test hammer pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_hammer)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Middle candle should be detected as hammer (if conditions are met)
        assert "candle_1_is_hammer" in features
        assert features["candle_1_is_hammer"] in [0.0, 1.0]
    
    def test_compute_all_candle_patterns_dynamic_engulfing(self, sample_rolling_windows_engulfing):
        """Test engulfing pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_engulfing)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Bullish engulfing should be detected (green candle 2 engulfs red candle 1)
        assert features["candle_1_is_green"] == 0.0  # Red candle means is_green = 0.0
        assert features["candle_2_is_green"] == 1.0
        assert features["pattern_bullish_engulfing"] == 1.0
    
    def test_compute_all_candle_patterns_dynamic_insufficient_data(self, sample_rolling_windows_insufficient):
        """Test pattern computation with insufficient data (less than 3 klines)."""
        rw = RollingWindows(**sample_rolling_windows_insufficient)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Function uses approximation for missing candles, so features should be computed
        assert isinstance(features, dict)
        # Features should be computed (not None) due to approximation
        assert "candle_0_is_green" in features
        assert "candle_1_is_green" in features
        assert "candle_2_is_green" in features
    
    def test_compute_all_candle_patterns_dynamic_body_trends(self, sample_rolling_windows_all_green):
        """Test body trend patterns."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Should detect body trends
        assert "pattern_body_increasing" in features
        assert "pattern_body_decreasing" in features
    
    def test_compute_all_candle_patterns_dynamic_volume_patterns(self, sample_rolling_windows_all_green):
        """Test volume pattern detection."""
        rw = RollingWindows(**sample_rolling_windows_all_green)
        feature_defs = create_feature_definitions_3m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Should detect volume trends
        assert "pattern_volume_increasing" in features
        assert "pattern_volume_decreasing" in features
    
    def test_compute_all_candle_patterns_dynamic_empty_klines(self):
        """Test pattern computation with empty klines."""
        base_time = datetime.now(timezone.utc)
        
        rw = RollingWindows(
            symbol="BTCUSDT",
            windows={
                "1m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            },
            last_update=base_time,
        )
        
        feature_defs = create_feature_definitions_3m()
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
        # Should return features but values may be None
        assert isinstance(features, dict)
        assert len(features) > 0


class TestCandlePatterns15m:
    """Test candlestick pattern features computation for 15m version."""
    
    @pytest.fixture
    def sample_rolling_windows_5m_klines(self):
        """Rolling windows with 5-minute klines for 15m pattern testing."""
        base_time = datetime.now(timezone.utc)
        
        # Create 3 klines of 5 minutes each (15 minutes total)
        klines_data = [
            {
                "timestamp": base_time - timedelta(minutes=15),
                "open": 50000.0,
                "high": 50100.0,
                "low": 49950.0,
                "close": 50050.0,  # Green candle
                "volume": 100.0,
            },
            {
                "timestamp": base_time - timedelta(minutes=10),
                "open": 50050.0,
                "high": 50060.0,
                "low": 50040.0,
                "close": 50045.0,  # Small red candle
                "volume": 50.0,
            },
            {
                "timestamp": base_time - timedelta(minutes=5),
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
    
    def test_compute_all_candle_patterns_dynamic_15m_basic(self, sample_rolling_windows_5m_klines):
        """Test basic pattern computation with 15m lookback."""
        rw = RollingWindows(**sample_rolling_windows_5m_klines)
        feature_defs = create_feature_definitions_15m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
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
        assert "pattern_body_increasing" in features
        assert "pattern_volume_increasing" in features
        assert "pattern_bullish_engulfing" in features
        
        # Verify binary features are 0.0 or 1.0, ratio features are floats
        for key, value in features.items():
            if value is not None:
                if "ratio" in key:
                    assert isinstance(value, float), f"Feature {key} should be float, got {type(value)}"
                    assert 0.0 <= value <= 1.0, f"Feature {key} should be between 0 and 1, got {value}"
                elif key == "pattern_candle_color_sequence_ternary":
                    assert isinstance(value, float) and 0.0 <= value <= 26.0, f"Feature {key} should be 0-26, got {value}"
                else:
                    assert value == 0.0 or value == 1.0, f"Feature {key} has invalid value: {value}"
    
    def test_compute_all_candle_patterns_dynamic_15m_ratios(self, sample_rolling_windows_5m_klines):
        """Test that ratio features are computed correctly."""
        rw = RollingWindows(**sample_rolling_windows_5m_klines)
        feature_defs = create_feature_definitions_15m()
        
        features = compute_all_candle_patterns_dynamic(rw, feature_definitions=feature_defs)
        
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
