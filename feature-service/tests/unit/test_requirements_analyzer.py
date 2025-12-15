"""
Unit tests for FeatureRequirementsAnalyzer.
"""
import pytest
from datetime import datetime, timezone

from src.models.feature_registry import FeatureRegistry, FeatureDefinition
from src.services.optimized_dataset.requirements_analyzer import (
    FeatureRequirementsAnalyzer,
    TimestampStrategy,
)


@pytest.fixture
def sample_registry():
    """Create a sample Feature Registry for testing."""
    features = [
        FeatureDefinition(
            name="mid_price",
            input_sources=["orderbook"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0,
        ),
        FeatureDefinition(
            name="returns_1m",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
        FeatureDefinition(
            name="signed_volume_1m",
            input_sources=["trades"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
        FeatureDefinition(
            name="ema_21",
            input_sources=["kline"],
            lookback_window="21m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
        FeatureDefinition(
            name="funding_rate",
            input_sources=["funding"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0,
        ),
    ]
    
    return FeatureRegistry(version="1.0.0", features=features)


@pytest.fixture
def analyzer():
    """Create FeatureRequirementsAnalyzer instance."""
    return FeatureRequirementsAnalyzer()


def test_analyze_basic(analyzer, sample_registry):
    """Test basic analysis of Feature Registry."""
    requirements = analyzer.analyze(sample_registry)
    
    assert requirements is not None
    assert "orderbook" in requirements.required_data_types
    assert "kline" in requirements.required_data_types
    assert "trades" in requirements.required_data_types
    assert "funding" in requirements.required_data_types
    
    assert requirements.needs_orderbook is True
    assert requirements.needs_klines is True
    assert requirements.needs_trades is True
    assert requirements.needs_funding is True


def test_max_lookback_computation(analyzer, sample_registry):
    """Test maximum lookback computation."""
    requirements = analyzer.analyze(sample_registry)
    
    # Should be at least 21 minutes (from ema_21) + 5 minute buffer = 26 minutes
    # But minimum is 30 minutes
    assert requirements.max_lookback_minutes >= 30


def test_timestamp_strategy_klines_only(analyzer):
    """Test timestamp strategy when only klines are needed."""
    features = [
        FeatureDefinition(
            name="returns_1m",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
    ]
    registry = FeatureRegistry(version="1.0.0", features=features)
    
    requirements = analyzer.analyze(registry)
    assert requirements.timestamp_strategy == TimestampStrategy.KLINES_ONLY


def test_timestamp_strategy_trades_only(analyzer):
    """Test timestamp strategy when only trades are needed."""
    features = [
        FeatureDefinition(
            name="signed_volume_1m",
            input_sources=["trades"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
    ]
    registry = FeatureRegistry(version="1.0.0", features=features)
    
    requirements = analyzer.analyze(registry)
    assert requirements.timestamp_strategy == TimestampStrategy.TRADES_ONLY


def test_timestamp_strategy_both(analyzer):
    """Test timestamp strategy when both klines and trades are needed."""
    features = [
        FeatureDefinition(
            name="returns_1m",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
        FeatureDefinition(
            name="signed_volume_1m",
            input_sources=["trades"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1,
        ),
    ]
    registry = FeatureRegistry(version="1.0.0", features=features)
    
    requirements = analyzer.analyze(registry)
    assert requirements.timestamp_strategy == TimestampStrategy.KLINES_WITH_TRADES


def test_parse_lookback_window(analyzer):
    """Test lookback window parsing."""
    # Test seconds
    assert analyzer._parse_lookback_window("60s") == 1  # 60 seconds = 1 minute
    
    # Test minutes
    assert analyzer._parse_lookback_window("21m") == 21
    assert analyzer._parse_lookback_window("5m") == 5
    
    # Test hours
    assert analyzer._parse_lookback_window("1h") == 60
    
    # Test days
    assert analyzer._parse_lookback_window("1d") == 1440  # 24 * 60
    
    # Test invalid
    assert analyzer._parse_lookback_window("invalid") is None
    assert analyzer._parse_lookback_window("") is None


def test_feature_grouping(analyzer, sample_registry):
    """Test feature grouping."""
    requirements = analyzer.analyze(sample_registry)
    
    assert "price" in requirements.feature_groups
    assert "technical" in requirements.feature_groups
    assert "orderflow" in requirements.feature_groups
    assert "perpetual" in requirements.feature_groups
    
    # Check that features are grouped correctly
    assert "ema_21" in requirements.feature_groups["technical"]
    assert "mid_price" in requirements.feature_groups["price"]  # mid_price is in price group, not orderbook


def test_storage_types_mapping(analyzer, sample_registry):
    """Test storage types mapping."""
    requirements = analyzer.analyze(sample_registry)
    
    assert "orderbook" in requirements.storage_types
    assert "orderbook_snapshots" in requirements.storage_types["orderbook"]
    assert "orderbook_deltas" in requirements.storage_types["orderbook"]
    
    assert "kline" in requirements.storage_types
    assert "klines" in requirements.storage_types["kline"]
    
    assert "trades" in requirements.storage_types
    assert "trades" in requirements.storage_types["trades"]
    
    assert "funding" in requirements.storage_types
    assert "funding" in requirements.storage_types["funding"]

