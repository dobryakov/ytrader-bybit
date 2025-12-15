"""
Unit tests for AdaptiveCacheStrategy.
"""
import pytest

from src.services.optimized_dataset.cache_strategy import (
    AdaptiveCacheStrategy,
    CacheStrategy,
    CacheUnit,
)


@pytest.fixture
def cache_strategy_selector():
    """Create AdaptiveCacheStrategy instance."""
    return AdaptiveCacheStrategy()


def test_short_period_strategy(cache_strategy_selector):
    """Test cache strategy for short periods (1-3 days)."""
    strategy = cache_strategy_selector.determine_strategy(
        period_days=1,
        symbol="BTCUSDT",
        data_types=["klines", "trades"],
        max_lookback_minutes=30,
    )
    
    assert strategy.cache_unit == CacheUnit.FULL_PERIOD
    assert strategy.cache_size_days == 1
    assert strategy.prefetch_enabled is False


def test_medium_period_strategy(cache_strategy_selector):
    """Test cache strategy for medium periods (4-7 days)."""
    strategy = cache_strategy_selector.determine_strategy(
        period_days=5,
        symbol="BTCUSDT",
        data_types=["klines", "trades"],
        max_lookback_minutes=30,
    )
    
    assert strategy.cache_unit == CacheUnit.DAYS
    assert strategy.cache_size_days == 2
    assert strategy.prefetch_enabled is True
    assert strategy.prefetch_ahead_hours == 24


def test_long_period_strategy(cache_strategy_selector):
    """Test cache strategy for long periods (8+ days)."""
    strategy = cache_strategy_selector.determine_strategy(
        period_days=10,
        symbol="BTCUSDT",
        data_types=["klines", "trades"],
        max_lookback_minutes=30,
    )
    
    assert strategy.cache_unit == CacheUnit.DAYS
    assert strategy.cache_size_days == 1
    assert strategy.prefetch_enabled is True
    assert strategy.prefetch_ahead_hours == 2
    assert strategy.prefetch_adaptive is True


def test_cache_strategy_model():
    """Test CacheStrategy dataclass."""
    strategy = CacheStrategy(
        cache_unit=CacheUnit.DAYS,
        cache_size_days=2,
        prefetch_enabled=True,
        prefetch_ahead_hours=24,
        ttl_seconds=3600,
    )
    
    assert strategy.cache_unit == CacheUnit.DAYS
    assert strategy.cache_size_days == 2
    assert strategy.prefetch_enabled is True
    assert strategy.prefetch_ahead_hours == 24
    assert strategy.ttl_seconds == 3600


def test_cache_strategy_defaults():
    """Test CacheStrategy default values."""
    strategy = CacheStrategy(
        cache_unit=CacheUnit.FULL_PERIOD,
        cache_size_days=1,
    )
    
    assert strategy.prefetch_enabled is False
    assert strategy.prefetch_ahead_hours is None
    assert strategy.prefetch_adaptive is False
    assert strategy.ttl_seconds == 3600  # Default 1 hour
    assert strategy.local_buffer_minutes == 60  # Default 60 minutes

