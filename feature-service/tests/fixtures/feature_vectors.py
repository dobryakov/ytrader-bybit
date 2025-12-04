"""
Test fixtures for feature vectors.
"""
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

# Note: These fixtures are used before models are implemented (TDD approach)


@pytest.fixture
def sample_feature_vector():
    """Sample feature vector with all computed features."""
    return {
        "timestamp": datetime.now(timezone.utc),
        "symbol": "BTCUSDT",
        "features": {
            # Price features
            "mid_price": 50000.5,
            "spread_abs": 1.0,
            "spread_rel": 0.00002,
            "returns_1s": 0.0001,
            "returns_3s": 0.0003,
            "returns_1m": 0.001,
            "vwap_3s": 50000.3,
            "vwap_15s": 50000.2,
            "vwap_1m": 50000.1,
            "volume_3s": 1.5,
            "volume_15s": 5.0,
            "volume_1m": 20.0,
            "volatility_1m": 0.0005,
            "volatility_5m": 0.002,
            # Orderflow features
            "signed_volume_3s": 0.8,
            "signed_volume_15s": 2.5,
            "signed_volume_1m": 10.0,
            "buy_sell_volume_ratio": 1.2,
            "trade_count_3s": 5,
            "net_aggressor_pressure": 0.15,
            # Orderbook features
            "depth_bid_top5": 10.0,
            "depth_bid_top10": 20.0,
            "depth_ask_top5": 12.0,
            "depth_ask_top10": 22.0,
            "depth_imbalance_top5": 0.1,
            # Perpetual features
            "funding_rate": 0.0001,
            "time_to_funding": 3600.0,
            # Temporal features
            "time_of_day_sin": 0.5,
            "time_of_day_cos": 0.866,
        },
        "feature_registry_version": "1.0.0",
        "trace_id": "test-trace-123",
    }


@pytest.fixture
def sample_feature_vector_minimal():
    """Minimal feature vector with only required fields."""
    return {
        "timestamp": datetime.now(timezone.utc),
        "symbol": "BTCUSDT",
        "features": {
            "mid_price": 50000.5,
        },
        "feature_registry_version": "1.0.0",
        "trace_id": "test-trace-456",
    }


@pytest.fixture
def sample_feature_vector_sequence():
    """Sequence of feature vectors over time."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "timestamp": base_time,
            "symbol": "BTCUSDT",
            "features": {
                "mid_price": 50000.0,
                "spread_abs": 1.0,
            },
            "feature_registry_version": "1.0.0",
            "trace_id": "test-trace-1",
        },
        {
            "timestamp": base_time.replace(microsecond=1000000),
            "symbol": "BTCUSDT",
            "features": {
                "mid_price": 50001.0,
                "spread_abs": 1.1,
            },
            "feature_registry_version": "1.0.0",
            "trace_id": "test-trace-2",
        },
        {
            "timestamp": base_time.replace(microsecond=2000000),
            "symbol": "BTCUSDT",
            "features": {
                "mid_price": 50002.0,
                "spread_abs": 1.2,
            },
            "feature_registry_version": "1.0.0",
            "trace_id": "test-trace-3",
        },
    ]


@pytest.fixture
def sample_feature_vector_multiple_symbols():
    """Feature vectors for multiple symbols."""
    base_time = datetime.now(timezone.utc)
    
    return {
        "BTCUSDT": {
            "timestamp": base_time,
            "symbol": "BTCUSDT",
            "features": {
                "mid_price": 50000.0,
                "spread_abs": 1.0,
            },
            "feature_registry_version": "1.0.0",
            "trace_id": "test-trace-btc",
        },
        "ETHUSDT": {
            "timestamp": base_time,
            "symbol": "ETHUSDT",
            "features": {
                "mid_price": 3000.0,
                "spread_abs": 0.5,
            },
            "feature_registry_version": "1.0.0",
            "trace_id": "test-trace-eth",
        },
    }

