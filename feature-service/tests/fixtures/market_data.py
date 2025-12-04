"""
Test fixtures for market data events.
"""
import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any


@pytest.fixture
def sample_orderbook_snapshot():
    """Sample orderbook snapshot event."""
    return {
        "event_type": "orderbook_snapshot",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": 1000,
        "bids": [
            [50000.0, 1.5],
            [49999.0, 2.0],
            [49998.0, 1.0],
        ],
        "asks": [
            [50001.0, 1.2],
            [50002.0, 2.5],
            [50003.0, 1.8],
        ],
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_orderbook_delta():
    """Sample orderbook delta event."""
    return {
        "event_type": "orderbook_delta",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": 1001,
        "delta_type": "update",
        "side": "bid",
        "price": 50000.0,
        "quantity": 1.8,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_trade():
    """Sample trade event."""
    return {
        "event_type": "trade",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": 50000.5,
        "quantity": 0.1,
        "side": "Buy",
        "trade_time": datetime.now(timezone.utc).isoformat(),
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_kline():
    """Sample kline/candlestick event."""
    return {
        "event_type": "kline",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interval": "1m",
        "open": 50000.0,
        "high": 50010.0,
        "low": 49990.0,
        "close": 50005.0,
        "volume": 10.5,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_ticker():
    """Sample ticker event."""
    return {
        "event_type": "ticker",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_price": 50000.5,
        "bid_price": 50000.0,
        "ask_price": 50001.0,
        "volume_24h": 1000.0,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_funding_rate():
    """Sample funding rate event."""
    return {
        "event_type": "funding_rate",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "funding_rate": 0.0001,
        "next_funding_time": (datetime.now(timezone.utc).timestamp() + 3600) * 1000,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_market_data_sequence():
    """Sequence of market data events for testing."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_snapshot",
            "symbol": "BTCUSDT",
            "timestamp": base_time.isoformat(),
            "sequence": 1000,
            "bids": [[50000.0, 1.5], [49999.0, 2.0]],
            "asks": [[50001.0, 1.2], [50002.0, 2.5]],
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": (base_time.replace(microsecond=100000)).isoformat(),
            "sequence": 1001,
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,
        },
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": (base_time.replace(microsecond=200000)).isoformat(),
            "price": 50000.5,
            "quantity": 0.1,
            "side": "Buy",
        },
    ]

