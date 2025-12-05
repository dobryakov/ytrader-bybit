"""
Test fixtures for raw market data storage.
"""
import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import pandas as pd


@pytest.fixture
def raw_orderbook_snapshot():
    """Raw orderbook snapshot event for storage testing."""
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
def raw_orderbook_delta():
    """Raw orderbook delta event for storage testing."""
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
def raw_orderbook_delta_insert():
    """Raw orderbook delta insert event for storage testing."""
    return {
        "event_type": "orderbook_delta",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": 1002,
        "delta_type": "insert",
        "side": "ask",
        "price": 50004.0,
        "quantity": 0.5,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_orderbook_delta_delete():
    """Raw orderbook delta delete event for storage testing."""
    return {
        "event_type": "orderbook_delta",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sequence": 1003,
        "delta_type": "delete",
        "side": "bid",
        "price": 49998.0,
        "quantity": 0.0,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_trade():
    """Raw trade event for storage testing."""
    return {
        "event_type": "trade",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": 50000.5,
        "quantity": 0.1,
        "side": "Buy",
        "trade_time": datetime.now(timezone.utc).isoformat(),
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_kline():
    """Raw kline/candlestick event for storage testing."""
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
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_ticker():
    """Raw ticker event for storage testing."""
    return {
        "event_type": "ticker",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "last_price": 50000.5,
        "bid_price": 50000.0,
        "ask_price": 50001.0,
        "volume_24h": 1000.0,
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_funding_rate():
    """Raw funding rate event for storage testing."""
    return {
        "event_type": "funding_rate",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "funding_rate": 0.0001,
        "next_funding_time": int((datetime.now(timezone.utc).timestamp() + 3600) * 1000),
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_execution_event():
    """Raw execution event for storage testing."""
    return {
        "event_type": "execution",
        "symbol": "BTCUSDT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "order_id": "order_123",
        "execution_id": "exec_456",
        "side": "Buy",
        "price": 50000.0,
        "quantity": 0.5,
        "status": "FILLED",
        "internal_timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange_timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def raw_market_data_batch():
    """Batch of raw market data events for storage testing."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_snapshot",
            "symbol": "BTCUSDT",
            "timestamp": base_time.isoformat(),
            "sequence": 1000,
            "bids": [[50000.0, 1.5], [49999.0, 2.0]],
            "asks": [[50001.0, 1.2], [50002.0, 2.5]],
            "internal_timestamp": base_time.isoformat(),
            "exchange_timestamp": base_time.isoformat(),
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": (base_time + timedelta(milliseconds=100)).isoformat(),
            "sequence": 1001,
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,
            "internal_timestamp": (base_time + timedelta(milliseconds=100)).isoformat(),
            "exchange_timestamp": (base_time + timedelta(milliseconds=100)).isoformat(),
        },
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": (base_time + timedelta(milliseconds=200)).isoformat(),
            "price": 50000.5,
            "quantity": 0.1,
            "side": "Buy",
            "internal_timestamp": (base_time + timedelta(milliseconds=200)).isoformat(),
            "exchange_timestamp": (base_time + timedelta(milliseconds=200)).isoformat(),
        },
        {
            "event_type": "kline",
            "symbol": "BTCUSDT",
            "timestamp": (base_time + timedelta(seconds=1)).isoformat(),
            "interval": "1m",
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 10.5,
            "internal_timestamp": (base_time + timedelta(seconds=1)).isoformat(),
            "exchange_timestamp": (base_time + timedelta(seconds=1)).isoformat(),
        },
    ]


@pytest.fixture
def raw_orderbook_deltas_sequence():
    """Sequence of orderbook deltas for offline reconstruction testing."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": (base_time + timedelta(milliseconds=i * 10)).isoformat(),
            "sequence": 1000 + i,
            "delta_type": "update" if i % 2 == 0 else "insert",
            "side": "bid" if i % 2 == 0 else "ask",
            "price": 50000.0 + (i * 0.1),
            "quantity": 1.0 + (i * 0.1),
            "internal_timestamp": (base_time + timedelta(milliseconds=i * 10)).isoformat(),
            "exchange_timestamp": (base_time + timedelta(milliseconds=i * 10)).isoformat(),
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_parquet_dataframe():
    """Sample pandas DataFrame for Parquet storage testing."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            datetime.now(timezone.utc) + timedelta(seconds=i)
            for i in range(5)
        ]),
        "symbol": ["BTCUSDT"] * 5,
        "price": [50000.0 + i for i in range(5)],
        "quantity": [1.0 + i * 0.1 for i in range(5)],
    })


@pytest.fixture
def expired_date():
    """Date that is older than retention period (for retention testing)."""
    return (datetime.now(timezone.utc) - timedelta(days=100)).date()


@pytest.fixture
def valid_date():
    """Date within retention period (for retention testing)."""
    return (datetime.now(timezone.utc) - timedelta(days=30)).date()


@pytest.fixture
def archive_path(tmp_path):
    """Temporary archive directory path."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return str(archive_dir)
