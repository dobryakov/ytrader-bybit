"""
Test fixtures for orderbook state (snapshots and deltas).
"""
import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any
from sortedcontainers import SortedDict


@pytest.fixture
def sample_orderbook_state():
    """Sample orderbook state (in-memory structure)."""
    return {
        "symbol": "BTCUSDT",
        "sequence": 1000,
        "timestamp": datetime.now(timezone.utc),
        "bids": SortedDict({
            50000.0: 1.5,
            49999.0: 2.0,
            49998.0: 1.0,
            49997.0: 0.5,
        }),
        "asks": SortedDict({
            50001.0: 1.2,
            50002.0: 2.5,
            50003.0: 1.8,
            50004.0: 0.8,
        }),
        "last_snapshot_at": datetime.now(timezone.utc),
        "delta_count": 0,
    }


@pytest.fixture
def sample_orderbook_snapshot():
    """Sample orderbook snapshot for initialization."""
    now = datetime.now(timezone.utc)
    return {
        "event_type": "orderbook_snapshot",
        "symbol": "BTCUSDT",
        "timestamp": now,
        "sequence": 1000,
        "bids": [
            [50000.0, 1.5],
            [49999.0, 2.0],
            [49998.0, 1.0],
            [49997.0, 0.5],
        ],
        "asks": [
            [50001.0, 1.2],
            [50002.0, 2.5],
            [50003.0, 1.8],
            [50004.0, 0.8],
        ],
        "internal_timestamp": now,
        "exchange_timestamp": now,
    }


@pytest.fixture
def sample_orderbook_deltas():
    """Sequence of orderbook delta events."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "sequence": 1001,
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,  # Updated quantity
            "internal_timestamp": base_time,
            "exchange_timestamp": base_time,
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": base_time.replace(microsecond=100000),
            "sequence": 1002,
            "delta_type": "insert",
            "side": "ask",
            "price": 50005.0,
            "quantity": 1.0,
            "internal_timestamp": base_time.replace(microsecond=100000),
            "exchange_timestamp": base_time.replace(microsecond=100000),
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": base_time.replace(microsecond=200000),
            "sequence": 1003,
            "delta_type": "delete",
            "side": "bid",
            "price": 49997.0,
            "quantity": 0.0,
            "internal_timestamp": base_time.replace(microsecond=200000),
            "exchange_timestamp": base_time.replace(microsecond=200000),
        },
    ]


@pytest.fixture
def sample_orderbook_desynchronized():
    """Orderbook state with sequence gap (desynchronized)."""
    return {
        "symbol": "BTCUSDT",
        "sequence": 1000,
        "timestamp": datetime.now(timezone.utc),
        "bids": SortedDict({
            50000.0: 1.5,
            49999.0: 2.0,
        }),
        "asks": SortedDict({
            50001.0: 1.2,
            50002.0: 2.5,
        }),
        "last_snapshot_at": datetime.now(timezone.utc).replace(second=-10),
        "delta_count": 100,
    }


@pytest.fixture
def sample_orderbook_empty():
    """Empty orderbook state."""
    return {
        "symbol": "BTCUSDT",
        "sequence": 0,
        "timestamp": datetime.now(timezone.utc),
        "bids": SortedDict(),
        "asks": SortedDict(),
        "last_snapshot_at": None,
        "delta_count": 0,
    }

