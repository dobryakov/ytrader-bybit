"""
Test fixtures for market data streams (orderbook, trades, klines, ticker, funding).
"""
import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any


@pytest.fixture
def sample_orderbook_stream():
    """Stream of orderbook events (snapshots and deltas)."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_snapshot",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
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
            "internal_timestamp": base_time,
            "exchange_timestamp": base_time,
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=100),
            "sequence": 1001,
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,
            "internal_timestamp": base_time + timedelta(milliseconds=100),
            "exchange_timestamp": base_time + timedelta(milliseconds=100),
        },
        {
            "event_type": "orderbook_delta",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=200),
            "sequence": 1002,
            "delta_type": "insert",
            "side": "ask",
            "price": 50005.0,
            "quantity": 1.0,
            "internal_timestamp": base_time + timedelta(milliseconds=200),
            "exchange_timestamp": base_time + timedelta(milliseconds=200),
        },
    ]


@pytest.fixture
def sample_trades_stream():
    """Stream of trade events."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "price": 50000.5,
            "quantity": 0.1,
            "side": "Buy",
            "trade_time": base_time,
            "internal_timestamp": base_time,
        },
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=500),
            "price": 50001.0,
            "quantity": 0.2,
            "side": "Sell",
            "trade_time": base_time + timedelta(milliseconds=500),
            "internal_timestamp": base_time + timedelta(milliseconds=500),
        },
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=1000),
            "price": 50000.8,
            "quantity": 0.15,
            "side": "Buy",
            "trade_time": base_time + timedelta(milliseconds=1000),
            "internal_timestamp": base_time + timedelta(milliseconds=1000),
        },
    ]


@pytest.fixture
def sample_klines_stream():
    """Stream of kline/candlestick events."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "kline",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "interval": "1m",
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 10.5,
            "internal_timestamp": base_time,
        },
        {
            "event_type": "kline",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(minutes=1),
            "interval": "1m",
            "open": 50005.0,
            "high": 50015.0,
            "low": 50000.0,
            "close": 50010.0,
            "volume": 12.0,
            "internal_timestamp": base_time + timedelta(minutes=1),
        },
    ]


@pytest.fixture
def sample_ticker_stream():
    """Stream of ticker events."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "ticker",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "last_price": 50000.5,
            "bid_price": 50000.0,
            "ask_price": 50001.0,
            "volume_24h": 1000.0,
            "internal_timestamp": base_time,
        },
        {
            "event_type": "ticker",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(seconds=1),
            "last_price": 50001.0,
            "bid_price": 50000.5,
            "ask_price": 50001.5,
            "volume_24h": 1001.0,
            "internal_timestamp": base_time + timedelta(seconds=1),
        },
    ]


@pytest.fixture
def sample_funding_stream():
    """Stream of funding rate events."""
    base_time = datetime.now(timezone.utc)
    next_funding = int((base_time + timedelta(hours=8)).timestamp() * 1000)
    
    return [
        {
            "event_type": "funding_rate",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "funding_rate": 0.0001,
            "next_funding_time": next_funding,
            "internal_timestamp": base_time,
        },
        {
            "event_type": "funding_rate",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(hours=8),
            "funding_rate": 0.00015,
            "next_funding_time": int((base_time + timedelta(hours=16)).timestamp() * 1000),
            "internal_timestamp": base_time + timedelta(hours=8),
        },
    ]


@pytest.fixture
def sample_mixed_market_data_stream():
    """Mixed stream of all market data types."""
    base_time = datetime.now(timezone.utc)
    
    return [
        {
            "event_type": "orderbook_snapshot",
            "symbol": "BTCUSDT",
            "timestamp": base_time,
            "sequence": 1000,
            "bids": [[50000.0, 1.5]],
            "asks": [[50001.0, 1.2]],
            "internal_timestamp": base_time,
            "exchange_timestamp": base_time,
        },
        {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=100),
            "price": 50000.5,
            "quantity": 0.1,
            "side": "Buy",
            "trade_time": base_time + timedelta(milliseconds=100),
            "internal_timestamp": base_time + timedelta(milliseconds=100),
        },
        {
            "event_type": "ticker",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(milliseconds=200),
            "last_price": 50000.5,
            "bid_price": 50000.0,
            "ask_price": 50001.0,
            "volume_24h": 1000.0,
            "internal_timestamp": base_time + timedelta(milliseconds=200),
        },
        {
            "event_type": "kline",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(seconds=1),
            "interval": "1m",
            "open": 50000.0,
            "high": 50010.0,
            "low": 49990.0,
            "close": 50005.0,
            "volume": 10.5,
            "internal_timestamp": base_time + timedelta(seconds=1),
        },
        {
            "event_type": "funding_rate",
            "symbol": "BTCUSDT",
            "timestamp": base_time + timedelta(seconds=2),
            "funding_rate": 0.0001,
            "next_funding_time": int((base_time + timedelta(hours=8)).timestamp() * 1000),
            "internal_timestamp": base_time + timedelta(seconds=2),
        },
    ]

