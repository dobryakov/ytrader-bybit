"""
Test fixtures for rolling windows data.
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict


@pytest.fixture
def sample_rolling_windows():
    """Sample rolling windows structure for all intervals."""
    base_time = datetime.now(timezone.utc)
    
    # Create sample data for each window
    window_1s = pd.DataFrame([
        {
            "timestamp": base_time - timedelta(seconds=0.5),
            "price": 50000.0,
            "volume": 0.1,
            "side": "Buy",
        },
        {
            "timestamp": base_time - timedelta(seconds=0.3),
            "price": 50000.5,
            "volume": 0.2,
            "side": "Buy",
        },
        {
            "timestamp": base_time,
            "price": 50001.0,
            "volume": 0.15,
            "side": "Sell",
        },
    ])
    
    window_3s = pd.DataFrame([
        {
            "timestamp": base_time - timedelta(seconds=2.5),
            "price": 49999.0,
            "volume": 0.5,
            "side": "Buy",
        },
        {
            "timestamp": base_time - timedelta(seconds=1.0),
            "price": 50000.0,
            "volume": 0.3,
            "side": "Buy",
        },
        {
            "timestamp": base_time,
            "price": 50001.0,
            "volume": 0.15,
            "side": "Sell",
        },
    ])
    
    window_15s = pd.DataFrame([
        {
            "timestamp": base_time - timedelta(seconds=12),
            "price": 49998.0,
            "volume": 1.0,
            "side": "Buy",
        },
        {
            "timestamp": base_time - timedelta(seconds=5),
            "price": 50000.0,
            "volume": 0.8,
            "side": "Buy",
        },
        {
            "timestamp": base_time,
            "price": 50001.0,
            "volume": 0.15,
            "side": "Sell",
        },
    ])
    
    window_1m = pd.DataFrame([
        {
            "timestamp": base_time - timedelta(seconds=50),
            "price": 49995.0,
            "volume": 5.0,
            "side": "Buy",
        },
        {
            "timestamp": base_time - timedelta(seconds=30),
            "price": 49998.0,
            "volume": 3.0,
            "side": "Buy",
        },
        {
            "timestamp": base_time,
            "price": 50001.0,
            "volume": 0.15,
            "side": "Sell",
        },
    ])
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1s": window_1s,
            "3s": window_3s,
            "15s": window_15s,
            "1m": window_1m,
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_empty():
    """Empty rolling windows structure."""
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "3s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "15s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "1m": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
        },
        "last_update": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_rolling_windows_trades():
    """Rolling windows populated with trade data."""
    base_time = datetime.now(timezone.utc)
    
    trades_data = [
        {
            "timestamp": base_time - timedelta(seconds=2),
            "price": 50000.0,
            "volume": 0.1,
            "side": "Buy",
        },
        {
            "timestamp": base_time - timedelta(seconds=1),
            "price": 50000.5,
            "volume": 0.2,
            "side": "Buy",
        },
        {
            "timestamp": base_time,
            "price": 50001.0,
            "volume": 0.15,
            "side": "Sell",
        },
    ]
    
    df = pd.DataFrame(trades_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1s": df[df["timestamp"] >= base_time - timedelta(seconds=1)],
            "3s": df[df["timestamp"] >= base_time - timedelta(seconds=3)],
            "15s": df[df["timestamp"] >= base_time - timedelta(seconds=15)],
            "1m": df[df["timestamp"] >= base_time - timedelta(minutes=1)],
        },
        "last_update": base_time,
    }


@pytest.fixture
def sample_rolling_windows_klines():
    """Rolling windows populated with kline/candlestick data."""
    base_time = datetime.now(timezone.utc)
    
    klines_data = [
        {
            "timestamp": base_time - timedelta(minutes=1),
            "open": 49995.0,
            "high": 50005.0,
            "low": 49990.0,
            "close": 50000.0,
            "volume": 10.0,
        },
        {
            "timestamp": base_time,
            "open": 50000.0,
            "high": 50010.0,
            "low": 49995.0,
            "close": 50005.0,
            "volume": 12.0,
        },
    ]
    
    df = pd.DataFrame(klines_data)
    
    return {
        "symbol": "BTCUSDT",
        "windows": {
            "1s": df[df["timestamp"] >= base_time - timedelta(seconds=1)],
            "3s": df[df["timestamp"] >= base_time - timedelta(seconds=3)],
            "15s": df[df["timestamp"] >= base_time - timedelta(seconds=15)],
            "1m": df[df["timestamp"] >= base_time - timedelta(minutes=1)],
        },
        "last_update": base_time,
    }

