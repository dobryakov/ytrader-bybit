"""
Test fixtures for historical market data (Parquet format).
"""
import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import os


@pytest.fixture
def sample_historical_orderbook_snapshots():
    """Sample historical orderbook snapshots as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "sequence": 1000 + i,
            "bids": [[50000.0 - i, 1.5], [49999.0 - i, 2.0], [49998.0 - i, 1.0]],
            "asks": [[50001.0 + i, 1.2], [50002.0 + i, 2.5], [50003.0 + i, 1.8]],
            "internal_timestamp": timestamp,
            "exchange_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_orderbook_deltas():
    """Sample historical orderbook deltas as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(200):
        timestamp = base_time + timedelta(milliseconds=i * 100)
        delta_type = ["insert", "update", "delete"][i % 3]
        side = ["bid", "ask"][i % 2]
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "sequence": 1001 + i,
            "delta_type": delta_type,
            "side": side,
            "price": 50000.0 + (i * 0.1),
            "quantity": 1.0 + (i * 0.01),
            "internal_timestamp": timestamp,
            "exchange_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_trades():
    """Sample historical trades as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(500):
        timestamp = base_time + timedelta(milliseconds=i * 50)
        side = ["Buy", "Sell"][i % 2]
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "price": 50000.0 + (i * 0.05),
            "quantity": 0.1 + (i * 0.001),
            "side": side,
            "trade_time": timestamp,
            "internal_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_klines():
    """Sample historical klines as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(1440):  # 24 hours of 1-minute klines
        timestamp = base_time + timedelta(minutes=i)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "interval": "1m",
            "open": 50000.0 + (i * 0.1),
            "high": 50010.0 + (i * 0.1),
            "low": 49990.0 + (i * 0.1),
            "close": 50005.0 + (i * 0.1),
            "volume": 10.0 + (i * 0.01),
            "internal_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_ticker():
    """Sample historical ticker data as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i * 10)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "last_price": 50000.0 + (i * 0.1),
            "bid_price": 50000.0 + (i * 0.1) - 0.5,
            "ask_price": 50000.0 + (i * 0.1) + 0.5,
            "volume_24h": 1000.0 + (i * 10),
            "internal_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_historical_funding():
    """Sample historical funding rate data as DataFrame."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(8):  # 8 funding periods per day (every 8 hours)
        timestamp = base_time + timedelta(hours=i * 8)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "funding_rate": 0.0001 + (i * 0.00001),
            "next_funding_time": (timestamp + timedelta(hours=8)).timestamp() * 1000,
            "internal_timestamp": timestamp,
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_parquet_file_orderbook(tmp_path):
    """Create a temporary Parquet file with orderbook snapshots."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "sequence": 1000 + i,
            "bids": [[50000.0 - i, 1.5], [49999.0 - i, 2.0]],
            "asks": [[50001.0 + i, 1.2], [50002.0 + i, 2.5]],
            "internal_timestamp": timestamp,
            "exchange_timestamp": timestamp,
        })
    
    df = pd.DataFrame(data)
    file_path = tmp_path / "orderbook_snapshots.parquet"
    
    # Convert to Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    
    return str(file_path)


@pytest.fixture
def sample_parquet_file_trades(tmp_path):
    """Create a temporary Parquet file with trades."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    
    data = []
    for i in range(500):
        timestamp = base_time + timedelta(milliseconds=i * 50)
        data.append({
            "timestamp": timestamp,
            "symbol": "BTCUSDT",
            "price": 50000.0 + (i * 0.05),
            "quantity": 0.1 + (i * 0.001),
            "side": ["Buy", "Sell"][i % 2],
            "trade_time": timestamp,
            "internal_timestamp": timestamp,
        })
    
    df = pd.DataFrame(data)
    file_path = tmp_path / "trades.parquet"
    
    # Convert to Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    
    return str(file_path)


@pytest.fixture
def sample_parquet_directory_structure(tmp_path):
    """Create a temporary directory structure mimicking Parquet storage layout."""
    base_time = datetime.now(timezone.utc) - timedelta(days=1)
    date_str = base_time.strftime("%Y-%m-%d")
    
    # Create directory structure
    orderbook_dir = tmp_path / "orderbook" / "snapshots" / date_str
    orderbook_dir.mkdir(parents=True)
    
    deltas_dir = tmp_path / "orderbook" / "deltas" / date_str
    deltas_dir.mkdir(parents=True)
    
    trades_dir = tmp_path / "trades" / date_str
    trades_dir.mkdir(parents=True)
    
    klines_dir = tmp_path / "klines" / date_str
    klines_dir.mkdir(parents=True)
    
    ticker_dir = tmp_path / "ticker" / date_str
    ticker_dir.mkdir(parents=True)
    
    funding_dir = tmp_path / "funding" / date_str
    funding_dir.mkdir(parents=True)
    
    # Create sample Parquet files
    symbol = "BTCUSDT"
    
    # Orderbook snapshots
    snapshot_data = []
    for i in range(100):
        timestamp = base_time + timedelta(seconds=i)
        snapshot_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "sequence": 1000 + i,
            "bids": [[50000.0 - i, 1.5], [49999.0 - i, 2.0]],
            "asks": [[50001.0 + i, 1.2], [50002.0 + i, 2.5]],
            "internal_timestamp": timestamp,
            "exchange_timestamp": timestamp,
        })
    
    df_snapshots = pd.DataFrame(snapshot_data)
    table_snapshots = pa.Table.from_pandas(df_snapshots)
    pq.write_table(table_snapshots, orderbook_dir / f"{symbol}.parquet")
    
    # Orderbook deltas
    delta_data = []
    for i in range(200):
        timestamp = base_time + timedelta(milliseconds=i * 100)
        delta_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "sequence": 1001 + i,
            "delta_type": ["insert", "update", "delete"][i % 3],
            "side": ["bid", "ask"][i % 2],
            "price": 50000.0 + (i * 0.1),
            "quantity": 1.0 + (i * 0.01),
            "internal_timestamp": timestamp,
            "exchange_timestamp": timestamp,
        })
    
    df_deltas = pd.DataFrame(delta_data)
    table_deltas = pa.Table.from_pandas(df_deltas)
    pq.write_table(table_deltas, deltas_dir / f"{symbol}.parquet")
    
    # Trades
    trade_data = []
    for i in range(500):
        timestamp = base_time + timedelta(milliseconds=i * 50)
        trade_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "price": 50000.0 + (i * 0.05),
            "quantity": 0.1 + (i * 0.001),
            "side": ["Buy", "Sell"][i % 2],
            "trade_time": timestamp,
            "internal_timestamp": timestamp,
        })
    
    df_trades = pd.DataFrame(trade_data)
    table_trades = pa.Table.from_pandas(df_trades)
    pq.write_table(table_trades, trades_dir / f"{symbol}.parquet")
    
    # Klines
    kline_data = []
    for i in range(1440):
        timestamp = base_time + timedelta(minutes=i)
        kline_data.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "interval": "1m",
            "open": 50000.0 + (i * 0.1),
            "high": 50010.0 + (i * 0.1),
            "low": 49990.0 + (i * 0.1),
            "close": 50005.0 + (i * 0.1),
            "volume": 10.0 + (i * 0.01),
            "internal_timestamp": timestamp,
        })
    
    df_klines = pd.DataFrame(kline_data)
    table_klines = pa.Table.from_pandas(df_klines)
    pq.write_table(table_klines, klines_dir / f"{symbol}.parquet")
    
    return str(tmp_path)
