"""
Price features computation module.
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows


def _ensure_datetime(value) -> datetime:
    """Ensure value is a datetime object."""
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        from dateutil.parser import parse
        return parse(value)
    else:
        return datetime.now(timezone.utc)


def compute_mid_price(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute mid price from orderbook."""
    if orderbook is None:
        return None
    
    return orderbook.get_mid_price()


def compute_spread_abs(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute absolute spread (ask - bid)."""
    if orderbook is None:
        return None
    
    return orderbook.get_spread_abs()


def compute_spread_rel(orderbook: Optional[OrderbookState]) -> Optional[float]:
    """Compute relative spread (spread / mid_price)."""
    if orderbook is None:
        return None
    
    return orderbook.get_spread_rel()


def compute_returns(
    rolling_windows: RollingWindows,
    window_seconds: int,
    current_price: Optional[float],
) -> Optional[float]:
    """Compute returns over specified window."""
    if current_price is None:
        return None
    
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get price data from window
    window_data = rolling_windows.get_window_data(f"{window_seconds}s")
    if len(window_data) == 0 or "price" not in window_data.columns:
        return None
    
    # Get first price in window
    window_data_sorted = window_data.sort_values("timestamp")
    if len(window_data_sorted) == 0:
        return None
    
    # Ensure price is numeric (convert from string if needed)
    first_price = pd.to_numeric(window_data_sorted.iloc[0]["price"], errors='coerce')
    if pd.isna(first_price) or first_price == 0:
        return None
    
    # Compute return: (current - first) / first
    return float((current_price - first_price) / first_price)


def compute_vwap(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute Volume-Weighted Average Price (VWAP) over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return None
    
    if "price" not in trades.columns or "volume" not in trades.columns:
        return None
    
    # Ensure numeric types (convert from string if needed)
    prices = pd.to_numeric(trades["price"], errors='coerce').fillna(0.0)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    
    # VWAP = sum(price * volume) / sum(volume)
    total_value = (prices * volumes).sum()
    total_volume = volumes.sum()
    
    if total_volume == 0:
        return None
    
    return total_value / total_volume


def compute_volume(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute total volume over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return 0.0
    
    if "volume" not in trades.columns:
        return 0.0
    
    # Ensure numeric type (convert from string if needed)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    return float(volumes.sum())


def compute_volatility(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute volatility (standard deviation of returns) over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get klines for window
    klines = rolling_windows.get_klines_for_window("1m", start_time, now)
    
    if len(klines) < 2:
        return None
    
    if "close" not in klines.columns:
        return None
    
    # Compute returns from close prices
    # Ensure closes are numeric (convert from string if needed)
    # First, get the close column and convert to numeric
    close_col = klines["close"]
    
    # Convert to numeric, handling any string values
    closes_series = pd.to_numeric(close_col, errors='coerce')
    
    # Fill NaN with 0.0 and convert to float
    closes_series = closes_series.fillna(0.0).astype(float)
    
    # Convert to numpy array with explicit float dtype
    closes = np.array(closes_series.values, dtype=float)
    
    # Additional safety check: ensure we have a proper float array
    if closes.dtype != np.float64:
        closes = closes.astype(float)
    
    if len(closes) < 2:
        return None
    
    # Filter out zero values to avoid division by zero
    closes = closes[closes > 0]
    
    # Ensure dtype is still float after filtering
    if len(closes) > 0 and closes.dtype != np.float64:
        closes = closes.astype(float)
    
    if len(closes) < 2:
        return None
    
    # Compute returns with explicit float conversion
    closes_float = closes.astype(float)
    returns = np.diff(closes_float) / closes_float[:-1]
    
    if len(returns) == 0:
        return None
    
    # Compute standard deviation of returns
    volatility = np.std(returns)
    
    return float(volatility)


def compute_all_price_features(
    orderbook: Optional[OrderbookState],
    rolling_windows: RollingWindows,
    current_price: Optional[float],
) -> Dict[str, Optional[float]]:
    """Compute all price features."""
    features = {}
    
    # Basic price features from orderbook
    features["mid_price"] = compute_mid_price(orderbook)
    features["spread_abs"] = compute_spread_abs(orderbook)
    features["spread_rel"] = compute_spread_rel(orderbook)
    
    # Returns
    features["returns_1s"] = compute_returns(rolling_windows, 1, current_price)
    features["returns_3s"] = compute_returns(rolling_windows, 3, current_price)
    features["returns_1m"] = compute_returns(rolling_windows, 60, current_price)
    
    # VWAP
    features["vwap_3s"] = compute_vwap(rolling_windows, 3)
    features["vwap_15s"] = compute_vwap(rolling_windows, 15)
    features["vwap_1m"] = compute_vwap(rolling_windows, 60)
    
    # Volume
    features["volume_3s"] = compute_volume(rolling_windows, 3)
    features["volume_15s"] = compute_volume(rolling_windows, 15)
    features["volume_1m"] = compute_volume(rolling_windows, 60)
    
    # Volatility
    features["volatility_1m"] = compute_volatility(rolling_windows, 60)
    features["volatility_5m"] = compute_volatility(rolling_windows, 300)
    
    return features

