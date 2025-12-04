"""
Orderflow features computation module.
"""
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta, timezone

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


def compute_signed_volume(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> Optional[float]:
    """Compute signed volume (buy volume - sell volume) over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return 0.0
    
    if "volume" not in trades.columns or "side" not in trades.columns:
        return 0.0
    
    # Ensure numeric type (convert from string if needed)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    
    # Compute signed volume: buy volume is positive, sell volume is negative
    buy_volume = volumes[trades["side"] == "Buy"].sum()
    sell_volume = volumes[trades["side"] == "Sell"].sum()
    
    return float(buy_volume - sell_volume)


def compute_buy_sell_volume_ratio(
    rolling_windows: RollingWindows,
    window_seconds: int = 3,
) -> Optional[float]:
    """Compute buy/sell volume ratio over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return 1.0  # Neutral ratio
    
    if "volume" not in trades.columns or "side" not in trades.columns:
        return 1.0
    
    # Ensure numeric type (convert from string if needed)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    
    buy_volume = volumes[trades["side"] == "Buy"].sum()
    sell_volume = volumes[trades["side"] == "Sell"].sum()
    
    if sell_volume == 0:
        return float("inf") if buy_volume > 0 else 1.0
    
    return float(buy_volume / sell_volume)


def compute_trade_count(
    rolling_windows: RollingWindows,
    window_seconds: int,
) -> int:
    """Compute trade count over specified window."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    return len(trades)


def compute_net_aggressor_pressure(
    rolling_windows: RollingWindows,
    window_seconds: int = 3,
) -> Optional[float]:
    """Compute net aggressor pressure (normalized signed volume)."""
    now = _ensure_datetime(rolling_windows.last_update)
    start_time = now - timedelta(seconds=window_seconds)
    
    # Get trades for window
    trades = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, now)
    
    if len(trades) == 0:
        return 0.0
    
    if "volume" not in trades.columns or "side" not in trades.columns:
        return 0.0
    
    # Ensure numeric type (convert from string if needed)
    volumes = pd.to_numeric(trades["volume"], errors='coerce').fillna(0.0)
    
    # Compute signed volume
    buy_volume = volumes[trades["side"] == "Buy"].sum()
    sell_volume = volumes[trades["side"] == "Sell"].sum()
    total_volume = buy_volume + sell_volume
    
    if total_volume == 0:
        return 0.0
    
    # Normalized: (buy - sell) / (buy + sell)
    return float((buy_volume - sell_volume) / total_volume)


def compute_all_orderflow_features(
    rolling_windows: RollingWindows,
) -> Dict[str, Optional[float]]:
    """Compute all orderflow features."""
    features = {}
    
    # Signed volume
    features["signed_volume_3s"] = compute_signed_volume(rolling_windows, 3)
    features["signed_volume_15s"] = compute_signed_volume(rolling_windows, 15)
    features["signed_volume_1m"] = compute_signed_volume(rolling_windows, 60)
    
    # Buy/sell ratio
    features["buy_sell_volume_ratio"] = compute_buy_sell_volume_ratio(rolling_windows, 3)
    
    # Trade count
    features["trade_count_3s"] = compute_trade_count(rolling_windows, 3)
    
    # Net aggressor pressure
    features["net_aggressor_pressure"] = compute_net_aggressor_pressure(rolling_windows, 3)
    
    return features

