"""
Technical indicators computation module.
"""
from typing import Dict, Optional
import pandas as pd
import numpy as np
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


def compute_ema(
    rolling_windows: RollingWindows,
    period: int = 21,
    candle_interval: str = "1m",
) -> Optional[float]:
    """
    Compute Exponential Moving Average (EMA) for specified period.
    
    EMA = price * multiplier + EMA_prev * (1 - multiplier)
    where multiplier = 2 / (period + 1)
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        period: EMA period (default: 21)
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        EMA value or None if insufficient price history
    """
    now = _ensure_datetime(rolling_windows.last_update)
    # Get enough history for EMA period (assuming 1m candles, need period minutes)
    start_time = now - timedelta(minutes=period + 5)  # Extra buffer
    
    # Get klines for window
    klines = rolling_windows.get_klines_for_window(candle_interval, start_time, now)
    
    if len(klines) < period:
        return None
    
    if "close" not in klines.columns:
        return None
    
    # Sort by timestamp
    klines_sorted = klines.sort_values("timestamp")
    if len(klines_sorted) < period:
        return None
    
    # Get close prices
    closes = pd.to_numeric(klines_sorted["close"], errors='coerce').fillna(0.0)
    
    # Filter out zero values
    closes = closes[closes > 0]
    
    if len(closes) < period:
        return None
    
    # Check if all prices are equal
    if closes.nunique() == 1:
        # EMA of constant prices equals the price
        return float(closes.iloc[-1])
    
    # Compute EMA
    multiplier = 2.0 / (period + 1)
    
    # Initialize EMA with first price
    ema = float(closes.iloc[0])
    
    # Compute EMA iteratively
    for i in range(1, len(closes)):
        price = float(closes.iloc[i])
        ema = price * multiplier + ema * (1 - multiplier)
    
    return float(ema)


def compute_ema_21(
    rolling_windows: RollingWindows,
    candle_interval: str = "1m",
) -> Optional[float]:
    """
    Compute EMA(21) indicator.
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        EMA(21) value or None if insufficient price history
    """
    return compute_ema(rolling_windows, period=21, candle_interval=candle_interval)


def compute_rsi(
    rolling_windows: RollingWindows,
    period: int = 14,
    candle_interval: str = "1m",
) -> Optional[float]:
    """
    Compute Relative Strength Index (RSI) for specified period.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = average_gain / average_loss
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        period: RSI period (default: 14)
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        RSI value (0-100) or None if insufficient price history
    """
    now = _ensure_datetime(rolling_windows.last_update)
    # Get enough history for RSI period
    start_time = now - timedelta(minutes=period + 5)  # Extra buffer
    
    # Get klines for window
    klines = rolling_windows.get_klines_for_window(candle_interval, start_time, now)
    
    if len(klines) < period + 1:
        return None
    
    if "close" not in klines.columns:
        return None
    
    # Sort by timestamp
    klines_sorted = klines.sort_values("timestamp")
    if len(klines_sorted) < period + 1:
        return None
    
    # Get close prices
    closes = pd.to_numeric(klines_sorted["close"], errors='coerce').fillna(0.0)
    
    # Filter out zero values
    closes = closes[closes > 0]
    
    if len(closes) < period + 1:
        return None
    
    # Check if all prices are equal
    if closes.nunique() == 1:
        # RSI of constant prices is 50 (neutral)
        return 50.0
    
    # Compute price changes
    price_changes = closes.diff().dropna()
    
    if len(price_changes) < period:
        return None
    
    # Separate gains and losses
    gains = price_changes[price_changes > 0]
    losses = -price_changes[price_changes < 0]
    
    # Compute average gain and loss
    avg_gain = gains.tail(period).mean() if len(gains) > 0 else 0.0
    avg_loss = losses.tail(period).mean() if len(losses) > 0 else 0.0
    
    if avg_loss == 0:
        # If no losses, RSI is 100
        return 100.0
    
    # Compute RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return float(rsi)


def compute_rsi_14(
    rolling_windows: RollingWindows,
    candle_interval: str = "1m",
) -> Optional[float]:
    """
    Compute RSI(14) indicator.
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        RSI(14) value (0-100) or None if insufficient price history
    """
    return compute_rsi(rolling_windows, period=14, candle_interval=candle_interval)


def compute_all_technical_indicators(
    rolling_windows: RollingWindows,
    candle_interval: str = "1m",
) -> Dict[str, Optional[float]]:
    """
    Compute all technical indicators.
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        candle_interval: Candle interval (default: "1m")
        
    Returns:
        Dictionary of technical indicator values
    """
    indicators = {}
    
    # EMA(21)
    indicators["ema_21"] = compute_ema_21(rolling_windows, candle_interval)
    
    # RSI(14)
    indicators["rsi_14"] = compute_rsi_14(rolling_windows, candle_interval)
    
    return indicators

