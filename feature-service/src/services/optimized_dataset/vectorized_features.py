"""
Vectorized feature computation for optimized dataset building.

Computes features for multiple timestamps simultaneously using pandas
vectorized operations for improved performance.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def compute_technical_indicators_vectorized(
    klines_df: pd.DataFrame,
    timestamps: pd.Series,
    period_ema: int = 21,
    period_rsi: int = 14,
) -> pd.DataFrame:
    """
    Compute technical indicators (EMA, RSI) for all timestamps vectorized.
    
    Args:
        klines_df: DataFrame with klines (columns: timestamp, open, high, low, close, volume)
        timestamps: Series of timestamps to compute features for
        period_ema: EMA period (default: 21)
        period_rsi: RSI period (default: 14)
        
    Returns:
        DataFrame with columns: [timestamp, ema_21, rsi_14]
    """
    if klines_df.empty or timestamps.empty:
        return pd.DataFrame(columns=["timestamp", "ema_21", "rsi_14"])
    
    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, utc=True)
    
    # Ensure klines timestamp is datetime
    if "timestamp" in klines_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(klines_df["timestamp"]):
            klines_df = klines_df.copy()
            klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], utc=True)
    
    # Sort klines by timestamp
    klines_sorted = klines_df.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure close column is numeric
    if "close" not in klines_sorted.columns:
        return pd.DataFrame(columns=["timestamp", "ema_21", "rsi_14"])
    
    closes = pd.to_numeric(klines_sorted["close"], errors='coerce').fillna(0.0)
    klines_sorted = klines_sorted.copy()
    klines_sorted["close"] = closes
    
    # Filter out zero prices
    klines_sorted = klines_sorted[klines_sorted["close"] > 0].reset_index(drop=True)
    
    if len(klines_sorted) < max(period_ema, period_rsi + 1):
        # Not enough data
        return pd.DataFrame({
            "timestamp": timestamps,
            "ema_21": [None] * len(timestamps),
            "rsi_14": [None] * len(timestamps),
        })
    
    # Prepare result DataFrame
    result = pd.DataFrame({"timestamp": timestamps})
    result["ema_21"] = None
    result["rsi_14"] = None
    
    # Compute EMA for each timestamp
    for idx, ts in enumerate(timestamps):
        # Get klines up to this timestamp
        klines_before = klines_sorted[klines_sorted["timestamp"] <= ts]
        
        if len(klines_before) < period_ema:
            continue
        
        # Get closes for EMA calculation
        closes_for_ema = klines_before["close"].tail(period_ema)
        
        # Compute EMA using pandas ewm
        ema = closes_for_ema.ewm(span=period_ema, adjust=False).mean().iloc[-1]
        result.at[idx, "ema_21"] = float(ema)
        
        # Compute RSI if enough data
        if len(klines_before) >= period_rsi + 1:
            closes_for_rsi = klines_before["close"].tail(period_rsi + 1)
            
            # Compute price changes
            price_changes = closes_for_rsi.diff().dropna()
            
            if len(price_changes) >= period_rsi:
                # Separate gains and losses
                gains = price_changes[price_changes > 0]
                losses = -price_changes[price_changes < 0]
                
                # Average gain and loss
                avg_gain = gains.tail(period_rsi).mean() if len(gains) > 0 else 0.0
                avg_loss = losses.tail(period_rsi).mean() if len(losses) > 0 else 0.0
                
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                result.at[idx, "rsi_14"] = float(rsi)
    
    return result


def compute_orderflow_features_vectorized(
    trades_df: pd.DataFrame,
    timestamps: pd.Series,
    windows: List[int] = [1, 3, 15, 60],
) -> pd.DataFrame:
    """
    Compute orderflow features for all timestamps vectorized.
    
    Args:
        trades_df: DataFrame with trades (columns: timestamp, price, volume, side)
        timestamps: Series of timestamps to compute features for
        windows: List of window sizes in seconds (default: [1, 3, 15, 60])
        
    Returns:
        DataFrame with columns: [timestamp, signed_volume_1s, signed_volume_3s, ...]
    """
    if trades_df.empty or timestamps.empty:
        # Return empty DataFrame with expected columns
        columns = ["timestamp"] + [
            f"signed_volume_{w}s" for w in windows
        ] + ["buy_sell_volume_ratio", "trade_count_3s", "net_aggressor_pressure"]
        return pd.DataFrame(columns=columns)
    
    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, utc=True)
    
    # Ensure trades timestamp is datetime
    if "timestamp" in trades_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(trades_df["timestamp"]):
            trades_df = trades_df.copy()
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True)
    
    # Ensure numeric columns
    if "volume" in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df["volume"] = pd.to_numeric(trades_df["volume"], errors='coerce').fillna(0.0)
    
    # Sort trades by timestamp
    trades_sorted = trades_df.sort_values("timestamp").reset_index(drop=True)
    
    # Prepare result DataFrame
    result = pd.DataFrame({"timestamp": timestamps})
    
    # Initialize columns
    for window in windows:
        result[f"signed_volume_{window}s"] = None
    result["buy_sell_volume_ratio"] = None
    result["trade_count_3s"] = None
    result["net_aggressor_pressure"] = None
    
    # Compute features for each timestamp
    for idx, ts in enumerate(timestamps):
        # Get trades up to this timestamp
        trades_before = trades_sorted[trades_sorted["timestamp"] <= ts]
        
        if trades_before.empty:
            continue
        
        # Compute signed volume for each window
        for window in windows:
            window_start = ts - timedelta(seconds=window)
            window_trades = trades_before[trades_before["timestamp"] > window_start]
            
            if not window_trades.empty and "volume" in window_trades.columns and "side" in window_trades.columns:
                volumes = window_trades["volume"]
                buy_volume = volumes[window_trades["side"] == "Buy"].sum()
                sell_volume = volumes[window_trades["side"] == "Sell"].sum()
                signed_volume = float(buy_volume - sell_volume)
                result.at[idx, f"signed_volume_{window}s"] = signed_volume
        
        # Compute buy/sell ratio for 3s window
        window_start_3s = ts - timedelta(seconds=3)
        window_trades_3s = trades_before[trades_before["timestamp"] > window_start_3s]
        
        if not window_trades_3s.empty and "volume" in window_trades_3s.columns and "side" in window_trades_3s.columns:
            volumes_3s = window_trades_3s["volume"]
            buy_volume_3s = volumes_3s[window_trades_3s["side"] == "Buy"].sum()
            sell_volume_3s = volumes_3s[window_trades_3s["side"] == "Sell"].sum()
            
            # Buy/sell ratio
            if sell_volume_3s > 0:
                ratio = float(buy_volume_3s / sell_volume_3s)
            else:
                ratio = float("inf") if buy_volume_3s > 0 else 1.0
            result.at[idx, "buy_sell_volume_ratio"] = ratio
            
            # Trade count
            result.at[idx, "trade_count_3s"] = len(window_trades_3s)
            
            # Net aggressor pressure
            total_volume = buy_volume_3s + sell_volume_3s
            if total_volume > 0:
                pressure = float((buy_volume_3s - sell_volume_3s) / total_volume)
                result.at[idx, "net_aggressor_pressure"] = pressure
    
    return result


def compute_price_features_vectorized(
    klines_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    timestamps: pd.Series,
    current_prices: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute price features for all timestamps vectorized.
    
    Args:
        klines_df: DataFrame with klines (columns: timestamp, open, high, low, close, volume)
        trades_df: DataFrame with trades (columns: timestamp, price, volume, side)
        timestamps: Series of timestamps to compute features for
        current_prices: Optional Series of current prices for each timestamp
        
    Returns:
        DataFrame with columns: [timestamp, returns_1s, returns_3s, vwap_3s, ...]
    """
    if timestamps.empty:
        return pd.DataFrame(columns=["timestamp"])
    
    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, utc=True)
    
    # Prepare result DataFrame
    result = pd.DataFrame({"timestamp": timestamps})
    
    # Initialize columns
    price_feature_columns = [
        "returns_1s", "returns_3s", "returns_1m", "returns_3m", "returns_5m",
        "vwap_3s", "vwap_15s", "vwap_1m", "vwap_3m", "vwap_5m",
        "volume_3s", "volume_15s", "volume_1m", "volume_3m", "volume_5m",
        "volatility_1m", "volatility_5m", "volatility_10m", "volatility_15m",
    ]
    
    for col in price_feature_columns:
        result[col] = None
    
    # Get current prices from klines if not provided
    if current_prices is None:
        if not klines_df.empty and "timestamp" in klines_df.columns and "close" in klines_df.columns:
            # For each timestamp, get latest close price
            current_prices = pd.Series([None] * len(timestamps))
            for idx, ts in enumerate(timestamps):
                klines_before = klines_df[klines_df["timestamp"] <= ts]
                if not klines_before.empty:
                    latest_close = klines_before.iloc[-1]["close"]
                    current_prices.iloc[idx] = pd.to_numeric(latest_close, errors='coerce')
    
    # Process klines
    if not klines_df.empty and "timestamp" in klines_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(klines_df["timestamp"]):
            klines_df = klines_df.copy()
            klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], utc=True)
        
        klines_sorted = klines_df.sort_values("timestamp").reset_index(drop=True)
        
        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in klines_sorted.columns:
                klines_sorted[col] = pd.to_numeric(klines_sorted[col], errors='coerce').fillna(0.0)
        
        # Compute returns and volatility for each timestamp
        for idx, ts in enumerate(timestamps):
            current_price = current_prices.iloc[idx] if current_prices is not None else None
            
            if pd.isna(current_price) or current_price == 0:
                continue
            
            klines_before = klines_sorted[klines_sorted["timestamp"] <= ts]
            
            if klines_before.empty:
                continue
            
            # Compute returns for different windows
            for window_seconds in [1, 3, 60, 180, 300]:
                window_start = ts - timedelta(seconds=window_seconds)
                window_klines = klines_before[klines_before["timestamp"] > window_start]
                
                if not window_klines.empty and "close" in window_klines.columns:
                    first_close = pd.to_numeric(window_klines.iloc[0]["close"], errors='coerce')
                    if not pd.isna(first_close) and first_close > 0:
                        returns = float((current_price - first_close) / first_close)
                        
                        if window_seconds == 1:
                            result.at[idx, "returns_1s"] = returns
                        elif window_seconds == 3:
                            result.at[idx, "returns_3s"] = returns
                        elif window_seconds == 60:
                            result.at[idx, "returns_1m"] = returns
                        elif window_seconds == 180:
                            result.at[idx, "returns_3m"] = returns
                        elif window_seconds == 300:
                            result.at[idx, "returns_5m"] = returns
            
            # Compute volatility for different windows
            for window_minutes in [1, 5, 10, 15]:
                window_start = ts - timedelta(minutes=window_minutes)
                window_klines = klines_before[klines_before["timestamp"] > window_start]
                
                if len(window_klines) >= 2 and "close" in window_klines.columns:
                    closes = pd.to_numeric(window_klines["close"], errors='coerce').fillna(0.0)
                    closes = closes[closes > 0]
                    
                    if len(closes) >= 2:
                        returns = closes.pct_change().dropna()
                        if len(returns) > 0:
                            volatility = float(returns.std())
                            
                            if window_minutes == 1:
                                result.at[idx, "volatility_1m"] = volatility
                            elif window_minutes == 5:
                                result.at[idx, "volatility_5m"] = volatility
                            elif window_minutes == 10:
                                result.at[idx, "volatility_10m"] = volatility
                            elif window_minutes == 15:
                                result.at[idx, "volatility_15m"] = volatility
    
    # Process trades for VWAP and volume
    if not trades_df.empty and "timestamp" in trades_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(trades_df["timestamp"]):
            trades_df = trades_df.copy()
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True)
        
        trades_sorted = trades_df.sort_values("timestamp").reset_index(drop=True)
        
        # Ensure numeric columns
        if "price" in trades_sorted.columns:
            trades_sorted["price"] = pd.to_numeric(trades_sorted["price"], errors='coerce').fillna(0.0)
        if "volume" in trades_sorted.columns:
            trades_sorted["volume"] = pd.to_numeric(trades_sorted["volume"], errors='coerce').fillna(0.0)
        
        # Compute VWAP and volume for each timestamp
        for idx, ts in enumerate(timestamps):
            trades_before = trades_sorted[trades_sorted["timestamp"] <= ts]
            
            if trades_before.empty:
                continue
            
            # Compute VWAP and volume for different windows
            for window_seconds in [3, 15, 60, 180, 300]:
                window_start = ts - timedelta(seconds=window_seconds)
                window_trades = trades_before[trades_before["timestamp"] > window_start]
                
                if not window_trades.empty:
                    if "price" in window_trades.columns and "volume" in window_trades.columns:
                        prices = window_trades["price"]
                        volumes = window_trades["volume"]
                        
                        total_value = (prices * volumes).sum()
                        total_volume = volumes.sum()
                        
                        if total_volume > 0:
                            vwap = float(total_value / total_volume)
                            
                            if window_seconds == 3:
                                result.at[idx, "vwap_3s"] = vwap
                                result.at[idx, "volume_3s"] = float(total_volume)
                            elif window_seconds == 15:
                                result.at[idx, "vwap_15s"] = vwap
                                result.at[idx, "volume_15s"] = float(total_volume)
                            elif window_seconds == 60:
                                result.at[idx, "vwap_1m"] = vwap
                                result.at[idx, "volume_1m"] = float(total_volume)
                            elif window_seconds == 180:
                                result.at[idx, "vwap_3m"] = vwap
                                result.at[idx, "volume_3m"] = float(total_volume)
                            elif window_seconds == 300:
                                result.at[idx, "vwap_5m"] = vwap
                                result.at[idx, "volume_5m"] = float(total_volume)
    
    return result

