"""
Optimized Rolling Window for dataset building.

Implements a fixed-size rolling window buffer with efficient incremental updates.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import pandas as pd
import structlog

from src.models.rolling_windows import RollingWindows

logger = structlog.get_logger(__name__)


class OptimizedRollingWindow:
    """
    Optimized rolling window with fixed-size buffer.
    
    Maintains a rolling window of historical data with automatic trimming
    of old data outside the lookback window. Optimized for incremental updates
    during dataset building.
    """
    
    def __init__(self, max_lookback_minutes: int, symbol: str):
        """
        Initialize optimized rolling window.
        
        Args:
            max_lookback_minutes: Maximum lookback period in minutes
            symbol: Trading pair symbol
        """
        self.max_lookback = timedelta(minutes=max_lookback_minutes + 5)  # 5 min buffer
        self.symbol = symbol
        
        # Initialize empty buffers
        self.trades_buffer: pd.DataFrame = pd.DataFrame(
            columns=["timestamp", "price", "volume", "side"]
        )
        self.klines_buffer: pd.DataFrame = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        self._last_timestamp: Optional[datetime] = None
        self._buffer_start: Optional[datetime] = None
        
        logger.debug(
            "optimized_rolling_window_initialized",
            symbol=symbol,
            max_lookback_minutes=max_lookback_minutes,
            max_lookback_timedelta=str(self.max_lookback),
        )
    
    def add_data(
        self,
        timestamp: datetime,
        trades: Optional[pd.DataFrame] = None,
        klines: Optional[pd.DataFrame] = None,
        skip_trim: bool = False,
    ) -> None:
        """
        Add new data to rolling window and automatically trim old data.
        
        Args:
            timestamp: Current timestamp
            trades: New trades DataFrame (optional)
            klines: New klines DataFrame (optional)
            skip_trim: If True, skip trimming old data (useful for historical data loading)
        """
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Add trades if provided
        if trades is not None and not trades.empty:
            # Ensure timestamp column is datetime
            if "timestamp" in trades.columns:
                trades = trades.copy()
                trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
                
                # Ensure numeric columns
                numeric_cols = ["price", "volume"]
                for col in numeric_cols:
                    if col in trades.columns:
                        trades[col] = pd.to_numeric(trades[col], errors='coerce').fillna(0.0)
                
                # Concatenate with existing buffer
                if self.trades_buffer.empty:
                    self.trades_buffer = trades.copy()
                else:
                    self.trades_buffer = pd.concat(
                        [self.trades_buffer, trades], ignore_index=True
                    )
                    # Remove duplicates by timestamp (keep last)
                    self.trades_buffer = self.trades_buffer.drop_duplicates(
                        subset=["timestamp"], keep="last"
                    )
                    # Sort by timestamp
                    self.trades_buffer = self.trades_buffer.sort_values("timestamp").reset_index(drop=True)
        
        # Add klines if provided
        if klines is not None and not klines.empty:
            # Ensure timestamp column is datetime
            if "timestamp" in klines.columns:
                klines = klines.copy()
                klines["timestamp"] = pd.to_datetime(klines["timestamp"], utc=True)
                
                # Ensure numeric columns
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in klines.columns:
                        klines[col] = pd.to_numeric(klines[col], errors='coerce').fillna(0.0)
                
                # Concatenate with existing buffer
                if self.klines_buffer.empty:
                    self.klines_buffer = klines.copy()
                else:
                    self.klines_buffer = pd.concat(
                        [self.klines_buffer, klines], ignore_index=True
                    )
                    # Remove duplicates by timestamp (keep last)
                    self.klines_buffer = self.klines_buffer.drop_duplicates(
                        subset=["timestamp"], keep="last"
                    )
                    # Sort by timestamp
                    self.klines_buffer = self.klines_buffer.sort_values("timestamp").reset_index(drop=True)
        
        # Update timestamps
        self._last_timestamp = timestamp
        
        # Trim old data (skip for historical data loading)
        if not skip_trim:
            self.trim_old_data(timestamp)
    
    def trim_old_data(self, current_timestamp: datetime) -> None:
        """
        Remove data older than max_lookback from current_timestamp.
        
        Args:
            current_timestamp: Current timestamp to calculate cutoff from
        """
        # Ensure timestamp is timezone-aware
        if current_timestamp.tzinfo is None:
            current_timestamp = current_timestamp.replace(tzinfo=timezone.utc)
        
        cutoff_time = current_timestamp - self.max_lookback
        
        # Trim trades buffer
        if not self.trades_buffer.empty and "timestamp" in self.trades_buffer.columns:
            self.trades_buffer = self.trades_buffer[
                self.trades_buffer["timestamp"] >= cutoff_time
            ].copy()
            self.trades_buffer = self.trades_buffer.reset_index(drop=True)
        
        # Trim klines buffer
        if not self.klines_buffer.empty and "timestamp" in self.klines_buffer.columns:
            self.klines_buffer = self.klines_buffer[
                self.klines_buffer["timestamp"] >= cutoff_time
            ].copy()
            self.klines_buffer = self.klines_buffer.reset_index(drop=True)
        
        # Update buffer start
        if self.trades_buffer.empty and self.klines_buffer.empty:
            self._buffer_start = None
        else:
            # Find earliest timestamp
            earliest = None
            if not self.trades_buffer.empty:
                earliest = self.trades_buffer["timestamp"].min()
            if not self.klines_buffer.empty:
                kline_earliest = self.klines_buffer["timestamp"].min()
                if earliest is None or kline_earliest < earliest:
                    earliest = kline_earliest
            self._buffer_start = earliest
    
    def get_window(self, end_timestamp: datetime) -> RollingWindows:
        """
        Get RollingWindows for specific timestamp.
        
        Filters data within max_lookback from end_timestamp and returns
        in RollingWindows format for compatibility with existing feature computation.
        
        Args:
            end_timestamp: End timestamp for window
            
        Returns:
            RollingWindows instance
        """
        # Ensure timestamp is timezone-aware
        if end_timestamp.tzinfo is None:
            end_timestamp = end_timestamp.replace(tzinfo=timezone.utc)
        
        cutoff_time = end_timestamp - self.max_lookback
        
        # Filter trades within window
        trades_window = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        if not self.trades_buffer.empty and "timestamp" in self.trades_buffer.columns:
            trades_window = self.trades_buffer[
                (self.trades_buffer["timestamp"] >= cutoff_time) &
                (self.trades_buffer["timestamp"] <= end_timestamp)
            ].copy()
        
        # Filter klines within window
        klines_window = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if not self.klines_buffer.empty and "timestamp" in self.klines_buffer.columns:
            klines_window = self.klines_buffer[
                (self.klines_buffer["timestamp"] >= cutoff_time) &
                (self.klines_buffer["timestamp"] <= end_timestamp)
            ].copy()
        
        # Create window structure for RollingWindows compatibility
        windows = {
            "1s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "3s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "15s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "1m": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
        }
        
        # Populate windows from trades
        if not trades_window.empty:
            # Filter trades by window sizes
            for window_name, window_seconds in [("1s", 1), ("3s", 3), ("15s", 15), ("1m", 60)]:
                window_start = end_timestamp - timedelta(seconds=window_seconds)
                window_trades = trades_window[
                    trades_window["timestamp"] > window_start
                ].copy()
                if not window_trades.empty:
                    windows[window_name] = window_trades
        
        # Populate 1m window with klines (if available)
        if not klines_window.empty:
            windows["1m"] = klines_window
        
        return RollingWindows(
            symbol=self.symbol,
            windows=windows,
            last_update=end_timestamp,
        )
    
    def get_buffer_stats(self) -> Dict[str, any]:
        """
        Get statistics about current buffer state.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            "trades_count": len(self.trades_buffer),
            "klines_count": len(self.klines_buffer),
            "last_timestamp": self._last_timestamp.isoformat() if self._last_timestamp else None,
            "buffer_start": self._buffer_start.isoformat() if self._buffer_start else None,
            "max_lookback_minutes": int(self.max_lookback.total_seconds() / 60),
        }
        
        if not self.trades_buffer.empty and "timestamp" in self.trades_buffer.columns:
            stats["trades_timestamp_range"] = {
                "min": self.trades_buffer["timestamp"].min().isoformat(),
                "max": self.trades_buffer["timestamp"].max().isoformat(),
            }
        
        if not self.klines_buffer.empty and "timestamp" in self.klines_buffer.columns:
            stats["klines_timestamp_range"] = {
                "min": self.klines_buffer["timestamp"].min().isoformat(),
                "max": self.klines_buffer["timestamp"].max().isoformat(),
            }
        
        return stats
    
    def clear(self) -> None:
        """Clear all buffers."""
        self.trades_buffer = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        self.klines_buffer = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        self._last_timestamp = None
        self._buffer_start = None

