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
        
        # Log diagnostic information for debugging
        import structlog
        logger = structlog.get_logger(__name__)
        logger.debug(
            "optimized_rolling_window_get_window",
            end_timestamp=end_timestamp.isoformat(),
            cutoff_time=cutoff_time.isoformat(),
            max_lookback_minutes=self.max_lookback.total_seconds() / 60,
            klines_buffer_size=len(self.klines_buffer),
            trades_buffer_size=len(self.trades_buffer),
            buffer_start=self._buffer_start.isoformat() if self._buffer_start else None,
            buffer_last=self._last_timestamp.isoformat() if self._last_timestamp else None,
        )
        
        # Filter trades within window
        trades_window = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        if not self.trades_buffer.empty and "timestamp" in self.trades_buffer.columns:
            trades_window = self.trades_buffer[
                (self.trades_buffer["timestamp"] >= cutoff_time) &
                (self.trades_buffer["timestamp"] <= end_timestamp)
            ].copy()
            logger.debug(
                "optimized_rolling_window_trades_filtered",
                end_timestamp=end_timestamp.isoformat(),
                cutoff_time=cutoff_time.isoformat(),
                trades_before_filter=len(self.trades_buffer),
                trades_after_filter=len(trades_window),
            )
        
        # Filter klines within window
        klines_window = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if not self.klines_buffer.empty and "timestamp" in self.klines_buffer.columns:
            klines_before_count = len(self.klines_buffer)
            klines_window = self.klines_buffer[
                (self.klines_buffer["timestamp"] >= cutoff_time) &
                (self.klines_buffer["timestamp"] <= end_timestamp)
            ].copy()
            klines_after_count = len(klines_window)
            
            # Log if we have significantly fewer klines than expected
            if klines_after_count < 15 and klines_before_count > 0:
                first_kline_ts = self.klines_buffer.iloc[0]["timestamp"].isoformat() if len(self.klines_buffer) > 0 else None
                last_kline_ts = self.klines_buffer.iloc[-1]["timestamp"].isoformat() if len(self.klines_buffer) > 0 else None
                
                # Calculate time span in window
                window_time_span_minutes = 0.0
                if not klines_window.empty and "timestamp" in klines_window.columns:
                    window_first = klines_window["timestamp"].min()
                    window_last = klines_window["timestamp"].max()
                    if pd.notna(window_first) and pd.notna(window_last):
                        window_time_span_minutes = (window_last - window_first).total_seconds() / 60.0
                
                # Calculate expected time span (from cutoff to end)
                expected_time_span_minutes = (end_timestamp - cutoff_time).total_seconds() / 60.0
                
                logger.warning(
                    "optimized_rolling_window_insufficient_klines",
                    end_timestamp=end_timestamp.isoformat(),
                    cutoff_time=cutoff_time.isoformat(),
                    klines_before_filter=klines_before_count,
                    klines_after_filter=klines_after_count,
                    buffer_first_timestamp=first_kline_ts,
                    buffer_last_timestamp=last_kline_ts,
                    expected_min_klines=15,
                    window_time_span_minutes=window_time_span_minutes,
                    expected_time_span_minutes=expected_time_span_minutes,
                    gap_minutes=expected_time_span_minutes - window_time_span_minutes if window_time_span_minutes < expected_time_span_minutes else 0.0,
                )
        
        # Create window structure for RollingWindows compatibility
        windows = {
            "1s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "3s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "15s": pd.DataFrame(columns=["timestamp", "price", "volume", "side"]),
            "1m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
            "5m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
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
            
            # Also create 5m window by aggregating 1m klines
            # This is needed for compute_all_candle_patterns_15m
            klines_sorted = klines_window.sort_values("timestamp").reset_index(drop=True)
            
            logger.debug(
                "optimized_rolling_window_creating_5m_window",
                end_timestamp=end_timestamp.isoformat(),
                klines_window_size=len(klines_window),
                klines_sorted_size=len(klines_sorted),
                first_kline_ts=klines_sorted.iloc[0]["timestamp"].isoformat() if len(klines_sorted) > 0 else None,
                last_kline_ts=klines_sorted.iloc[-1]["timestamp"].isoformat() if len(klines_sorted) > 0 else None,
            )
            
            if len(klines_sorted) >= 5:
                # Group by 5-minute intervals using standard time alignment
                # Use standard 5-minute boundaries instead of relative grouping
                # This ensures consistent grouping regardless of where data starts
                klines_sorted = klines_sorted.copy()
                
                # Ensure timezone-aware timestamps
                if not pd.api.types.is_datetime64_any_dtype(klines_sorted["timestamp"]):
                    klines_sorted["timestamp"] = pd.to_datetime(klines_sorted["timestamp"], utc=True)
                
                if klines_sorted["timestamp"].dt.tz is None:
                    klines_sorted["timestamp"] = klines_sorted["timestamp"].dt.tz_localize(timezone.utc)
                
                # Calculate 5-minute bucket: round down each timestamp to nearest 5-minute boundary
                # Method: convert to seconds since epoch, floor divide by 300 (5 minutes in seconds)
                from pandas import Timestamp
                epoch_start = Timestamp("1970-01-01", tz=timezone.utc)
                seconds_since_epoch = (klines_sorted["timestamp"] - epoch_start).dt.total_seconds()
                bucket_number = (seconds_since_epoch // 300).astype(int)  # Floor division by 5 minutes
                klines_sorted["time_bucket"] = bucket_number
                
                grouped = klines_sorted.groupby("time_bucket")
                klines_5m = pd.DataFrame({
                    "timestamp": grouped["timestamp"].first(),
                    "open": grouped["open"].first(),
                    "high": grouped["high"].max(),
                    "low": grouped["low"].min(),
                    "close": grouped["close"].last(),
                    "volume": grouped["volume"].sum(),
                }).reset_index(drop=True)
                
                # Sort by timestamp to ensure correct order
                klines_5m = klines_5m.sort_values("timestamp").reset_index(drop=True)
                
                windows["5m"] = klines_5m
                
                logger.debug(
                    "optimized_rolling_window_5m_window_created",
                    end_timestamp=end_timestamp.isoformat(),
                    klines_5m_count=len(klines_5m),
                    first_5m_ts=klines_5m.iloc[0]["timestamp"].isoformat() if len(klines_5m) > 0 else None,
                    last_5m_ts=klines_5m.iloc[-1]["timestamp"].isoformat() if len(klines_5m) > 0 else None,
                )
            else:
                logger.warning(
                    "optimized_rolling_window_insufficient_klines_for_5m",
                    end_timestamp=end_timestamp.isoformat(),
                    klines_sorted_size=len(klines_sorted),
                    required_minimum=5,
                )
        
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

