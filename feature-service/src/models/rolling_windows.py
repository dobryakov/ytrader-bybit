"""
Rolling Windows model for managing time-based rolling windows.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class RollingWindows(BaseModel):
    """Rolling windows for time-based feature computations."""
    
    symbol: str = Field(description="Trading pair symbol")
    windows: Dict[str, pd.DataFrame] = Field(description="Dictionary of interval to DataFrame")
    last_update: datetime = Field(description="Last update timestamp")
    window_intervals: Optional[set] = Field(default=None, description="Set of window intervals (e.g., {'1s', '3s', '15s', '1m'})")
    max_lookback_minutes_1m: Optional[int] = Field(default=None, description="Maximum lookback in minutes for 1m window")
    
    # Internal field for performance optimization (not serialized)
    _last_trim_time: Optional[datetime] = None
    
    def add_trade(self, trade: Dict) -> None:
        """Add trade to all relevant rolling windows.

        Важно: какие именно интервалы существуют, теперь определяется
        снаружи (FeatureComputer.get_rolling_windows). Здесь мы больше
        не создаём жёстко заданный набор окон, а обновляем только уже
        существующие ключи в self.windows.
        """
        # Parse timestamp - handle string, int (Unix ms), float, or None
        timestamp = trade.get("timestamp")
        if isinstance(timestamp, (int, float)):
            # Convert numeric timestamp (Unix milliseconds) to datetime
            timestamp = datetime.fromtimestamp(
                timestamp / 1000 if timestamp > 1e10 else timestamp,
                tz=timezone.utc
            )
        elif isinstance(timestamp, str):
            from dateutil.parser import parse
            timestamp = parse(timestamp)
            # Ensure timezone-aware datetime
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Ensure timestamp is timezone-aware
        if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Convert string values to float
        def to_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        trade_df = pd.DataFrame([{
            "timestamp": timestamp,
            "price": to_float(trade.get("price")),
            "volume": to_float(trade.get("quantity") or trade.get("volume"), 0.0),
            "side": trade.get("side", "Buy"),
        }])
        
        # Ensure numeric columns are float type
        numeric_cols = ["price", "volume"]
        for col in numeric_cols:
            if col in trade_df.columns:
                trade_df[col] = pd.to_numeric(trade_df[col], errors='coerce').fillna(0.0)
        
        # Add to all existing trade windows
        for interval, df in self.windows.items():
            # пропускаем не-трейдовые окна (например, клайны с колонками open/close)
            expected_cols = {"timestamp", "price", "volume", "side"}
            if not expected_cols.issubset(set(df.columns)):
                continue

            # Avoid FutureWarning: check if DataFrame is empty before concat
            if len(df) == 0:
                self.windows[interval] = trade_df.copy()
            else:
                self.windows[interval] = pd.concat([df, trade_df], ignore_index=True)
            # Ensure types are correct after concat (existing data might have string types)
            for col in numeric_cols:
                if col in self.windows[interval].columns:
                    # Convert entire column to numeric, handling any string values
                    self.windows[interval][col] = (
                        pd.to_numeric(self.windows[interval][col], errors="coerce")
                        .astype(float)
                        .fillna(0.0)
                    )
        
        self.last_update = timestamp
        self.trim_old_data()
    
    def add_kline(self, kline: Dict) -> None:
        """Add kline/candlestick to rolling windows."""
        # Extract kline data from payload if present
        payload = kline.get("payload", {})
        if not payload:
            payload = kline
        
        # Handle timestamp - can be in event or payload
        timestamp = kline.get("timestamp") or payload.get("timestamp") or payload.get("start")
        if isinstance(timestamp, str):
            from dateutil.parser import parse
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                timestamp = parse(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        elif not isinstance(timestamp, datetime):
            # Convert numeric timestamp to datetime
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
        
        # Normalize timestamp to timezone-aware UTC (critical for datetime operations)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
        
        # Convert string values to float
        def to_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        kline_df = pd.DataFrame([{
            "timestamp": timestamp,
            "open": to_float(payload.get("open") or payload.get("openPrice")),
            "high": to_float(payload.get("high") or payload.get("highPrice")),
            "low": to_float(payload.get("low") or payload.get("lowPrice")),
            "close": to_float(payload.get("close") or payload.get("closePrice")),
            "volume": to_float(payload.get("volume") or payload.get("volume24h"), 0.0),
        }])
        
        # Log incoming kline timestamp for debugging
        import structlog
        logger = structlog.get_logger(__name__)
        now_for_log = datetime.now(timezone.utc)
        # timestamp is already normalized to timezone-aware UTC above
        time_diff_seconds = (now_for_log - timestamp).total_seconds() if isinstance(timestamp, datetime) else None
        logger.debug(
            "kline_received",
            symbol=self.symbol,
            kline_timestamp=timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
            now=now_for_log.isoformat(),
            time_diff_seconds=round(time_diff_seconds, 2) if time_diff_seconds is not None else None,
            time_diff_minutes=round(time_diff_seconds / 60.0, 2) if time_diff_seconds is not None else None,
            is_old=time_diff_seconds > 60 if time_diff_seconds is not None else False,
        )
        
        # Ensure numeric columns are float type
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in kline_df.columns:
                kline_df[col] = pd.to_numeric(kline_df[col], errors='coerce').fillna(0.0)
        
        # Add to 1m window (klines are typically 1m+ intervals)
        if "1m" not in self.windows:
            self.windows["1m"] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Avoid FutureWarning: check if DataFrame is empty before concat
        if len(self.windows["1m"]) == 0:
            self.windows["1m"] = kline_df.copy()
        else:
            self.windows["1m"] = pd.concat([self.windows["1m"], kline_df], ignore_index=True)
        # Ensure types are correct after concat (existing data might have string types)
        # This is critical: convert ALL numeric columns to float, not just new ones
        for col in numeric_cols:
            if col in self.windows["1m"].columns:
                # Convert entire column to numeric, handling any string values
                self.windows["1m"][col] = pd.to_numeric(self.windows["1m"][col], errors='coerce').astype(float).fillna(0.0)
        
        # Ensure last_update is always a datetime object
        # Always use the maximum timestamp between current last_update and new kline timestamp
        # This ensures last_update reflects the most recent data, even if klines arrive out of order
        new_last_update = None
        if isinstance(timestamp, datetime):
            new_last_update = timestamp
        elif isinstance(timestamp, str):
            from dateutil.parser import parse
            try:
                new_last_update = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                new_last_update = parse(timestamp)
        else:
            new_last_update = datetime.now(timezone.utc)
        
        # Ensure timezone-aware
        if new_last_update.tzinfo is None:
            new_last_update = new_last_update.replace(tzinfo=timezone.utc)
        
        # Update last_update to maximum of current and new timestamp
        current_last_update = self.last_update
        if isinstance(current_last_update, str):
            from dateutil.parser import parse
            try:
                current_last_update = datetime.fromisoformat(current_last_update.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                current_last_update = parse(current_last_update)
        elif not isinstance(current_last_update, datetime):
            current_last_update = datetime.now(timezone.utc)
        
        if current_last_update.tzinfo is None:
            current_last_update = current_last_update.replace(tzinfo=timezone.utc)
        
        # Use maximum timestamp to ensure last_update always reflects most recent data
        old_last_update = self.last_update
        self.last_update = max(current_last_update, new_last_update)
        
        # Log if last_update was updated
        if old_last_update != self.last_update:
            import structlog
            logger = structlog.get_logger(__name__)
            logger.debug(
                "last_update_updated",
                symbol=self.symbol,
                old_last_update=str(old_last_update),
                new_last_update=self.last_update.isoformat() if isinstance(self.last_update, datetime) else str(self.last_update),
                kline_timestamp=new_last_update.isoformat() if isinstance(new_last_update, datetime) else str(new_last_update),
            )
        
        # Log kline addition for debugging
        klines_count_before_trim = len(self.windows.get("1m", pd.DataFrame()))
        # IMPORTANT: trim_old_data() uses self.last_update, so we must update it BEFORE trimming
        # This ensures that new klines are not immediately trimmed
        # Performance optimization: only trim if we have significantly more data than expected
        # AND if enough time has passed since last trim (to avoid excessive CPU usage)
        # Expected max is ~35 minutes * klines per second (klines come as updates, not just once per minute)
        # Use a higher threshold and time-based throttling to reduce CPU load
        expected_max_klines = (self.max_lookback_minutes_1m + 5) * 60 if self.max_lookback_minutes_1m else 2100  # 35 min * 60 sec
        now = datetime.now(timezone.utc)
        # Use instance attribute for tracking last trim time (not part of Pydantic model)
        if not hasattr(self, '_last_trim_time'):
            self._last_trim_time = None
        time_since_last_trim = (now - self._last_trim_time).total_seconds() if self._last_trim_time else float('inf')
        should_trim = (
            klines_count_before_trim > max(expected_max_klines * 1.2, 300) and  # At least 300 or 1.2x expected
            time_since_last_trim > 10  # Don't trim more often than every 10 seconds
        )
        if should_trim:
            self.trim_old_data()
            self._last_trim_time = now
        klines_count_after_trim = len(self.windows.get("1m", pd.DataFrame()))
        
        import structlog
        logger = structlog.get_logger(__name__)
        if klines_count_before_trim != klines_count_after_trim:
            logger.warning(
                "klines_trimmed_after_add",
                symbol=self.symbol,
                before=klines_count_before_trim,
                after=klines_count_after_trim,
                max_lookback_minutes_1m=self.max_lookback_minutes_1m,
                last_update=self.last_update.isoformat() if isinstance(self.last_update, datetime) else str(self.last_update),
            )
        
        # Log current klines count for monitoring
        # Calculate time span of klines to verify window width
        klines_df = self.windows.get("1m", pd.DataFrame())
        time_span_minutes = 0.0
        oldest_timestamp = None
        newest_timestamp = None
        if len(klines_df) > 0 and "timestamp" in klines_df.columns:
            # Normalize timestamps
            if klines_df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
                klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], unit='ms', utc=True)
            elif klines_df["timestamp"].dtype == 'object':
                klines_df["timestamp"] = pd.to_datetime(klines_df["timestamp"], errors='coerce', utc=True)
            
            if klines_df["timestamp"].dtype.name.startswith('datetime'):
                if klines_df["timestamp"].dt.tz is None:
                    klines_df["timestamp"] = klines_df["timestamp"].dt.tz_localize(timezone.utc)
                oldest_timestamp = klines_df["timestamp"].min()
                newest_timestamp = klines_df["timestamp"].max()
                if oldest_timestamp and newest_timestamp:
                    # Normalize timestamps to timezone-aware UTC before subtraction
                    if isinstance(oldest_timestamp, pd.Timestamp):
                        oldest_timestamp = oldest_timestamp.to_pydatetime()
                    if isinstance(newest_timestamp, pd.Timestamp):
                        newest_timestamp = newest_timestamp.to_pydatetime()
                    if isinstance(oldest_timestamp, datetime) and oldest_timestamp.tzinfo is None:
                        oldest_timestamp = oldest_timestamp.replace(tzinfo=timezone.utc)
                    if isinstance(newest_timestamp, datetime) and newest_timestamp.tzinfo is None:
                        newest_timestamp = newest_timestamp.replace(tzinfo=timezone.utc)
                    time_span_minutes = (newest_timestamp - oldest_timestamp).total_seconds() / 60.0
        
        logger.info(
            "klines_count_after_add",
            symbol=self.symbol,
            klines_count=klines_count_after_trim,
            max_lookback_minutes_1m=self.max_lookback_minutes_1m,
            expected_minimum=15,  # Minimum for candle patterns
            time_span_minutes=round(time_span_minutes, 2),
            oldest_timestamp=oldest_timestamp.isoformat() if oldest_timestamp else None,
            newest_timestamp=newest_timestamp.isoformat() if newest_timestamp else None,
            window_width_should_be_minutes=(self.max_lookback_minutes_1m + 5) if self.max_lookback_minutes_1m else 30,
        )
    
    def trim_old_data(
        self, 
        window_seconds: Optional[int] = None,
        max_lookback_minutes_1m: Optional[int] = None,
    ) -> None:
        """
        Trim old data outside window boundaries.
        
        Args:
            window_seconds: Override window size in seconds for all intervals
            max_lookback_minutes_1m: Maximum lookback in minutes for "1m" window.
                If None, uses default 30 minutes. Should be computed from Feature Registry
                using FeatureRequirementsAnalyzer to ensure all features
                have sufficient historical data after trimming.
        """
        from datetime import timezone
        # IMPORTANT: Use current time for trimming, not self.last_update
        # This ensures that klines with timestamps in the past (from queue backlog) are not
        # immediately trimmed. We trim based on actual current time, keeping data within
        # the lookback window from NOW, not from last_update.
        now = datetime.now(timezone.utc)
        
        # Normalize self.last_update format if needed, but don't use it for trimming cutoff
        if isinstance(self.last_update, str):
            from dateutil.parser import parse
            try:
                parsed = datetime.fromisoformat(self.last_update.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                parsed = parse(self.last_update)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            # Only update if parsed is newer than current last_update
            if isinstance(self.last_update, datetime):
                self.last_update = max(self.last_update, parsed)
            else:
                self.last_update = parsed
        elif not isinstance(self.last_update, datetime):
            self.last_update = now
        elif self.last_update.tzinfo is None:
            self.last_update = self.last_update.replace(tzinfo=timezone.utc)
        
        # Window sizes for trimming old data
        # For "1m" window, use parameter if provided, otherwise use instance field, otherwise default to 30 minutes
        # This ensures all features have sufficient historical data after trimming
        # Default 30 minutes covers maximum feature lookback (26 min for ema_21) + buffer
        default_1m_window_seconds = 1800  # 30 minutes default
        lookback_to_use = max_lookback_minutes_1m if max_lookback_minutes_1m is not None else self.max_lookback_minutes_1m
        if lookback_to_use is not None:
            # Add 5 minute buffer to max_lookback for safety
            window_size_1m_seconds = (lookback_to_use + 5) * 60
        else:
            window_size_1m_seconds = default_1m_window_seconds
        
        window_sizes = {
            "1s": 1,
            "3s": 3,
            "15s": 15,
            "1m": window_size_1m_seconds,
        }
        
        # Log window sizes for debugging (changed to DEBUG to reduce logging overhead)
        import structlog
        logger = structlog.get_logger(__name__)
        logger.debug(
            "trim_old_data_window_sizes",
            symbol=self.symbol,
            max_lookback_minutes_1m=self.max_lookback_minutes_1m,
            lookback_to_use=lookback_to_use,
            window_size_1m_seconds=window_size_1m_seconds,
            window_size_1m_minutes=round(window_size_1m_seconds / 60.0, 2),
            now=now.isoformat(),
        )
        
        for interval, df in self.windows.items():
            if len(df) == 0:
                continue
            
            window_size = window_sizes.get(interval, 60)
            if window_seconds is not None:
                window_size = window_seconds
            
            cutoff_time = now - timedelta(seconds=window_size)
            
            # Log cutoff time for debugging (changed to DEBUG to reduce logging overhead)
            if interval == "1m":
                logger.debug(
                    "trim_old_data_cutoff",
                    symbol=self.symbol,
                    interval=interval,
                    now=now.isoformat(),
                    window_size_seconds=window_size,
                    cutoff_time=cutoff_time.isoformat(),
                    data_count_before=len(df),
                )
            
            # Keep only data within window
            if "timestamp" in df.columns:
                # Performance optimization: avoid double copying
                # Only copy if we need to normalize timestamps
                before_count = len(df)
                needs_normalization = (
                    df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32'] or
                    df["timestamp"].dtype == 'object' or
                    (df["timestamp"].dtype.name.startswith('datetime') and df["timestamp"].dt.tz is None)
                )
                
                if needs_normalization:
                    # Copy and normalize timestamp column
                    df = df.copy()
                    if df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
                    elif df["timestamp"].dtype == 'object':
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
                    # Ensure all timestamps are timezone-aware
                    if df["timestamp"].dtype.name.startswith('datetime'):
                        if df["timestamp"].dt.tz is None:
                            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
                
                # Filter creates a new DataFrame (boolean indexing always returns a copy)
                filtered_df = df[df["timestamp"] >= cutoff_time]
                after_count = len(filtered_df)
                self.windows[interval] = filtered_df
                if before_count != after_count and interval == "1m":
                    import structlog
                    logger = structlog.get_logger(__name__)
                    logger.debug(
                        "klines_trimmed_in_trim_old_data",
                        symbol=self.symbol,
                        interval=interval,
                        before=before_count,
                        after=after_count,
                        cutoff_time=cutoff_time.isoformat(),
                        window_size_seconds=window_size,
                        max_lookback_minutes_1m=self.max_lookback_minutes_1m,
                    )
    
    def get_window_data(self, interval: str) -> pd.DataFrame:
        """Get data for specific window interval."""
        return self.windows.get(interval, pd.DataFrame())
    
    def get_last_available_timestamp(self, interval: str) -> Optional[datetime]:
        """
        Get the most recent timestamp from available data in the specified interval.
        
        Args:
            interval: Window interval (e.g., "1m", "1s")
            
        Returns:
            Most recent timestamp from available data, or None if no data available
        """
        df = self.get_window_data(interval)
        
        if len(df) == 0 or "timestamp" not in df.columns:
            return None
        
        # Normalize timestamp column to timezone-aware datetime if needed
        df = df.copy()
        if df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        elif df["timestamp"].dtype == 'object':
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        
        # Ensure all timestamps are timezone-aware
        if df["timestamp"].dtype.name.startswith('datetime'):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
        
        # Get maximum timestamp (most recent)
        max_timestamp = df["timestamp"].max()
        
        # Handle case where max_timestamp might be NaT (if all timestamps were invalid)
        if pd.isna(max_timestamp):
            return None
        
        # Ensure timezone-aware
        if isinstance(max_timestamp, datetime) and max_timestamp.tzinfo is None:
            max_timestamp = max_timestamp.replace(tzinfo=timezone.utc)
        
        return max_timestamp
    
    def get_trades_for_window(self, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get trades within time window."""
        df = self.get_window_data(interval)
        
        if len(df) == 0 or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        
        # Normalize timestamp column to timezone-aware datetime if needed
        df = df.copy()
        if df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Convert Unix timestamp (milliseconds) to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        elif df["timestamp"].dtype == 'object':
            # Try to convert string or mixed types to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        
        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        # Ensure all timestamps in DataFrame are timezone-aware
        if df["timestamp"].dtype.name.startswith('datetime'):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
        
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        result_df = df[mask].copy()
        
        # Ensure numeric columns are float type (defensive conversion)
        numeric_cols = ["price", "volume"]
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype(float).fillna(0.0)
        
        return result_df
    
    def get_klines_for_window(self, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get klines within time window."""
        df = self.get_window_data(interval)
        
        if len(df) == 0 or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Normalize timestamp column to timezone-aware datetime if needed (same as get_trades_for_window)
        df = df.copy()
        if df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Convert Unix timestamp (milliseconds) to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        elif df["timestamp"].dtype == 'object':
            # Try to convert string or mixed types to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
        
        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        # Ensure all timestamps in DataFrame are timezone-aware
        if df["timestamp"].dtype.name.startswith('datetime'):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
        
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        result_df = df[mask].copy()
        
        # Ensure numeric columns are float type (defensive conversion)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype(float).fillna(0.0)
        
        return result_df
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow pandas DataFrame
        # Note: json_encoders deprecated in Pydantic v2
        # datetime serialization handled automatically by Pydantic
        # DataFrame serialization should use custom serializer if needed
    )

