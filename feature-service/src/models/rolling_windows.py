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
    
    def add_trade(self, trade: Dict) -> None:
        """Add trade to all relevant rolling windows."""
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
        
        # Add to all windows
        for interval in ["1s", "3s", "15s", "1m"]:
            if interval not in self.windows:
                self.windows[interval] = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
            
            # Avoid FutureWarning: check if DataFrame is empty before concat
            if len(self.windows[interval]) == 0:
                self.windows[interval] = trade_df.copy()
            else:
                self.windows[interval] = pd.concat([self.windows[interval], trade_df], ignore_index=True)
            # Ensure types are correct after concat (existing data might have string types)
            for col in numeric_cols:
                if col in self.windows[interval].columns:
                    # Convert entire column to numeric, handling any string values
                    self.windows[interval][col] = pd.to_numeric(self.windows[interval][col], errors='coerce').astype(float).fillna(0.0)
        
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
        if isinstance(timestamp, datetime):
            self.last_update = timestamp
        elif isinstance(timestamp, str):
            from dateutil.parser import parse
            try:
                self.last_update = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                self.last_update = parse(timestamp)
        else:
            self.last_update = datetime.now(timezone.utc)
        self.trim_old_data()
    
    def trim_old_data(self, window_seconds: Optional[int] = None) -> None:
        """Trim old data outside window boundaries."""
        from datetime import timezone
        # Ensure now is always a timezone-aware datetime object
        now = self.last_update
        if isinstance(now, str):
            from dateutil.parser import parse
            try:
                now = datetime.fromisoformat(now.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                now = parse(now)
            # Ensure timezone-aware
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            # Update last_update to datetime object to prevent future issues
            self.last_update = now
        elif not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
            # Update last_update to datetime object
            self.last_update = now
        elif now.tzinfo is None:
            # Make timezone-aware if naive
            now = now.replace(tzinfo=timezone.utc)
            self.last_update = now
        
        window_sizes = {
            "1s": 1,
            "3s": 3,
            "15s": 15,
            "1m": 60,
        }
        
        for interval, df in self.windows.items():
            if len(df) == 0:
                continue
            
            window_size = window_sizes.get(interval, 60)
            if window_seconds is not None:
                window_size = window_seconds
            
            cutoff_time = now - timedelta(seconds=window_size)
            
            # Keep only data within window
            if "timestamp" in df.columns:
                # Normalize timestamp column to timezone-aware datetime if needed
                df = df.copy()
                if df["timestamp"].dtype in ['int64', 'float64', 'int32', 'float32']:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
                elif df["timestamp"].dtype == 'object':
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
                # Ensure all timestamps are timezone-aware
                if df["timestamp"].dtype.name.startswith('datetime'):
                    df["timestamp"] = df["timestamp"].dt.tz_localize(None).dt.tz_localize(timezone.utc) if df["timestamp"].dt.tz is None else df["timestamp"]
                self.windows[interval] = df[df["timestamp"] >= cutoff_time].copy()
    
    def get_window_data(self, interval: str) -> pd.DataFrame:
        """Get data for specific window interval."""
        return self.windows.get(interval, pd.DataFrame())
    
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

