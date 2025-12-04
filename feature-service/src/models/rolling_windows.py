"""
Rolling Windows model for managing time-based rolling windows.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import pandas as pd
from pydantic import BaseModel, Field


class RollingWindows(BaseModel):
    """Rolling windows for time-based feature computations."""
    
    symbol: str = Field(description="Trading pair symbol")
    windows: Dict[str, pd.DataFrame] = Field(description="Dictionary of interval to DataFrame")
    last_update: datetime = Field(description="Last update timestamp")
    
    def add_trade(self, trade: Dict) -> None:
        """Add trade to all relevant rolling windows."""
        # Parse timestamp if it's a string
        timestamp = trade.get("timestamp")
        if isinstance(timestamp, str):
            from dateutil.parser import parse
            timestamp = parse(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        trade_df = pd.DataFrame([{
            "timestamp": timestamp,
            "price": trade["price"],
            "volume": trade.get("quantity", trade.get("volume", 0.0)),
            "side": trade["side"],
        }])
        
        # Add to all windows
        for interval in ["1s", "3s", "15s", "1m"]:
            if interval not in self.windows:
                self.windows[interval] = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
            
            self.windows[interval] = pd.concat([self.windows[interval], trade_df], ignore_index=True)
        
        self.last_update = timestamp
        self.trim_old_data()
    
    def add_kline(self, kline: Dict) -> None:
        """Add kline/candlestick to rolling windows."""
        kline_df = pd.DataFrame([{
            "timestamp": kline["timestamp"],
            "open": kline["open"],
            "high": kline["high"],
            "low": kline["low"],
            "close": kline["close"],
            "volume": kline["volume"],
        }])
        
        # Add to 1m window (klines are typically 1m+ intervals)
        if "1m" not in self.windows:
            self.windows["1m"] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        self.windows["1m"] = pd.concat([self.windows["1m"], kline_df], ignore_index=True)
        self.last_update = kline["timestamp"]
        self.trim_old_data()
    
    def trim_old_data(self, window_seconds: Optional[int] = None) -> None:
        """Trim old data outside window boundaries."""
        from datetime import timezone
        now = self.last_update if self.last_update else datetime.now(timezone.utc)
        
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
                self.windows[interval] = df[df["timestamp"] >= cutoff_time].copy()
    
    def get_window_data(self, interval: str) -> pd.DataFrame:
        """Get data for specific window interval."""
        return self.windows.get(interval, pd.DataFrame())
    
    def get_trades_for_window(self, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get trades within time window."""
        df = self.get_window_data(interval)
        
        if len(df) == 0 or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        return df[mask].copy()
    
    def get_klines_for_window(self, interval: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get klines within time window."""
        df = self.get_window_data(interval)
        
        if len(df) == 0 or "timestamp" not in df.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
        return df[mask].copy()
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True  # Allow pandas DataFrame
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            pd.DataFrame: lambda v: v.to_dict("records"),
        }

