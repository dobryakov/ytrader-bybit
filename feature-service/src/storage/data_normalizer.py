"""
Data normalization utilities for unifying data formats from WebSocket and backfilling.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
from src.logging import get_logger

logger = get_logger(__name__)


def normalize_kline_data(
    kline_data: Dict[str, Any],
    source: str = "unknown",
    internal_timestamp: Optional[datetime] = None,
    exchange_timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Normalize kline data to unified format.
    
    Unified format:
    - timestamp: datetime (timezone-aware UTC)
    - symbol: str
    - interval: str (e.g., "1m")
    - open: float
    - high: float
    - low: float
    - close: float
    - volume: float
    - internal_timestamp: datetime (timezone-aware UTC) or None
    - exchange_timestamp: datetime (timezone-aware UTC) or None
    
    Args:
        kline_data: Raw kline data dictionary
        source: Source of data ("websocket" or "backfilling")
        internal_timestamp: Internal timestamp (for WebSocket events)
        exchange_timestamp: Exchange timestamp (for WebSocket events)
        
    Returns:
        Normalized kline data dictionary
    """
    # Extract timestamp
    timestamp = kline_data.get("timestamp")
    if timestamp is None:
        # Try alternative field names
        timestamp = kline_data.get("start") or kline_data.get("t")
    
    # Parse timestamp
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            from dateutil.parser import parse
            timestamp = parse(timestamp)
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp (ms or s)
        timestamp = datetime.fromtimestamp(
            timestamp / 1000 if timestamp > 1e10 else timestamp,
            tz=timezone.utc
        )
    elif not isinstance(timestamp, datetime):
        # Fallback to current time
        timestamp = datetime.now(timezone.utc)
    
    # Ensure timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    
    # Extract symbol
    symbol = kline_data.get("symbol") or kline_data.get("s")
    if not symbol:
        raise ValueError("Symbol is required for kline data")
    
    # Extract interval
    interval = kline_data.get("interval") or kline_data.get("i") or "1m"
    # Normalize interval format (ensure it ends with 'm' for minutes)
    if isinstance(interval, (int, float)):
        interval = f"{int(interval)}m"
    elif not isinstance(interval, str):
        interval = "1m"
    
    # Extract prices and volume
    def to_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    open_price = to_float(kline_data.get("open") or kline_data.get("o"))
    high_price = to_float(kline_data.get("high") or kline_data.get("h"))
    low_price = to_float(kline_data.get("low") or kline_data.get("l"))
    close_price = to_float(kline_data.get("close") or kline_data.get("c"))
    volume = to_float(kline_data.get("volume") or kline_data.get("v"))
    
    # Use exchange_timestamp from parameter if provided, otherwise from data
    if exchange_timestamp is None:
        exchange_timestamp = timestamp
    
    # Use internal_timestamp from parameter if provided, otherwise None
    if internal_timestamp is None:
        internal_timestamp = None
    
    normalized = {
        "timestamp": timestamp,
        "symbol": symbol,
        "interval": interval,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "internal_timestamp": internal_timestamp,
        "exchange_timestamp": exchange_timestamp,
    }
    
    logger.debug(
        "kline_data_normalized",
        source=source,
        symbol=symbol,
        timestamp=timestamp.isoformat(),
        interval=interval,
    )
    
    return normalized


def normalize_trade_data(
    trade_data: Dict[str, Any],
    source: str = "unknown",
    internal_timestamp: Optional[datetime] = None,
    exchange_timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Normalize trade data to unified format.
    
    Unified format:
    - timestamp: datetime (timezone-aware UTC)
    - symbol: str
    - price: float
    - quantity: float
    - side: str ("Buy" or "Sell")
    - trade_time: datetime (timezone-aware UTC) or None
    - internal_timestamp: datetime (timezone-aware UTC) or None
    - exchange_timestamp: datetime (timezone-aware UTC) or None
    
    Args:
        trade_data: Raw trade data dictionary
        source: Source of data ("websocket" or "backfilling")
        internal_timestamp: Internal timestamp (for WebSocket events)
        exchange_timestamp: Exchange timestamp (for WebSocket events)
        
    Returns:
        Normalized trade data dictionary
    """
    # Extract timestamp
    timestamp = trade_data.get("timestamp") or trade_data.get("T") or trade_data.get("trade_time")
    
    # Parse timestamp
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            from dateutil.parser import parse
            timestamp = parse(timestamp)
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp (ms or s)
        timestamp = datetime.fromtimestamp(
            timestamp / 1000 if timestamp > 1e10 else timestamp,
            tz=timezone.utc
        )
    elif not isinstance(timestamp, datetime):
        # Fallback to current time
        timestamp = datetime.now(timezone.utc)
    
    # Ensure timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    
    # Extract symbol
    symbol = trade_data.get("symbol") or trade_data.get("s")
    if not symbol:
        raise ValueError("Symbol is required for trade data")
    
    # Extract price and quantity
    def to_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    price = to_float(trade_data.get("price") or trade_data.get("p"))
    quantity = to_float(trade_data.get("quantity") or trade_data.get("volume") or trade_data.get("v"))
    
    # Extract side
    side = trade_data.get("side") or trade_data.get("S") or "Buy"
    if side not in ["Buy", "Sell"]:
        # Normalize side
        side = "Buy" if str(side).lower() in ["buy", "b", "1"] else "Sell"
    
    # Extract trade_time (may be different from timestamp)
    trade_time = trade_data.get("trade_time") or trade_data.get("T")
    if trade_time:
        if isinstance(trade_time, str):
            try:
                trade_time = datetime.fromisoformat(trade_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                from dateutil.parser import parse
                trade_time = parse(trade_time)
        elif isinstance(trade_time, (int, float)):
            trade_time = datetime.fromtimestamp(
                trade_time / 1000 if trade_time > 1e10 else trade_time,
                tz=timezone.utc
            )
        elif not isinstance(trade_time, datetime):
            trade_time = timestamp
    else:
        trade_time = timestamp
    
    # Ensure timezone-aware
    if trade_time.tzinfo is None:
        trade_time = trade_time.replace(tzinfo=timezone.utc)
    else:
        trade_time = trade_time.astimezone(timezone.utc)
    
    # Use exchange_timestamp from parameter if provided, otherwise from data
    if exchange_timestamp is None:
        exchange_timestamp = timestamp
    
    # Use internal_timestamp from parameter if provided, otherwise None
    if internal_timestamp is None:
        internal_timestamp = None
    
    normalized = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": price,
        "quantity": quantity,
        "side": side,
        "trade_time": trade_time,
        "internal_timestamp": internal_timestamp,
        "exchange_timestamp": exchange_timestamp,
    }
    
    logger.debug(
        "trade_data_normalized",
        source=source,
        symbol=symbol,
        timestamp=timestamp.isoformat(),
        price=price,
    )
    
    return normalized


def normalize_funding_data(
    funding_data: Dict[str, Any],
    source: str = "unknown",
    internal_timestamp: Optional[datetime] = None,
    exchange_timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Normalize funding rate data to unified format.
    
    Unified format:
    - timestamp: datetime (timezone-aware UTC)
    - symbol: str
    - funding_rate: float
    - next_funding_time: datetime (timezone-aware UTC) or None
    - internal_timestamp: datetime (timezone-aware UTC) or None
    - exchange_timestamp: datetime (timezone-aware UTC) or None
    
    Args:
        funding_data: Raw funding rate data dictionary
        source: Source of data ("websocket" or "backfilling")
        internal_timestamp: Internal timestamp (for WebSocket events)
        exchange_timestamp: Exchange timestamp (for WebSocket events)
        
    Returns:
        Normalized funding rate data dictionary
    """
    # Extract timestamp
    timestamp = funding_data.get("timestamp") or funding_data.get("fundingRateTimestamp")
    
    # Parse timestamp
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            from dateutil.parser import parse
            timestamp = parse(timestamp)
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp (ms or s)
        timestamp = datetime.fromtimestamp(
            timestamp / 1000 if timestamp > 1e10 else timestamp,
            tz=timezone.utc
        )
    elif not isinstance(timestamp, datetime):
        # Fallback to current time
        timestamp = datetime.now(timezone.utc)
    
    # Ensure timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    
    # Extract symbol
    symbol = funding_data.get("symbol") or funding_data.get("s")
    if not symbol:
        raise ValueError("Symbol is required for funding rate data")
    
    # Extract funding rate
    def to_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    funding_rate = to_float(
        funding_data.get("funding_rate") or 
        funding_data.get("fundingRate") or 
        funding_data.get("rate")
    )
    
    # Extract next funding time
    next_funding_time = funding_data.get("next_funding_time") or funding_data.get("nextFundingTime")
    if next_funding_time:
        if isinstance(next_funding_time, str):
            try:
                next_funding_time = datetime.fromisoformat(next_funding_time.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                from dateutil.parser import parse
                next_funding_time = parse(next_funding_time)
        elif isinstance(next_funding_time, (int, float)):
            next_funding_time = datetime.fromtimestamp(
                next_funding_time / 1000 if next_funding_time > 1e10 else next_funding_time,
                tz=timezone.utc
            )
        elif not isinstance(next_funding_time, datetime):
            next_funding_time = None
        
        if next_funding_time and next_funding_time.tzinfo is None:
            next_funding_time = next_funding_time.replace(tzinfo=timezone.utc)
        elif next_funding_time:
            next_funding_time = next_funding_time.astimezone(timezone.utc)
    else:
        next_funding_time = None
    
    # Use exchange_timestamp from parameter if provided, otherwise from data
    if exchange_timestamp is None:
        exchange_timestamp = timestamp
    
    # Use internal_timestamp from parameter if provided, otherwise None
    if internal_timestamp is None:
        internal_timestamp = None
    
    normalized = {
        "timestamp": timestamp,
        "symbol": symbol,
        "funding_rate": funding_rate,
        "next_funding_time": next_funding_time,
        "internal_timestamp": internal_timestamp,
        "exchange_timestamp": exchange_timestamp,
    }
    
    logger.debug(
        "funding_data_normalized",
        source=source,
        symbol=symbol,
        timestamp=timestamp.isoformat(),
        funding_rate=funding_rate,
    )
    
    return normalized

