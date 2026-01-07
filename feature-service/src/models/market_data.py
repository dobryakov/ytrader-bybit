"""
Market data models.

Contains both event models (OrderbookSnapshot, Trade, etc.) and snapshot model (MarketData).
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OrderbookSnapshot(BaseModel):
    """Orderbook snapshot event."""
    event_type: str = Field(default="orderbook_snapshot")
    symbol: str
    timestamp: datetime
    sequence: int
    bids: List[List[float]] = Field(description="List of [price, quantity] pairs")
    asks: List[List[float]] = Field(description="List of [price, quantity] pairs")
    internal_timestamp: Optional[datetime] = None
    exchange_timestamp: Optional[datetime] = None


class OrderbookDelta(BaseModel):
    """Orderbook delta event."""
    event_type: str = Field(default="orderbook_delta")
    symbol: str
    timestamp: datetime
    sequence: int
    delta_type: str = Field(description="Type: insert, update, or delete")
    side: str = Field(description="bid or ask")
    price: float
    quantity: float
    internal_timestamp: Optional[datetime] = None
    exchange_timestamp: Optional[datetime] = None


class Trade(BaseModel):
    """Trade event."""
    event_type: str = Field(default="trade")
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str = Field(description="Buy or Sell")
    trade_time: Optional[datetime] = None
    internal_timestamp: Optional[datetime] = None


class Kline(BaseModel):
    """Kline/candlestick event."""
    event_type: str = Field(default="kline")
    symbol: str
    timestamp: datetime
    interval: str = Field(description="Time interval, e.g., 1m, 5m")
    open: float
    high: float
    low: float
    close: float
    volume: float
    internal_timestamp: Optional[datetime] = None


class Ticker(BaseModel):
    """Ticker event."""
    event_type: str = Field(default="ticker")
    symbol: str
    timestamp: datetime
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    internal_timestamp: Optional[datetime] = None


class FundingRate(BaseModel):
    """Funding rate event."""
    event_type: str = Field(default="funding_rate")
    symbol: str
    timestamp: datetime
    funding_rate: float
    next_funding_time: Optional[int] = Field(description="Unix timestamp in milliseconds")
    internal_timestamp: Optional[datetime] = None


class MarketDataEvent(BaseModel):
    """Generic market data event (union of all types)."""
    event_type: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    internal_timestamp: Optional[datetime] = None
    exchange_timestamp: Optional[datetime] = None


class MarketData(BaseModel):
    """
    Market data snapshot at a specific timestamp.
    
    Represents current market data (price, spread, volume, etc.) at a specific timestamp.
    This is separate from feature vector and contains reference market information.
    """
    price: float = Field(..., description="Current market price", gt=0)
    spread: float = Field(default=0.0, description="Bid-ask spread", ge=0)
    volume_24h: Optional[float] = Field(default=None, description="24-hour trading volume", ge=0)
    volatility: Optional[float] = Field(default=None, description="Current volatility measure", ge=0)
    orderbook_depth: Optional[Dict[str, float]] = Field(
        default=None, description="Order book depth (bid_depth, ask_depth)"
    )
    timestamp: datetime = Field(..., description="Timestamp when market data was captured")
