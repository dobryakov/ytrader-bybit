"""
Market Data model.

Represents current market data (price, spread, volume, etc.) at a specific timestamp.
Matches Feature Service MarketData structure.
"""
from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    """Market data snapshot at a specific timestamp."""
    
    price: float = Field(..., description="Current market price", gt=0)
    spread: float = Field(default=0.0, description="Bid-ask spread", ge=0)
    volume_24h: Optional[float] = Field(default=None, description="24-hour trading volume", ge=0)
    volatility: Optional[float] = Field(default=None, description="Current volatility measure", ge=0)
    orderbook_depth: Optional[Dict[str, float]] = Field(
        default=None, description="Order book depth (bid_depth, ask_depth)"
    )
    timestamp: datetime = Field(..., description="Timestamp when market data was captured")

