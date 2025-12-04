"""
Orderbook State model for managing in-memory orderbook state.
"""
from datetime import datetime, timezone
from typing import Dict, Optional
from sortedcontainers import SortedDict
from pydantic import BaseModel, Field, field_validator


class OrderbookState(BaseModel):
    """In-memory orderbook state for a symbol."""
    
    symbol: str = Field(description="Trading pair symbol")
    sequence: int = Field(description="Current sequence number")
    timestamp: datetime = Field(description="Last update timestamp")
    bids: SortedDict = Field(description="Sorted bids by price (descending): {price: quantity}")
    asks: SortedDict = Field(description="Sorted asks by price (ascending): {price: quantity}")
    last_snapshot_at: Optional[datetime] = Field(default=None, description="When last snapshot was applied")
    delta_count: int = Field(default=0, description="Number of deltas applied since snapshot")
    
    @field_validator("bids", "asks", mode="before")
    @classmethod
    def convert_to_sorted_dict(cls, v):
        """Convert dict/list to SortedDict if needed."""
        if isinstance(v, dict) and not isinstance(v, SortedDict):
            return SortedDict(v)
        elif isinstance(v, list):
            return SortedDict(v)
        return v
    
    @classmethod
    def from_snapshot(cls, snapshot: Dict) -> "OrderbookState":
        """Create orderbook state from snapshot event."""
        bids = SortedDict()
        asks = SortedDict()
        
        # Populate bids (descending order - highest first)
        # Convert prices and quantities to float (Bybit sends them as strings)
        for price, quantity in snapshot.get("bids", []):
            try:
                price_float = float(price) if isinstance(price, str) else price
                quantity_float = float(quantity) if isinstance(quantity, str) else quantity
                bids[price_float] = quantity_float
            except (ValueError, TypeError):
                # Skip invalid entries
                continue
        
        # Populate asks (ascending order - lowest first)
        for price, quantity in snapshot.get("asks", []):
            try:
                price_float = float(price) if isinstance(price, str) else price
                quantity_float = float(quantity) if isinstance(quantity, str) else quantity
                asks[price_float] = quantity_float
            except (ValueError, TypeError):
                # Skip invalid entries
                continue
        
        return cls(
            symbol=snapshot["symbol"],
            sequence=snapshot["sequence"],
            timestamp=snapshot["timestamp"],
            bids=bids,
            asks=asks,
            last_snapshot_at=snapshot.get("timestamp"),
            delta_count=0,
        )
    
    def apply_delta(self, delta: Dict) -> None:
        """Apply delta update to orderbook state."""
        delta_type = delta["delta_type"]
        side = delta["side"]
        price = delta["price"]
        quantity = delta["quantity"]
        
        # Convert to float if strings (Bybit sends as strings)
        try:
            price = float(price) if isinstance(price, str) else price
            quantity = float(quantity) if isinstance(quantity, str) else quantity
        except (ValueError, TypeError):
            return  # Skip invalid delta
        
        target = self.bids if side == "bid" else self.asks
        
        if delta_type == "insert" or delta_type == "update":
            target[price] = quantity
        elif delta_type == "delete":
            if price in target:
                del target[price]
        
        self.sequence = delta["sequence"]
        self.timestamp = delta["timestamp"]
        self.delta_count += 1
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price (highest)."""
        if len(self.bids) == 0:
            return None
        return self.bids.peekitem(-1)[0]  # Last item (highest price)
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price (lowest)."""
        if len(self.asks) == 0:
            return None
        return self.asks.peekitem(0)[0]  # First item (lowest price)
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price (average of best bid and ask)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def get_spread_abs(self) -> Optional[float]:
        """Get absolute spread (ask - bid)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return best_ask - best_bid
    
    def get_spread_rel(self) -> Optional[float]:
        """Get relative spread (spread / mid_price)."""
        spread_abs = self.get_spread_abs()
        mid_price = self.get_mid_price()
        
        if spread_abs is None or mid_price is None or mid_price == 0:
            return None
        
        return spread_abs / mid_price
    
    def get_depth_bid_top5(self) -> float:
        """Get total depth of top 5 bid levels."""
        total = 0.0
        count = 0
        for price, quantity in reversed(self.bids.items()):
            if count >= 5:
                break
            total += quantity
            count += 1
        return total
    
    def get_depth_bid_top10(self) -> float:
        """Get total depth of top 10 bid levels."""
        total = 0.0
        count = 0
        for price, quantity in reversed(self.bids.items()):
            if count >= 10:
                break
            total += quantity
            count += 1
        return total
    
    def get_depth_ask_top5(self) -> float:
        """Get total depth of top 5 ask levels."""
        total = 0.0
        count = 0
        for price, quantity in self.asks.items():
            if count >= 5:
                break
            total += quantity
            count += 1
        return total
    
    def get_depth_ask_top10(self) -> float:
        """Get total depth of top 10 ask levels."""
        total = 0.0
        count = 0
        for price, quantity in self.asks.items():
            if count >= 10:
                break
            total += quantity
            count += 1
        return total
    
    def get_imbalance_top5(self) -> float:
        """Get orderbook imbalance for top 5 levels (normalized: -1 to 1)."""
        depth_bid = self.get_depth_bid_top5()
        depth_ask = self.get_depth_ask_top5()
        
        total = depth_bid + depth_ask
        if total == 0:
            return 0.0
        
        # Normalized: (bid - ask) / (bid + ask)
        return (depth_bid - depth_ask) / total
    
    def has_sequence_gap(self, next_sequence: int) -> bool:
        """Check if there's a sequence gap."""
        return next_sequence != self.sequence + 1
    
    def is_desynchronized(self, max_delta_count: int = 1000, max_age_seconds: int = 60) -> bool:
        """Check if orderbook is desynchronized (needs snapshot)."""
        if self.last_snapshot_at is None:
            return True
        
        # Check delta count
        if self.delta_count > max_delta_count:
            return True
        
        # Check snapshot age
        age_seconds = (datetime.now(timezone.utc) - self.last_snapshot_at).total_seconds()
        if age_seconds > max_age_seconds:
            return True
        
        return False
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True  # Allow SortedDict
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            SortedDict: lambda v: dict(v),
        }

