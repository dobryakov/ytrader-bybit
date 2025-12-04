"""
Orderbook Manager service for managing orderbook state and reconstruction.
"""
from datetime import datetime, timezone
from typing import Dict, Optional
import structlog

from src.models.orderbook_state import OrderbookState

logger = structlog.get_logger(__name__)


class OrderbookManager:
    """Manages orderbook state for multiple symbols."""
    
    def __init__(self):
        """Initialize orderbook manager."""
        self._orderbooks: Dict[str, OrderbookState] = {}
        self._max_delta_count = 1000
        self._max_snapshot_age_seconds = 60
    
    def get_orderbook(self, symbol: str) -> Optional[OrderbookState]:
        """Get orderbook state for symbol."""
        return self._orderbooks.get(symbol)
    
    def apply_snapshot(self, snapshot: Dict) -> None:
        """Apply orderbook snapshot to initialize or reset orderbook state."""
        symbol = snapshot["symbol"]
        
        orderbook = OrderbookState.from_snapshot(snapshot)
        self._orderbooks[symbol] = orderbook
        
        logger.info(
            "orderbook_snapshot_applied",
            symbol=symbol,
            sequence=orderbook.sequence,
            bids_count=len(orderbook.bids),
            asks_count=len(orderbook.asks),
        )
    
    def apply_delta(self, delta: Dict) -> bool:
        """Apply delta to orderbook state. Returns True if applied, False if desynchronized."""
        symbol = delta["symbol"]
        orderbook = self._orderbooks.get(symbol)
        
        if orderbook is None:
            logger.warning(
                "orderbook_delta_no_state",
                symbol=symbol,
                sequence=delta.get("sequence"),
                message="No orderbook state exists, need snapshot first",
            )
            return False
        
        # Check for sequence gap
        next_sequence = delta["sequence"]
        if orderbook.has_sequence_gap(next_sequence):
            logger.warning(
                "orderbook_sequence_gap",
                symbol=symbol,
                expected_sequence=orderbook.sequence + 1,
                received_sequence=next_sequence,
                message="Sequence gap detected, orderbook may be desynchronized",
            )
            # Mark as desynchronized
            orderbook.last_snapshot_at = None
            return False
        
        # Apply delta
        orderbook.apply_delta(delta)
        
        # Check if desynchronized
        if orderbook.is_desynchronized(self._max_delta_count, self._max_snapshot_age_seconds):
            logger.warning(
                "orderbook_desynchronized",
                symbol=symbol,
                delta_count=orderbook.delta_count,
                snapshot_age_seconds=(datetime.now(timezone.utc) - orderbook.last_snapshot_at).total_seconds() if orderbook.last_snapshot_at else None,
                message="Orderbook desynchronized, snapshot needed",
            )
            return False
        
        return True
    
    def is_desynchronized(self, symbol: str) -> bool:
        """Check if orderbook for symbol is desynchronized."""
        orderbook = self._orderbooks.get(symbol)
        if orderbook is None:
            return True
        
        return orderbook.is_desynchronized(self._max_delta_count, self._max_snapshot_age_seconds)
    
    def request_snapshot(self, symbol: str) -> None:
        """Request snapshot for symbol (to be handled by consumer)."""
        logger.info(
            "orderbook_snapshot_requested",
            symbol=symbol,
            message="Snapshot requested for desynchronized orderbook",
        )
        # This will trigger snapshot request via ws-gateway API
        # Implementation in market_data_consumer.py
    
    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price for symbol."""
        orderbook = self._orderbooks.get(symbol)
        if orderbook is None:
            return None
        
        return orderbook.get_mid_price()
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """Get spread for symbol."""
        orderbook = self._orderbooks.get(symbol)
        if orderbook is None:
            return None
        
        return orderbook.get_spread_abs()
    
    def get_depth(self, symbol: str, side: str, top_n: int = 5) -> float:
        """Get orderbook depth for symbol."""
        orderbook = self._orderbooks.get(symbol)
        if orderbook is None:
            return 0.0
        
        if side == "bid":
            if top_n == 5:
                return orderbook.get_depth_bid_top5()
            elif top_n == 10:
                return orderbook.get_depth_bid_top10()
        else:
            if top_n == 5:
                return orderbook.get_depth_ask_top5()
            elif top_n == 10:
                return orderbook.get_depth_ask_top10()
        
        return 0.0
    
    def get_imbalance(self, symbol: str, top_n: int = 5) -> float:
        """Get orderbook imbalance for symbol."""
        orderbook = self._orderbooks.get(symbol)
        if orderbook is None:
            return 0.0
        
        if top_n == 5:
            return orderbook.get_imbalance_top5()
        
        return 0.0

