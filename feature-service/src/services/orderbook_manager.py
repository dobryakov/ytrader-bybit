"""
Orderbook Manager service for managing orderbook state and reconstruction.
"""
from datetime import datetime, timezone
from typing import Dict, Optional, List
from collections import defaultdict
import structlog
import threading

from src.models.orderbook_state import OrderbookState

logger = structlog.get_logger(__name__)


class OrderbookManager:
    """Manages orderbook state for multiple symbols."""
    
    def __init__(self, enable_delta_batching: bool = True, max_buffer_size: int = 5000):
        """
        Initialize orderbook manager.
        
        Args:
            enable_delta_batching: If True, deltas are buffered and applied in batches
            max_buffer_size: Maximum number of deltas to buffer per symbol before forced application
        """
        self._orderbooks: Dict[str, OrderbookState] = {}
        self._max_delta_count = 1000
        self._max_snapshot_age_seconds = 60
        self._enable_delta_batching = enable_delta_batching
        self._max_buffer_size = max_buffer_size
        # Buffer for accumulating deltas (performance optimization)
        self._delta_buffer: Dict[str, List[Dict]] = defaultdict(list)
        # Buffer for snapshots - store only the latest snapshot per symbol (snapshots replace state)
        self._snapshot_buffer: Dict[str, Optional[Dict]] = {}
        self._buffer_locks: Dict[str, threading.Lock] = {}
    
    def _get_lock(self, symbol: str) -> threading.Lock:
        """Get or create threading.Lock for symbol."""
        if symbol not in self._buffer_locks:
            self._buffer_locks[symbol] = threading.Lock()
        return self._buffer_locks[symbol]
    
    def get_orderbook(self, symbol: str) -> Optional[OrderbookState]:
        """Get orderbook state for symbol."""
        return self._orderbooks.get(symbol)
    
    def apply_snapshot(self, snapshot: Dict, buffered: bool = True) -> None:
        """
        Apply orderbook snapshot to initialize or reset orderbook state.
        
        Args:
            snapshot: Snapshot dictionary
            buffered: If True and batching is enabled, buffer snapshot instead of applying immediately
        """
        symbol = snapshot["symbol"]
        
        if buffered and self._enable_delta_batching:
            # Buffer snapshot (only latest is kept, as each snapshot replaces state)
            # Clear delta buffer immediately since snapshot makes all previous deltas invalid
            lock = self._get_lock(symbol)
            with lock:
                self._snapshot_buffer[symbol] = snapshot
                # Clear delta buffer - snapshot replaces state, so previous deltas are invalid
                if symbol in self._delta_buffer:
                    self._delta_buffer[symbol].clear()
            logger.debug(
                "orderbook_snapshot_buffered",
                symbol=symbol,
                sequence=snapshot.get("sequence"),
            )
        else:
            # Apply immediately
            self._apply_snapshot_immediate(snapshot)
    
    def _apply_snapshot_immediate(self, snapshot: Dict) -> None:
        """Apply snapshot immediately (internal method)."""
        symbol = snapshot["symbol"]
        
        orderbook = OrderbookState.from_snapshot(snapshot)
        self._orderbooks[symbol] = orderbook
        
        # Clear delta buffer when snapshot is applied (snapshot resets state)
        if self._enable_delta_batching and symbol in self._delta_buffer:
            self._delta_buffer[symbol].clear()
        
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
    
    def apply_delta_buffered(self, delta: Dict) -> None:
        """
        Add delta to buffer instead of immediate application (performance optimization).
        
        Args:
            delta: Orderbook delta dictionary
        """
        if not self._enable_delta_batching:
            # If batching is disabled, apply immediately
            self.apply_delta(delta)
            return
        
        symbol = delta["symbol"]
        lock = self._get_lock(symbol)
        with lock:
            self._delta_buffer[symbol].append(delta)
            
            # If buffer is too large, apply batch to prevent memory issues
            if len(self._delta_buffer[symbol]) > self._max_buffer_size:
                logger.warning(
                    "orderbook_delta_buffer_overflow",
                    symbol=symbol,
                    buffer_size=len(self._delta_buffer[symbol]),
                    max_size=self._max_buffer_size,
                    message="Buffer size exceeded, applying batch",
                )
                self.apply_delta_batch(symbol)
    
    def apply_buffered_updates(self, symbol: str) -> int:
        """
        Apply all buffered updates (snapshot and deltas) for a symbol.
        Snapshot is applied first (replaces state), then deltas if any.
        Returns number of successfully applied deltas.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of deltas successfully applied (after snapshot if present)
        """
        if not self._enable_delta_batching:
            return 0
        
        lock = self._get_lock(symbol)
        with lock:
            # Apply snapshot first if present (replaces state, clears delta buffer)
            snapshot = self._snapshot_buffer.pop(symbol, None)
            if snapshot:
                self._apply_snapshot_immediate(snapshot)
                # Snapshot clears delta buffer, so return 0 deltas applied
                return 0
            
            # Apply deltas if no snapshot
            return self._apply_delta_batch_unsafe(symbol)
    
    def _apply_delta_batch_unsafe(self, symbol: str) -> int:
        """Apply delta batch without lock (caller must hold lock)."""
        if not self._delta_buffer[symbol]:
            return 0
        
        deltas = self._delta_buffer[symbol].copy()
        self._delta_buffer[symbol].clear()
        
        # Release lock before applying (to avoid holding lock during expensive operations)
        # But we need to apply deltas one by one, so we'll do it here
        applied_count = 0
        failed_count = 0
        
        for delta in deltas:
            try:
                if self.apply_delta(delta):
                    applied_count += 1
                else:
                    failed_count += 1
                    # If delta application failed, stop processing batch
                    logger.warning(
                        "orderbook_batch_application_failed",
                        symbol=symbol,
                        applied_count=applied_count,
                        failed_count=failed_count,
                        total_count=len(deltas),
                        failed_sequence=delta.get("sequence"),
                        message="Delta application failed, stopping batch",
                    )
                    if self.is_desynchronized(symbol):
                        self.request_snapshot(symbol)
                    break
            except Exception as e:
                logger.error(
                    "orderbook_batch_application_error",
                    symbol=symbol,
                    applied_count=applied_count,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                failed_count += 1
                break
        
        if applied_count > 0:
            logger.debug(
                "orderbook_batch_applied",
                symbol=symbol,
                applied_count=applied_count,
                total_count=len(deltas),
            )
        
        return applied_count
    
    def apply_delta_batch(self, symbol: str) -> int:
        """
        Apply all buffered deltas for a symbol. Returns number of successfully applied deltas.
        
        Note: Use apply_buffered_updates() instead to handle both snapshots and deltas.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of deltas successfully applied
        """
        if not self._enable_delta_batching:
            return 0
        
        lock = self._get_lock(symbol)
        with lock:
            return self._apply_delta_batch_unsafe(symbol)
    
    def has_pending_updates(self, symbol: str) -> bool:
        """
        Check if there are pending updates (snapshot or deltas) in buffer for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if there are pending updates, False otherwise
        """
        if not self._enable_delta_batching:
            return False
        return symbol in self._snapshot_buffer or bool(self._delta_buffer.get(symbol))
    
    def has_pending_deltas(self, symbol: str) -> bool:
        """
        Check if there are pending deltas in buffer for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if there are pending deltas, False otherwise
        """
        if not self._enable_delta_batching:
            return False
        return bool(self._delta_buffer.get(symbol))
    
    def get_pending_delta_count(self, symbol: str) -> int:
        """
        Get number of pending deltas in buffer for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of pending deltas
        """
        if not self._enable_delta_batching:
            return 0
        return len(self._delta_buffer.get(symbol, []))

