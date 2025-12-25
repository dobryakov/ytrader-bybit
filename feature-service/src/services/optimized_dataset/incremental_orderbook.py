"""
Incremental Orderbook Manager for optimized dataset building.

Manages orderbook state with incremental updates for streaming processing.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import pandas as pd
from copy import deepcopy
import structlog

from src.models.orderbook_state import OrderbookState
from src.services.orderbook_manager import OrderbookManager

logger = structlog.get_logger(__name__)


class IncrementalOrderbookManager:
    """
    Manages orderbook state with incremental updates for streaming processing.
    
    Optimized for dataset building where we process timestamps sequentially
    and can reuse orderbook state between timestamps.
    """
    
    def __init__(
        self,
        symbol: str,
        snapshot_refresh_interval: int = 3600,  # 1 hour
        max_delta_count: int = 1000,
    ):
        """
        Initialize incremental orderbook manager.
        
        Args:
            symbol: Trading pair symbol
            snapshot_refresh_interval: Snapshot refresh interval in seconds (default: 3600)
            max_delta_count: Maximum deltas before refresh (default: 1000)
        """
        self.symbol = symbol
        self.snapshot_refresh_interval = snapshot_refresh_interval
        self.max_delta_count = max_delta_count
        
        # Current orderbook state
        self.current_state: Optional[OrderbookState] = None
        self.last_snapshot_time: Optional[datetime] = None
        self.last_processed_timestamp: Optional[datetime] = None
        
        # Orderbook manager for snapshot application
        # Disable batching for incremental orderbook (data is processed sequentially from files)
        self._orderbook_manager = OrderbookManager(enable_delta_batching=False)
        
        logger.info(
            "incremental_orderbook_manager_initialized",
            symbol=symbol,
            snapshot_refresh_interval=snapshot_refresh_interval,
            max_delta_count=max_delta_count,
        )
    
    def update_to_timestamp(
        self,
        timestamp: datetime,
        snapshots: pd.DataFrame,
        deltas: pd.DataFrame,
    ) -> Optional[OrderbookState]:
        """
        Update orderbook state to specified timestamp incrementally.
        
        Args:
            timestamp: Target timestamp
            snapshots: DataFrame with orderbook snapshots
            deltas: DataFrame with orderbook deltas
            
        Returns:
            OrderbookState at timestamp, or None if insufficient data
        """
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # If no current state, do full reconstruction
        if self.current_state is None or self.last_processed_timestamp is None:
            return self._reconstruct_full(timestamp, snapshots, deltas)
        
        # Check if snapshot refresh is needed
        if self._needs_snapshot_refresh(timestamp):
            logger.debug(
                "orderbook_snapshot_refresh_needed",
                symbol=self.symbol,
                timestamp=timestamp.isoformat(),
                last_snapshot_at=(
                    self.last_snapshot_time.isoformat()
                    if self.last_snapshot_time
                    else None
                ),
                delta_count=self.current_state.delta_count,
            )
            # Do full reconstruction with refresh
            return self._reconstruct_full(timestamp, snapshots, deltas)
        
        # Incremental update: apply only new deltas
        return self._update_incremental(timestamp, deltas)
    
    def _reconstruct_full(
        self,
        timestamp: datetime,
        snapshots: pd.DataFrame,
        deltas: pd.DataFrame,
    ) -> Optional[OrderbookState]:
        """
        Fully reconstruct orderbook state at timestamp.
        
        Args:
            timestamp: Target timestamp
            snapshots: DataFrame with snapshots
            deltas: DataFrame with deltas
            
        Returns:
            OrderbookState or None
        """
        if snapshots.empty:
            return None
        
        # Find latest snapshot before or at timestamp
        snapshots_before = snapshots[snapshots["timestamp"] <= timestamp]
        if snapshots_before.empty:
            return None
        
        latest_snapshot = snapshots_before.iloc[-1]
        
        # Get timestamp - handle both Series and dict-like access
        if "timestamp" in latest_snapshot.index:
            snapshot_time = latest_snapshot["timestamp"]
        elif hasattr(latest_snapshot, "timestamp"):
            snapshot_time = latest_snapshot.timestamp
        else:
            # Try to get from Series index if it's a Series
            snapshot_time = latest_snapshot.get("timestamp")
            if snapshot_time is None:
                raise ValueError("timestamp column not found in snapshot")
        
        # Normalize timestamp
        if isinstance(snapshot_time, pd.Timestamp):
            snapshot_time = snapshot_time.to_pydatetime()
        if isinstance(snapshot_time, datetime):
            if snapshot_time.tzinfo is None:
                snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
            snapshot_time = snapshot_time.astimezone(timezone.utc)
        else:
            # Fallback: try to parse as string
            snapshot_time = pd.to_datetime(snapshot_time, utc=True).to_pydatetime()
        
        # Check if sequence column exists
        has_sequence = (
            "sequence" in snapshots.columns and "sequence" in deltas.columns
        )
        
        # Get deltas after snapshot
        if has_sequence:
            snapshot_sequence = latest_snapshot["sequence"] if "sequence" in latest_snapshot.index else None
            if snapshot_sequence is None or deltas.empty or "sequence" not in deltas.columns:
                deltas_after = pd.DataFrame()
            else:
                deltas_after = deltas[
                    (deltas["sequence"] > snapshot_sequence) &
                    (deltas["timestamp"] <= timestamp)
                ].sort_values("sequence")
        else:
            if deltas.empty or "timestamp" not in deltas.columns:
                deltas_after = pd.DataFrame()
            else:
                deltas_after = deltas[
                    (deltas["timestamp"] > snapshot_time) &
                    (deltas["timestamp"] <= timestamp)
                ].sort_values("timestamp")
        
        # Apply snapshot
        # Get values from Series correctly
        bids_val = latest_snapshot["bids"] if "bids" in latest_snapshot.index else []
        asks_val = latest_snapshot["asks"] if "asks" in latest_snapshot.index else []
        
        # Get sequence value - OrderbookState.from_snapshot requires sequence
        seq_val = 0  # Default
        if "sequence" in latest_snapshot.index:
            seq_val = latest_snapshot["sequence"]
        elif has_sequence and "sequence" in snapshots.columns:
            # Try to get from original DataFrame if not in Series
            seq_val = latest_snapshot.get("sequence", 0) if hasattr(latest_snapshot, "get") else 0
        
        snapshot_dict = {
            "event_type": "orderbook_snapshot",
            "symbol": self.symbol,
            "timestamp": snapshot_time.isoformat(),
            "sequence": int(seq_val),
            "bids": bids_val,
            "asks": asks_val,
        }
        
        # Apply snapshot immediately (not buffered) for incremental processing
        self._orderbook_manager.apply_snapshot(snapshot_dict, buffered=False)
        orderbook_state = self._orderbook_manager.get_orderbook(self.symbol)
        
        if orderbook_state is None:
            return None
        
        # Apply deltas
        for _, delta_row in deltas_after.iterrows():
            delta_dict = {
                "event_type": "orderbook_delta",
                "symbol": self.symbol,
                "timestamp": (
                    delta_row["timestamp"].isoformat()
                    if isinstance(delta_row["timestamp"], datetime)
                    else str(delta_row["timestamp"])
                ),
                "delta_type": delta_row.get("delta_type", "update"),
                "side": delta_row.get("side", "bid"),
                "price": float(delta_row.get("price", 0)),
                "quantity": float(delta_row.get("quantity", 0)),
            }
            
            if has_sequence and "sequence" in delta_row:
                delta_dict["sequence"] = int(delta_row["sequence"])
            
            orderbook_state.apply_delta(delta_dict)
        
        # Update state
        self.current_state = orderbook_state
        self.last_snapshot_time = snapshot_time
        self.last_processed_timestamp = timestamp
        
        return orderbook_state
    
    def _update_incremental(
        self,
        timestamp: datetime,
        deltas: pd.DataFrame,
    ) -> Optional[OrderbookState]:
        """
        Incrementally update orderbook state by applying only new deltas.
        
        Args:
            timestamp: Target timestamp
            deltas: DataFrame with deltas
            
        Returns:
            Updated OrderbookState
        """
        if self.current_state is None or self.last_processed_timestamp is None:
            return None
        
        # Create a copy to avoid mutating the original
        orderbook_state = deepcopy(self.current_state)
        
        # Get new deltas
        has_sequence = "sequence" in deltas.columns if not deltas.empty else False
        
        if has_sequence:
            last_sequence = self.current_state.sequence
            new_deltas = deltas[
                (deltas["sequence"] > last_sequence) &
                (deltas["timestamp"] > self.last_processed_timestamp) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("sequence")
        else:
            new_deltas = deltas[
                (deltas["timestamp"] > self.last_processed_timestamp) &
                (deltas["timestamp"] <= timestamp)
            ].sort_values("timestamp")
        
        # Apply new deltas
        for _, delta_row in new_deltas.iterrows():
            delta_dict = {
                "event_type": "orderbook_delta",
                "symbol": self.symbol,
                "timestamp": (
                    delta_row["timestamp"].isoformat()
                    if isinstance(delta_row["timestamp"], datetime)
                    else str(delta_row["timestamp"])
                ),
                "delta_type": delta_row.get("delta_type", "update"),
                "side": delta_row.get("side", "bid"),
                "price": float(delta_row.get("price", 0)),
                "quantity": float(delta_row.get("quantity", 0)),
            }
            
            if has_sequence and "sequence" in delta_row:
                delta_dict["sequence"] = int(delta_row["sequence"])
            
            orderbook_state.apply_delta(delta_dict)
        
        # Update state
        self.current_state = orderbook_state
        self.last_processed_timestamp = timestamp
        
        logger.debug(
            "orderbook_incremental_update",
            symbol=self.symbol,
            timestamp=timestamp.isoformat(),
            last_timestamp=(
                self.last_processed_timestamp.isoformat()
                if self.last_processed_timestamp
                else None
            ),
            new_deltas_count=len(new_deltas),
            delta_count=orderbook_state.delta_count,
        )
        
        return orderbook_state
    
    def _needs_snapshot_refresh(self, timestamp: datetime) -> bool:
        """
        Check if snapshot refresh is needed.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if refresh needed
        """
        if self.current_state is None:
            return True
        
        if self.last_snapshot_time is None:
            return True
        
        # Check time since last snapshot
        time_since_snapshot = (timestamp - self.last_snapshot_time).total_seconds()
        if time_since_snapshot >= self.snapshot_refresh_interval:
            return True
        
        # Check delta count
        if self.current_state.delta_count > self.max_delta_count:
            return True
        
        return False
    
    def get_current_state(self) -> Optional[OrderbookState]:
        """Get current orderbook state."""
        return self.current_state
    
    def reset(self) -> None:
        """Reset orderbook manager state."""
        self.current_state = None
        self.last_snapshot_time = None
        self.last_processed_timestamp = None
        logger.debug("incremental_orderbook_manager_reset", symbol=self.symbol)

