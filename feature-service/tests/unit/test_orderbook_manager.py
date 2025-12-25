"""
Unit tests for Orderbook Manager service.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.services.orderbook_manager import OrderbookManager
from src.models.orderbook_state import OrderbookState


class TestOrderbookManager:
    """Test Orderbook Manager service."""
    
    def test_orderbook_manager_init(self):
        """Test initializing orderbook manager."""
        manager = OrderbookManager()
        
        assert manager._orderbooks == {}
    
    def test_apply_snapshot(self, sample_orderbook_snapshot):
        """Test applying orderbook snapshot."""
        manager = OrderbookManager()
        
        # Apply snapshot immediately (not buffered) for backward compatibility
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook is not None
        assert orderbook.symbol == "BTCUSDT"
        assert orderbook.sequence == 1000
    
    def test_apply_delta_update(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying update delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        delta = sample_orderbook_deltas[0]  # update delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1001
    
    def test_apply_delta_insert(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying insert delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        # First apply update delta (1001) to get sequence in sync
        manager.apply_delta(sample_orderbook_deltas[0])
        
        # Then apply insert delta (1002)
        delta = sample_orderbook_deltas[1]  # insert delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert 50005.0 in orderbook.asks
    
    def test_apply_delta_delete(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying delete delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        # First apply update (1001) and insert (1002) deltas to get sequence in sync
        manager.apply_delta(sample_orderbook_deltas[0])
        manager.apply_delta(sample_orderbook_deltas[1])
        
        # Then apply delete delta (1003)
        delta = sample_orderbook_deltas[2]  # delete delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert 49997.0 not in orderbook.bids
    
    def test_apply_delta_sequence_gap(self, sample_orderbook_snapshot):
        """Test applying delta with sequence gap."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        # Delta with sequence gap
        delta = {
            "symbol": "BTCUSDT",
            "sequence": 1005,  # Gap: expected 1001
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,
            "timestamp": datetime.now(timezone.utc),
        }
        
        result = manager.apply_delta(delta)
        
        assert result is False  # Should fail due to sequence gap
    
    def test_apply_delta_no_state(self, sample_orderbook_deltas):
        """Test applying delta when no orderbook state exists."""
        manager = OrderbookManager()
        
        delta = sample_orderbook_deltas[0]
        result = manager.apply_delta(delta)
        
        assert result is False
    
    def test_is_desynchronized(self, sample_orderbook_snapshot):
        """Test checking if orderbook is desynchronized."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        # Initially should not be desynchronized
        assert manager.is_desynchronized("BTCUSDT") is False
        
        # Make it desynchronized by setting high delta count
        orderbook = manager.get_orderbook("BTCUSDT")
        orderbook.delta_count = 2000
        
        assert manager.is_desynchronized("BTCUSDT") is True
    
    def test_get_mid_price(self, sample_orderbook_snapshot):
        """Test getting mid price."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        mid_price = manager.get_mid_price("BTCUSDT")
        
        assert mid_price is not None
        assert mid_price > 0
    
    def test_get_spread(self, sample_orderbook_snapshot):
        """Test getting spread."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        spread = manager.get_spread("BTCUSDT")
        
        assert spread is not None
        assert spread > 0
    
    def test_get_depth(self, sample_orderbook_snapshot):
        """Test getting orderbook depth."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        depth_bid = manager.get_depth("BTCUSDT", "bid", top_n=5)
        depth_ask = manager.get_depth("BTCUSDT", "ask", top_n=5)
        
        assert depth_bid > 0
        assert depth_ask > 0
    
    def test_get_imbalance(self, sample_orderbook_snapshot):
        """Test getting orderbook imbalance."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        imbalance = manager.get_imbalance("BTCUSDT", top_n=5)
        
        assert -1.0 <= imbalance <= 1.0
    
    def test_apply_delta_buffered(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test buffered delta application."""
        manager = OrderbookManager(enable_delta_batching=True)
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        delta = sample_orderbook_deltas[0]  # update delta
        manager.apply_delta_buffered(delta)
        
        # Delta should be in buffer, not applied yet
        assert manager.has_pending_deltas("BTCUSDT") is True
        assert manager.get_pending_delta_count("BTCUSDT") == 1
        
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1000  # Not updated yet
    
    def test_apply_delta_batch(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test batch delta application."""
        manager = OrderbookManager(enable_delta_batching=True)
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        # Add multiple deltas to buffer
        delta1 = sample_orderbook_deltas[0]  # sequence 1001
        delta2 = sample_orderbook_deltas[1]  # sequence 1002 (needs 1001 applied first)
        manager.apply_delta_buffered(delta1)
        manager.apply_delta_buffered(delta2)
        
        assert manager.get_pending_delta_count("BTCUSDT") == 2
        
        # Apply batch
        applied_count = manager.apply_buffered_updates("BTCUSDT")
        
        assert applied_count == 2
        assert manager.has_pending_deltas("BTCUSDT") is False
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1002  # Both deltas applied
    
    def test_apply_delta_buffered_disabled(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test that buffered delta application immediately applies when batching is disabled."""
        manager = OrderbookManager(enable_delta_batching=False)
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)
        
        delta = sample_orderbook_deltas[0]
        manager.apply_delta_buffered(delta)
        
        # Should be applied immediately (no batching)
        assert manager.has_pending_deltas("BTCUSDT") is False
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1001
    
    def test_apply_snapshot_clears_buffer(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test that applying snapshot clears delta buffer."""
        manager = OrderbookManager(enable_delta_batching=True)
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=False)  # Apply first snapshot immediately
        
        # Add delta to buffer
        delta = sample_orderbook_deltas[0]
        manager.apply_delta_buffered(delta)
        assert manager.has_pending_deltas("BTCUSDT") is True
        
        # Buffer snapshot - should clear delta buffer immediately
        manager.apply_snapshot(sample_orderbook_snapshot, buffered=True)
        assert manager.has_pending_deltas("BTCUSDT") is False  # Delta buffer cleared
        assert manager.has_pending_updates("BTCUSDT") is True  # But snapshot is buffered
    
    def test_apply_snapshot_buffered(self, sample_orderbook_snapshot):
        """Test that snapshots are buffered and only latest is kept."""
        manager = OrderbookManager(enable_delta_batching=True)
        
        # Buffer multiple snapshots - only latest should be kept
        snapshot1 = {**sample_orderbook_snapshot, "sequence": 1000}
        snapshot2 = {**sample_orderbook_snapshot, "sequence": 2000}
        snapshot3 = {**sample_orderbook_snapshot, "sequence": 3000}
        
        manager.apply_snapshot(snapshot1, buffered=True)
        manager.apply_snapshot(snapshot2, buffered=True)
        manager.apply_snapshot(snapshot3, buffered=True)
        
        # Snapshot should be buffered, not applied yet
        assert manager.has_pending_updates("BTCUSDT") is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook is None  # Not applied yet
        
        # Apply buffered updates - should apply only latest snapshot (sequence 3000)
        manager.apply_buffered_updates("BTCUSDT")
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook is not None
        assert orderbook.sequence == 3000  # Latest snapshot applied

