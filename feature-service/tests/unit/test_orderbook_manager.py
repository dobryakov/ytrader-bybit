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
        
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook is not None
        assert orderbook.symbol == "BTCUSDT"
        assert orderbook.sequence == 1000
    
    def test_apply_delta_update(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying update delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        delta = sample_orderbook_deltas[0]  # update delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1001
    
    def test_apply_delta_insert(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying insert delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        delta = sample_orderbook_deltas[1]  # insert delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert 50005.0 in orderbook.asks
    
    def test_apply_delta_delete(self, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test applying delete delta."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        delta = sample_orderbook_deltas[2]  # delete delta
        result = manager.apply_delta(delta)
        
        assert result is True
        orderbook = manager.get_orderbook("BTCUSDT")
        assert 49997.0 not in orderbook.bids
    
    def test_apply_delta_sequence_gap(self, sample_orderbook_snapshot):
        """Test applying delta with sequence gap."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
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
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        # Initially should not be desynchronized
        assert manager.is_desynchronized("BTCUSDT") is False
        
        # Make it desynchronized by setting high delta count
        orderbook = manager.get_orderbook("BTCUSDT")
        orderbook.delta_count = 2000
        
        assert manager.is_desynchronized("BTCUSDT") is True
    
    def test_get_mid_price(self, sample_orderbook_snapshot):
        """Test getting mid price."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        mid_price = manager.get_mid_price("BTCUSDT")
        
        assert mid_price is not None
        assert mid_price > 0
    
    def test_get_spread(self, sample_orderbook_snapshot):
        """Test getting spread."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        spread = manager.get_spread("BTCUSDT")
        
        assert spread is not None
        assert spread > 0
    
    def test_get_depth(self, sample_orderbook_snapshot):
        """Test getting orderbook depth."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        depth_bid = manager.get_depth("BTCUSDT", "bid", top_n=5)
        depth_ask = manager.get_depth("BTCUSDT", "ask", top_n=5)
        
        assert depth_bid > 0
        assert depth_ask > 0
    
    def test_get_imbalance(self, sample_orderbook_snapshot):
        """Test getting orderbook imbalance."""
        manager = OrderbookManager()
        manager.apply_snapshot(sample_orderbook_snapshot)
        
        imbalance = manager.get_imbalance("BTCUSDT", top_n=5)
        
        assert -1.0 <= imbalance <= 1.0

