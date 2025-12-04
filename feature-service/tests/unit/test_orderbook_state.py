"""
Unit tests for Orderbook State model.
"""
import pytest
from datetime import datetime, timezone, timedelta
from sortedcontainers import SortedDict
from tests.fixtures.orderbook import (
    sample_orderbook_state,
    sample_orderbook_snapshot,
    sample_orderbook_deltas,
)


class TestOrderbookState:
    """Test Orderbook State model."""
    
    def test_orderbook_state_creation(self, sample_orderbook_state):
        """Test creating an orderbook state."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        assert state.symbol == "BTCUSDT"
        assert state.sequence == 1000
        assert len(state.bids) > 0
        assert len(state.asks) > 0
        assert state.delta_count == 0
    
    def test_orderbook_state_from_snapshot(self, sample_orderbook_snapshot):
        """Test initializing orderbook state from snapshot."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState.from_snapshot(sample_orderbook_snapshot)
        
        assert state.symbol == "BTCUSDT"
        assert state.sequence == 1000
        assert len(state.bids) == 4
        assert len(state.asks) == 4
        assert state.last_snapshot_at is not None
    
    def test_orderbook_state_apply_delta_update(self, sample_orderbook_state, sample_orderbook_deltas):
        """Test applying update delta to orderbook state."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        delta = sample_orderbook_deltas[0]  # update delta
        
        initial_quantity = state.bids.get(50000.0, 0)
        state.apply_delta(delta)
        
        assert state.sequence == 1001
        assert state.bids[50000.0] == 1.8  # Updated quantity
        assert state.delta_count == 1
    
    def test_orderbook_state_apply_delta_insert(self, sample_orderbook_state, sample_orderbook_deltas):
        """Test applying insert delta to orderbook state."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        delta = sample_orderbook_deltas[1]  # insert delta
        
        state.apply_delta(delta)
        
        assert state.sequence == 1002
        assert 50005.0 in state.asks
        assert state.asks[50005.0] == 1.0
        assert state.delta_count == 1
    
    def test_orderbook_state_apply_delta_delete(self, sample_orderbook_state, sample_orderbook_deltas):
        """Test applying delete delta to orderbook state."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        delta = sample_orderbook_deltas[2]  # delete delta
        
        assert 49997.0 in state.bids
        state.apply_delta(delta)
        
        assert state.sequence == 1003
        assert 49997.0 not in state.bids
        assert state.delta_count == 1
    
    def test_orderbook_state_get_best_bid(self, sample_orderbook_state):
        """Test getting best bid price."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        best_bid = state.get_best_bid()
        
        assert best_bid == 50000.0  # Highest bid price
    
    def test_orderbook_state_get_best_ask(self, sample_orderbook_state):
        """Test getting best ask price."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        best_ask = state.get_best_ask()
        
        assert best_ask == 50001.0  # Lowest ask price
    
    def test_orderbook_state_get_mid_price(self, sample_orderbook_state):
        """Test getting mid price."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        mid_price = state.get_mid_price()
        
        assert mid_price == (50000.0 + 50001.0) / 2
    
    def test_orderbook_state_get_spread(self, sample_orderbook_state):
        """Test getting spread."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        spread_abs = state.get_spread_abs()
        spread_rel = state.get_spread_rel()
        
        assert spread_abs == 1.0
        assert spread_rel == 1.0 / 50000.5  # spread / mid_price
    
    def test_orderbook_state_get_depth(self, sample_orderbook_state):
        """Test getting orderbook depth."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        depth_bid_top5 = state.get_depth_bid_top5()
        depth_ask_top5 = state.get_depth_ask_top5()
        
        assert depth_bid_top5 > 0
        assert depth_ask_top5 > 0
    
    def test_orderbook_state_get_imbalance(self, sample_orderbook_state):
        """Test getting orderbook imbalance."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        imbalance = state.get_imbalance_top5()
        
        assert -1.0 <= imbalance <= 1.0  # Normalized imbalance
    
    def test_orderbook_state_sequence_gap_detection(self, sample_orderbook_state):
        """Test detecting sequence gaps."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        
        # Next expected sequence is 1001, but we receive 1003
        delta = {
            "sequence": 1003,
            "delta_type": "update",
            "side": "bid",
            "price": 50000.0,
            "quantity": 1.8,
        }
        
        has_gap = state.has_sequence_gap(delta["sequence"])
        
        assert has_gap is True
    
    def test_orderbook_state_is_desynchronized(self, sample_orderbook_state):
        """Test checking if orderbook is desynchronized."""
        from src.models.orderbook_state import OrderbookState
        
        state = OrderbookState(**sample_orderbook_state)
        state.delta_count = 1000  # High delta count
        state.last_snapshot_at = datetime.now(timezone.utc).replace(second=0) - timedelta(seconds=70)  # Old snapshot (>60s)
        
        is_desync = state.is_desynchronized(max_delta_count=500, max_age_seconds=5)
        
        assert is_desync is True

