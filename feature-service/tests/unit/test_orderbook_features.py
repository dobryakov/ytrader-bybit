"""
Unit tests for orderbook features computation.
"""
import pytest
from src.models.orderbook_state import OrderbookState
from src.features.orderbook_features import (
    compute_depth_bid_top5,
    compute_depth_ask_top5,
    compute_depth_imbalance_top5,
    compute_all_orderbook_features,
)


class TestOrderbookFeatures:
    """Test orderbook features computation."""
    
    def test_compute_depth_bid_top5(self, sample_orderbook_state):
        """Test computing bid depth top 5."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        
        depth = compute_depth_bid_top5(orderbook)
        
        assert depth >= 0
    
    def test_compute_depth_ask_top5(self, sample_orderbook_state):
        """Test computing ask depth top 5."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        
        depth = compute_depth_ask_top5(orderbook)
        
        assert depth >= 0
    
    def test_compute_depth_imbalance_top5(self, sample_orderbook_state):
        """Test computing orderbook imbalance."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        
        imbalance = compute_depth_imbalance_top5(orderbook)
        
        assert -1.0 <= imbalance <= 1.0
    
    def test_compute_all_orderbook_features(self, sample_orderbook_state):
        """Test computing all orderbook features."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        
        features = compute_all_orderbook_features(orderbook)
        
        assert "depth_bid_top5" in features
        assert "depth_ask_top5" in features
        assert "depth_imbalance_top5" in features

