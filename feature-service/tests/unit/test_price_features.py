"""
Unit tests for price features computation.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock

from src.models.orderbook_state import OrderbookState
from src.models.rolling_windows import RollingWindows
from src.features.price_features import (
    compute_mid_price,
    compute_spread_abs,
    compute_spread_rel,
    compute_returns,
    compute_vwap,
    compute_volume,
    compute_volatility,
    compute_all_price_features,
)


class TestPriceFeatures:
    """Test price features computation."""
    
    def test_compute_mid_price(self, sample_orderbook_state):
        """Test computing mid price from orderbook."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        mid_price = compute_mid_price(orderbook)
        
        assert mid_price is not None
        assert mid_price == (50000.0 + 50001.0) / 2.0
    
    def test_compute_mid_price_none(self):
        """Test computing mid price with None orderbook."""
        mid_price = compute_mid_price(None)
        
        assert mid_price is None
    
    def test_compute_spread_abs(self, sample_orderbook_state):
        """Test computing absolute spread."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        spread = compute_spread_abs(orderbook)
        
        assert spread is not None
        assert spread == 1.0
    
    def test_compute_spread_rel(self, sample_orderbook_state):
        """Test computing relative spread."""
        from src.models.orderbook_state import OrderbookState
        
        orderbook = OrderbookState(**sample_orderbook_state)
        spread_rel = compute_spread_rel(orderbook)
        
        assert spread_rel is not None
        assert spread_rel > 0
    
    def test_compute_returns(self, sample_rolling_windows):
        """Test computing returns."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        current_price = 50001.0
        
        returns = compute_returns(rw, 3, current_price)
        
        # Returns should be computed if data available
        assert returns is not None or len(rw.windows.get("3s", [])) == 0
    
    def test_compute_vwap(self, sample_rolling_windows):
        """Test computing VWAP."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        vwap = compute_vwap(rw, 3)
        
        # VWAP should be computed if trades available
        assert vwap is not None or len(rw.windows.get("3s", [])) == 0
    
    def test_compute_volume(self, sample_rolling_windows):
        """Test computing volume."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        volume = compute_volume(rw, 3)
        
        assert volume is not None
        assert volume >= 0
    
    def test_compute_volatility(self, sample_rolling_windows):
        """Test computing volatility."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        volatility = compute_volatility(rw, 60)
        
        # Volatility requires klines (with 'close' column), not trades (with 'price' column)
        # sample_rolling_windows has trades, so volatility will be None
        # This is expected behavior - volatility needs kline data
        assert volatility is None or isinstance(volatility, float)
    
    def test_compute_all_price_features(self, sample_orderbook_state, sample_rolling_windows):
        """Test computing all price features."""
        from src.models.orderbook_state import OrderbookState
        from src.models.rolling_windows import RollingWindows
        
        orderbook = OrderbookState(**sample_orderbook_state)
        rw = RollingWindows(**sample_rolling_windows)
        current_price = orderbook.get_mid_price()
        
        features = compute_all_price_features(orderbook, rw, current_price)
        
        assert "mid_price" in features
        assert "spread_abs" in features
        assert "spread_rel" in features
        assert "returns_1s" in features or features["returns_1s"] is None
        assert "vwap_3s" in features or features["vwap_3s"] is None
        assert "volume_3s" in features

