"""
Unit tests for orderflow features computation.
"""
import pytest
from src.models.rolling_windows import RollingWindows
from src.features.orderflow_features import (
    compute_signed_volume,
    compute_buy_sell_volume_ratio,
    compute_trade_count,
    compute_net_aggressor_pressure,
    compute_all_orderflow_features,
)


class TestOrderflowFeatures:
    """Test orderflow features computation."""
    
    def test_compute_signed_volume(self, sample_rolling_windows):
        """Test computing signed volume."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        signed_volume = compute_signed_volume(rw, 3)
        
        assert signed_volume is not None
    
    def test_compute_buy_sell_volume_ratio(self, sample_rolling_windows):
        """Test computing buy/sell volume ratio."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        ratio = compute_buy_sell_volume_ratio(rw, 3)
        
        assert ratio is not None
        assert ratio >= 0
    
    def test_compute_trade_count(self, sample_rolling_windows):
        """Test computing trade count."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        count = compute_trade_count(rw, 3)
        
        assert isinstance(count, int)
        assert count >= 0
    
    def test_compute_net_aggressor_pressure(self, sample_rolling_windows):
        """Test computing net aggressor pressure."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        pressure = compute_net_aggressor_pressure(rw, 3)
        
        assert pressure is not None
        assert -1.0 <= pressure <= 1.0
    
    def test_compute_all_orderflow_features(self, sample_rolling_windows):
        """Test computing all orderflow features."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        
        features = compute_all_orderflow_features(rw)
        
        assert "signed_volume_3s" in features
        assert "buy_sell_volume_ratio" in features
        assert "trade_count_3s" in features
        assert "net_aggressor_pressure" in features

