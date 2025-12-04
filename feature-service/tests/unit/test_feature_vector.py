"""
Unit tests for Feature Vector model.
"""
import pytest
from datetime import datetime, timezone
from tests.fixtures.feature_vectors import (
    sample_feature_vector,
    sample_feature_vector_minimal,
    sample_feature_vector_sequence,
)


class TestFeatureVector:
    """Test Feature Vector model."""
    
    def test_feature_vector_creation(self, sample_feature_vector):
        """Test creating a feature vector with all features."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert fv.symbol == "BTCUSDT"
        assert fv.timestamp is not None
        assert len(fv.features) > 0
        assert fv.feature_registry_version == "1.0.0"
        assert fv.trace_id == "test-trace-123"
    
    def test_feature_vector_minimal(self, sample_feature_vector_minimal):
        """Test creating a minimal feature vector."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector_minimal)
        
        assert fv.symbol == "BTCUSDT"
        assert "mid_price" in fv.features
        assert fv.feature_registry_version == "1.0.0"
    
    def test_feature_vector_price_features(self, sample_feature_vector):
        """Test price features are present."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert "mid_price" in fv.features
        assert "spread_abs" in fv.features
        assert "spread_rel" in fv.features
        assert "returns_1s" in fv.features
        assert "vwap_3s" in fv.features
        assert "volatility_1m" in fv.features
    
    def test_feature_vector_orderflow_features(self, sample_feature_vector):
        """Test orderflow features are present."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert "signed_volume_3s" in fv.features
        assert "buy_sell_volume_ratio" in fv.features
        assert "trade_count_3s" in fv.features
        assert "net_aggressor_pressure" in fv.features
    
    def test_feature_vector_orderbook_features(self, sample_feature_vector):
        """Test orderbook features are present."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert "depth_bid_top5" in fv.features
        assert "depth_ask_top5" in fv.features
        assert "depth_imbalance_top5" in fv.features
    
    def test_feature_vector_perpetual_features(self, sample_feature_vector):
        """Test perpetual features are present."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert "funding_rate" in fv.features
        assert "time_to_funding" in fv.features
    
    def test_feature_vector_temporal_features(self, sample_feature_vector):
        """Test temporal features are present."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        
        assert "time_of_day_sin" in fv.features
        assert "time_of_day_cos" in fv.features
    
    def test_feature_vector_serialization(self, sample_feature_vector):
        """Test feature vector can be serialized to dict."""
        from src.models.feature_vector import FeatureVector
        
        fv = FeatureVector(**sample_feature_vector)
        data = fv.model_dump()
        
        assert isinstance(data, dict)
        assert "symbol" in data
        assert "features" in data
        assert "timestamp" in data
    
    def test_feature_vector_sequence(self, sample_feature_vector_sequence):
        """Test sequence of feature vectors."""
        from src.models.feature_vector import FeatureVector
        
        fvs = [FeatureVector(**fv_data) for fv_data in sample_feature_vector_sequence]
        
        assert len(fvs) == 3
        assert all(fv.symbol == "BTCUSDT" for fv in fvs)
        assert fvs[0].features["mid_price"] < fvs[2].features["mid_price"]

