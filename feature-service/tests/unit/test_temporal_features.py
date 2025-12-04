"""
Unit tests for temporal features computation.
"""
import pytest
from datetime import datetime, timezone
from src.features.temporal_features import (
    compute_time_of_day_sin,
    compute_time_of_day_cos,
    compute_all_temporal_features,
)


class TestTemporalFeatures:
    """Test temporal features computation."""
    
    def test_compute_time_of_day_sin(self):
        """Test computing time of day sine encoding."""
        timestamp = datetime.now(timezone.utc)
        
        sin_value = compute_time_of_day_sin(timestamp)
        
        assert -1.0 <= sin_value <= 1.0
    
    def test_compute_time_of_day_cos(self):
        """Test computing time of day cosine encoding."""
        timestamp = datetime.now(timezone.utc)
        
        cos_value = compute_time_of_day_cos(timestamp)
        
        assert -1.0 <= cos_value <= 1.0
    
    def test_temporal_features_cyclic(self):
        """Test that temporal features are cyclic (24-hour period)."""
        # Test that 0:00 and 24:00 (next day 0:00) produce same values
        timestamp1 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamp2 = datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        
        sin1 = compute_time_of_day_sin(timestamp1)
        sin2 = compute_time_of_day_sin(timestamp2)
        
        assert abs(sin1 - sin2) < 0.0001  # Should be approximately equal
    
    def test_compute_all_temporal_features(self):
        """Test computing all temporal features."""
        timestamp = datetime.now(timezone.utc)
        
        features = compute_all_temporal_features(timestamp)
        
        assert "time_of_day_sin" in features
        assert "time_of_day_cos" in features
        assert -1.0 <= features["time_of_day_sin"] <= 1.0
        assert -1.0 <= features["time_of_day_cos"] <= 1.0

