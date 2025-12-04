"""
Unit tests for perpetual features computation.
"""
import pytest
from datetime import datetime, timezone
from src.models.rolling_windows import RollingWindows
from src.features.perpetual_features import (
    compute_funding_rate,
    compute_time_to_funding,
    compute_all_perpetual_features,
)


class TestPerpetualFeatures:
    """Test perpetual features computation."""
    
    def test_compute_all_perpetual_features_with_data(self, sample_rolling_windows):
        """Test computing all perpetual features with provided data."""
        from src.models.rolling_windows import RollingWindows
        
        rw = RollingWindows(**sample_rolling_windows)
        funding_rate = 0.0001
        next_funding_time = int((datetime.now(timezone.utc).timestamp() + 3600) * 1000)
        
        features = compute_all_perpetual_features(rw, funding_rate, next_funding_time)
        
        assert "funding_rate" in features
        assert "time_to_funding" in features
        assert features["funding_rate"] == 0.0001
        assert features["time_to_funding"] is not None
        assert features["time_to_funding"] >= 0

