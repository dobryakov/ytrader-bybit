"""
Unit tests for temporal features computation.
"""
import pytest
from datetime import datetime, timezone
from src.features.temporal_features import (
    compute_time_of_day_sin,
    compute_time_of_day_cos,
    compute_day_of_week_sin,
    compute_day_of_week_cos,
    compute_is_asia_session,
    compute_is_london_session,
    compute_is_ny_session,
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
    
    def test_compute_day_of_week_sin(self):
        """Test computing day of week sine encoding."""
        timestamp = datetime.now(timezone.utc)
        
        sin_value = compute_day_of_week_sin(timestamp)
        
        assert -1.0 <= sin_value <= 1.0
    
    def test_compute_day_of_week_cos(self):
        """Test computing day of week cosine encoding."""
        timestamp = datetime.now(timezone.utc)
        
        cos_value = compute_day_of_week_cos(timestamp)
        
        assert -1.0 <= cos_value <= 1.0
    
    def test_day_of_week_cyclic(self):
        """Test that day of week features are cyclic (7-day period)."""
        # Test that same weekday in different weeks produce same values
        # Monday, January 1, 2024
        timestamp1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Monday, January 8, 2024 (next week same day)
        timestamp2 = datetime(2024, 1, 8, 12, 0, 0, tzinfo=timezone.utc)
        
        sin1 = compute_day_of_week_sin(timestamp1)
        sin2 = compute_day_of_week_sin(timestamp2)
        
        assert abs(sin1 - sin2) < 0.0001  # Should be approximately equal
    
    def test_compute_is_asia_session(self):
        """Test computing Asia session flag."""
        # Within session: 00:00 - 08:00 UTC
        timestamp_in = datetime(2025, 1, 1, 4, 0, 0, tzinfo=timezone.utc)
        assert compute_is_asia_session(timestamp_in) == 1.0
        
        # Outside session
        timestamp_out = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert compute_is_asia_session(timestamp_out) == 0.0
        
        # Boundary: 00:00 (inclusive)
        timestamp_start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert compute_is_asia_session(timestamp_start) == 1.0
        
        # Boundary: 07:59 (inclusive)
        timestamp_end = datetime(2025, 1, 1, 7, 59, 59, tzinfo=timezone.utc)
        assert compute_is_asia_session(timestamp_end) == 1.0
        
        # Boundary: 08:00 (exclusive)
        timestamp_after = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
        assert compute_is_asia_session(timestamp_after) == 0.0
    
    def test_compute_is_london_session(self):
        """Test computing London session flag."""
        # Within session: 07:00 - 15:00 UTC
        timestamp_in = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        assert compute_is_london_session(timestamp_in) == 1.0
        
        # Outside session
        timestamp_out = datetime(2025, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
        assert compute_is_london_session(timestamp_out) == 0.0
        
        # Boundary: 07:00 (inclusive)
        timestamp_start = datetime(2025, 1, 1, 7, 0, 0, tzinfo=timezone.utc)
        assert compute_is_london_session(timestamp_start) == 1.0
        
        # Boundary: 14:59 (inclusive)
        timestamp_end = datetime(2025, 1, 1, 14, 59, 59, tzinfo=timezone.utc)
        assert compute_is_london_session(timestamp_end) == 1.0
        
        # Boundary: 15:00 (exclusive)
        timestamp_after = datetime(2025, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        assert compute_is_london_session(timestamp_after) == 0.0
    
    def test_compute_is_ny_session(self):
        """Test computing NY session flag."""
        # Within session: 13:00 - 21:00 UTC
        timestamp_in = datetime(2025, 1, 1, 17, 0, 0, tzinfo=timezone.utc)
        assert compute_is_ny_session(timestamp_in) == 1.0
        
        # Outside session
        timestamp_out = datetime(2025, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
        assert compute_is_ny_session(timestamp_out) == 0.0
        
        # Boundary: 13:00 (inclusive)
        timestamp_start = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        assert compute_is_ny_session(timestamp_start) == 1.0
        
        # Boundary: 20:59 (inclusive)
        timestamp_end = datetime(2025, 1, 1, 20, 59, 59, tzinfo=timezone.utc)
        assert compute_is_ny_session(timestamp_end) == 1.0
        
        # Boundary: 21:00 (exclusive)
        timestamp_after = datetime(2025, 1, 1, 21, 0, 0, tzinfo=timezone.utc)
        assert compute_is_ny_session(timestamp_after) == 0.0
    
    def test_compute_all_temporal_features(self):
        """Test computing all temporal features."""
        timestamp = datetime.now(timezone.utc)
        
        features = compute_all_temporal_features(timestamp)
        
        assert "time_of_day_sin" in features
        assert "time_of_day_cos" in features
        assert "dow_sin" in features
        assert "dow_cos" in features
        assert "is_asia_session" in features
        assert "is_london_session" in features
        assert "is_ny_session" in features
        assert -1.0 <= features["time_of_day_sin"] <= 1.0
        assert -1.0 <= features["time_of_day_cos"] <= 1.0
        assert -1.0 <= features["dow_sin"] <= 1.0
        assert -1.0 <= features["dow_cos"] <= 1.0
        assert features["is_asia_session"] in [0.0, 1.0]
        assert features["is_london_session"] in [0.0, 1.0]
        assert features["is_ny_session"] in [0.0, 1.0]

