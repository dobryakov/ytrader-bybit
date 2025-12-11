"""
Unit tests for Feature Registry configuration validation.
"""
import pytest
from datetime import datetime, timedelta
from src.models.feature_registry import FeatureRegistry, FeatureDefinition
from tests.fixtures.feature_registry import (
    get_valid_feature_registry_config,
    get_data_leakage_feature_registry_config,
    get_data_leakage_feature_registry_config_negative_lookback,
    get_data_leakage_feature_registry_config_excessive_lookback,
    get_feature_registry_config_without_temporal_boundaries,
)


class TestTemporalBoundariesValidation:
    """Tests for temporal boundaries validation."""
    
    def test_valid_temporal_boundaries(self):
        """Test valid temporal boundaries."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        # All features should have valid temporal boundaries
        for feature in registry.features:
            assert feature.lookahead_forbidden is True
            assert feature.max_lookback_days >= 0
    
    def test_missing_temporal_boundaries(self):
        """Test missing temporal boundaries fields."""
        config = get_feature_registry_config_without_temporal_boundaries()
        
        # Should fail validation because lookback_window is required
        with pytest.raises(Exception):  # Either ValidationError or ValueError
            FeatureRegistry(**config)
    
    def test_lookback_window_zero(self):
        """Test zero lookback window (current timestamp only)."""
        feature = FeatureDefinition(
            name="current_price",
            input_sources=["orderbook"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0
        )
        
        assert feature.lookback_window == "0s"
        assert feature.max_lookback_days == 0
    
    def test_lookback_window_seconds(self):
        """Test lookback window in seconds."""
        feature = FeatureDefinition(
            name="returns_1s",
            input_sources=["kline"],
            lookback_window="1s",
            lookahead_forbidden=True,
            max_lookback_days=1
        )
        
        assert feature.lookback_window == "1s"
    
    def test_lookback_window_minutes(self):
        """Test lookback window in minutes."""
        feature = FeatureDefinition(
            name="returns_1m",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1
        )
        
        assert feature.lookback_window == "1m"
    
    def test_lookback_window_hours(self):
        """Test lookback window in hours."""
        feature = FeatureDefinition(
            name="returns_1h",
            input_sources=["kline"],
            lookback_window="1h",
            lookahead_forbidden=True,
            max_lookback_days=1
        )
        
        assert feature.lookback_window == "1h"
    
    def test_lookback_window_days(self):
        """Test lookback window in days."""
        feature = FeatureDefinition(
            name="returns_1d",
            input_sources=["kline"],
            lookback_window="1d",
            lookahead_forbidden=True,
            max_lookback_days=1
        )
        
        assert feature.lookback_window == "1d"


class TestDataLeakagePrevention:
    """Tests for data leakage prevention."""
    
    def test_lookahead_forbidden_true(self):
        """Test that lookahead_forbidden=True prevents data leakage."""
        feature = FeatureDefinition(
            name="valid_feature",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,  # Prevents data leakage
            max_lookback_days=1
        )
        
        assert feature.lookahead_forbidden is True
    
    def test_lookahead_forbidden_false_rejected(self):
        """Test that lookahead_forbidden=False is rejected."""
        with pytest.raises(Exception) as exc_info:
            FeatureDefinition(
                name="invalid_feature",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=False,  # Data leakage
                max_lookback_days=1
            )
        
        assert "lookahead_forbidden must be True" in str(exc_info.value)
    
    def test_negative_lookback_window_rejected(self):
        """Test that negative lookback window is rejected."""
        config = get_data_leakage_feature_registry_config_negative_lookback()
        
        with pytest.raises(Exception) as exc_info:
            FeatureRegistry(**config)
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_excessive_max_lookback_days_rejected(self):
        """Test that excessive max_lookback_days is rejected."""
        config = get_data_leakage_feature_registry_config_excessive_lookback()
        
        with pytest.raises(Exception) as exc_info:
            FeatureRegistry(**config)
        
        assert "exceeds recommended limit" in str(exc_info.value)
    
    def test_max_lookback_days_zero(self):
        """Test max_lookback_days=0 (no historical data)."""
        feature = FeatureDefinition(
            name="current_only",
            input_sources=["orderbook"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0  # No historical data
        )
        
        assert feature.max_lookback_days == 0
    
    def test_max_lookback_days_reasonable(self):
        """Test reasonable max_lookback_days values."""
        # Test various reasonable values
        for days in [1, 7, 30, 90]:
            feature = FeatureDefinition(
                name=f"feature_{days}d",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=True,
                max_lookback_days=days
            )
            
            assert feature.max_lookback_days == days
    
    def test_max_lookback_days_boundary(self):
        """Test max_lookback_days boundary (90 days)."""
        # 90 days should be accepted
        feature = FeatureDefinition(
            name="boundary_feature",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=90
        )
        
        assert feature.max_lookback_days == 90
        
        # 91 days should be rejected
        with pytest.raises(Exception) as exc_info:
            FeatureDefinition(
                name="excessive_feature",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=True,
                max_lookback_days=91  # Exceeds limit
            )
        
        assert "exceeds recommended limit" in str(exc_info.value)


class TestMaxLookbackDaysValidation:
    """Tests for max_lookback_days validation."""
    
    def test_max_lookback_days_negative_rejected(self):
        """Test that negative max_lookback_days is rejected."""
        with pytest.raises(Exception) as exc_info:
            FeatureDefinition(
                name="invalid",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=True,
                max_lookback_days=-1
            )
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_max_lookback_days_zero_allowed(self):
        """Test that max_lookback_days=0 is allowed."""
        feature = FeatureDefinition(
            name="current_only",
            input_sources=["orderbook"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0
        )
        
        assert feature.max_lookback_days == 0
    
    def test_max_lookback_days_consistency(self):
        """Test consistency between lookback_window and max_lookback_days."""
        # Feature with 1 minute lookback should allow max_lookback_days >= 1
        feature = FeatureDefinition(
            name="consistent",
            input_sources=["kline"],
            lookback_window="1m",
            lookahead_forbidden=True,
            max_lookback_days=1
        )
        
        assert feature.lookback_window == "1m"
        assert feature.max_lookback_days == 1
    
    def test_max_lookback_days_with_different_lookback_windows(self):
        """Test max_lookback_days with different lookback window units."""
        test_cases = [
            ("1s", 1),
            ("1m", 1),
            ("1h", 1),
            ("1d", 1),
            ("5m", 1),
            ("15m", 1),
        ]
        
        for lookback_window, max_days in test_cases:
            feature = FeatureDefinition(
                name=f"test_{lookback_window}",
                input_sources=["kline"],
                lookback_window=lookback_window,
                lookahead_forbidden=True,
                max_lookback_days=max_days
            )
            
            assert feature.lookback_window == lookback_window
            assert feature.max_lookback_days == max_days


class TestFeatureRegistryValidationIntegration:
    """Integration tests for Feature Registry validation."""
    
    def test_valid_registry_passes_all_validations(self):
        """Test that a valid registry passes all validations."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        # Should not raise any exceptions
        assert registry.version == "1.0.0"
        assert len(registry.features) == 3
        
        # All features should have valid temporal boundaries
        for feature in registry.features:
            assert feature.lookahead_forbidden is True
            assert feature.max_lookback_days >= 0
    
    def test_invalid_registry_fails_validation(self):
        """Test that an invalid registry fails validation."""
        # Test with data leakage
        config = get_data_leakage_feature_registry_config()
        
        with pytest.raises(Exception):
            FeatureRegistry(**config)
    
    def test_validation_preserves_feature_order(self):
        """Test that validation preserves feature order."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        # Features should be in the same order as input
        assert registry.features[0].name == "mid_price"
        assert registry.features[1].name == "returns_1s"
        assert registry.features[2].name == "funding_rate"

