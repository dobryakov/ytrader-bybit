"""
Unit tests for Feature Registry model.
"""
import pytest
from pydantic import ValidationError
from src.models.feature_registry import (
    FeatureRegistry,
    FeatureDefinition,
    DataSource,
)
from tests.fixtures.feature_registry import (
    get_valid_feature_registry_config,
    get_invalid_feature_registry_config_missing_version,
    get_invalid_feature_registry_config_missing_features,
    get_data_leakage_feature_registry_config,
    get_data_leakage_feature_registry_config_negative_lookback,
    get_data_leakage_feature_registry_config_excessive_lookback,
)


class TestFeatureDefinition:
    """Tests for FeatureDefinition model."""
    
    def test_valid_feature_definition(self):
        """Test creating a valid feature definition."""
        feature = FeatureDefinition(
            name="mid_price",
            input_sources=["orderbook"],
            lookback_window="0s",
            lookahead_forbidden=True,
            max_lookback_days=0
        )
        
        assert feature.name == "mid_price"
        assert feature.input_sources == ["orderbook"]
        assert feature.lookback_window == "0s"
        assert feature.lookahead_forbidden is True
        assert feature.max_lookback_days == 0
    
    def test_feature_definition_with_data_sources(self):
        """Test feature definition with data sources."""
        feature = FeatureDefinition(
            name="returns_1s",
            input_sources=["kline"],
            lookback_window="1s",
            lookahead_forbidden=True,
            max_lookback_days=1,
            data_sources=[
                DataSource(source="kline", timestamp_required=True)
            ]
        )
        
        assert len(feature.data_sources) == 1
        assert feature.data_sources[0].source == "kline"
    
    def test_invalid_lookback_window_format(self):
        """Test invalid lookback window format."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureDefinition(
                name="test",
                input_sources=["kline"],
                lookback_window="invalid",
                lookahead_forbidden=True,
                max_lookback_days=1
            )
        
        assert "lookback_window" in str(exc_info.value)
    
    def test_negative_lookback_window(self):
        """Test negative lookback window (data leakage)."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureDefinition(
                name="test",
                input_sources=["kline"],
                lookback_window="-1m",
                lookahead_forbidden=True,
                max_lookback_days=1
            )
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_negative_max_lookback_days(self):
        """Test negative max_lookback_days."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureDefinition(
                name="test",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=True,
                max_lookback_days=-1
            )
        
        assert "cannot be negative" in str(exc_info.value)
    
    def test_lookahead_forbidden_false(self):
        """Test lookahead_forbidden=False (data leakage)."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureDefinition(
                name="test",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=False,  # Data leakage
                max_lookback_days=1
            )
        
        assert "lookahead_forbidden must be True" in str(exc_info.value)
    
    def test_excessive_max_lookback_days(self):
        """Test excessive max_lookback_days (data leakage risk)."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureDefinition(
                name="test",
                input_sources=["kline"],
                lookback_window="1m",
                lookahead_forbidden=True,
                max_lookback_days=365  # Excessive
            )
        
        assert "exceeds recommended limit" in str(exc_info.value)
    
    def test_valid_lookback_window_units(self):
        """Test valid lookback window units."""
        valid_units = ["s", "m", "h", "d"]
        for unit in valid_units:
            feature = FeatureDefinition(
                name=f"test_{unit}",
                input_sources=["kline"],
                lookback_window=f"1{unit}",
                lookahead_forbidden=True,
                max_lookback_days=1
            )
            assert feature.lookback_window == f"1{unit}"


class TestFeatureRegistry:
    """Tests for FeatureRegistry model."""
    
    def test_valid_feature_registry(self):
        """Test creating a valid feature registry."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        assert registry.version == "1.0.0"
        assert len(registry.features) == 3
        assert registry.get_feature("mid_price") is not None
        assert registry.get_feature("returns_1s") is not None
        assert registry.get_feature("funding_rate") is not None
    
    def test_feature_registry_missing_version(self):
        """Test feature registry with missing version."""
        config = get_invalid_feature_registry_config_missing_version()
        
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(**config)
        
        assert "version" in str(exc_info.value)
    
    def test_feature_registry_missing_features(self):
        """Test feature registry with missing features."""
        config = get_invalid_feature_registry_config_missing_features()
        
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(**config)
        
        assert "features" in str(exc_info.value)
    
    def test_feature_registry_empty_features(self):
        """Test feature registry with empty features list."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(
                version="1.0.0",
                features=[]
            )
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_duplicate_feature_names(self):
        """Test feature registry with duplicate feature names."""
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(
                version="1.0.0",
                features=[
                    FeatureDefinition(
                        name="duplicate",
                        input_sources=["kline"],
                        lookback_window="1s",
                        lookahead_forbidden=True,
                        max_lookback_days=1
                    ),
                    FeatureDefinition(
                        name="duplicate",
                        input_sources=["kline"],
                        lookback_window="1s",
                        lookahead_forbidden=True,
                        max_lookback_days=1
                    )
                ]
            )
        
        assert "Duplicate feature names" in str(exc_info.value)
    
    def test_get_feature(self):
        """Test getting a feature by name."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        feature = registry.get_feature("mid_price")
        assert feature is not None
        assert feature.name == "mid_price"
        
        # Test non-existent feature
        feature = registry.get_feature("non_existent")
        assert feature is None
    
    def test_get_required_data_types(self):
        """Test getting required data types."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        data_types = registry.get_required_data_types()
        assert "orderbook" in data_types
        assert "kline" in data_types
        assert "funding" in data_types
    
    def test_to_dict(self):
        """Test converting registry to dictionary."""
        config = get_valid_feature_registry_config()
        registry = FeatureRegistry(**config)
        
        registry_dict = registry.to_dict()
        assert registry_dict["version"] == "1.0.0"
        assert len(registry_dict["features"]) == 3
        assert isinstance(registry_dict["features"][0], dict)
    
    def test_data_leakage_detection(self):
        """Test detection of data leakage configurations."""
        # Test lookahead_forbidden=False
        config = get_data_leakage_feature_registry_config()
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(**config)
        
        assert "lookahead_forbidden must be True" in str(exc_info.value)
        
        # Test negative lookback
        config = get_data_leakage_feature_registry_config_negative_lookback()
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(**config)
        
        assert "cannot be negative" in str(exc_info.value)
        
        # Test excessive lookback
        config = get_data_leakage_feature_registry_config_excessive_lookback()
        with pytest.raises(ValidationError) as exc_info:
            FeatureRegistry(**config)
        
        assert "exceeds recommended limit" in str(exc_info.value)

