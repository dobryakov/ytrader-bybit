"""
Unit tests for Feature Registry configuration loader.
"""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import yaml
import json


class TestFeatureRegistryLoader:
    """Tests for Feature Registry configuration loader."""
    
    def test_loader_loads_yaml_config(self):
        """Test that loader loads YAML configuration."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            config = loader.load()
            
            assert config is not None
            assert config["version"] == "1.0.0"
            assert len(config["features"]) > 0
    
    def test_loader_validates_config_structure(self):
        """Test that loader validates configuration structure."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        invalid_yaml = """
        version: "1.0.0"
        # Missing features section
        """
        
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            
            with pytest.raises((ValueError, KeyError)):
                config = loader.load()
    
    def test_loader_validates_feature_definitions(self):
        """Test that loader validates feature definitions."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            # Missing required fields
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            
            with pytest.raises(ValueError):
                config = loader.load()
    
    def test_loader_handles_missing_file(self):
        """Test that loader handles missing configuration file."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        loader = FeatureRegistryLoader(config_path="/nonexistent/config.yaml")
        
        with pytest.raises(FileNotFoundError):
            config = loader.load()
    
    def test_loader_validates_temporal_boundaries(self):
        """Test that loader validates temporal boundaries."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "future_feature"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: false  # Invalid: should be true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            
            with pytest.raises(ValueError):
                config = loader.load()
    
    def test_loader_validates_max_lookback_days(self):
        """Test that loader validates max_lookback_days."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "1d"
            lookahead_forbidden: true
            max_lookback_days: -1  # Invalid: should be non-negative
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            
            # Should validate max_lookback_days is non-negative
            with pytest.raises(ValueError):
                config = loader.load()

