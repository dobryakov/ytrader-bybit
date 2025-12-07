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
    
    def test_get_required_data_types_all_types(self):
        """Test get_required_data_types() with registry containing all data types."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: true
          - name: "returns_1m"
            input_sources: ["kline"]
            lookback_window: "1m"
            lookahead_forbidden: true
          - name: "vwap_3s"
            input_sources: ["trades"]
            lookback_window: "3s"
            lookahead_forbidden: true
          - name: "returns_1s"
            input_sources: ["ticker"]
            lookback_window: "1s"
            lookahead_forbidden: true
          - name: "funding_rate"
            input_sources: ["funding"]
            lookback_window: "0s"
            lookahead_forbidden: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            loader.load()
            
            required_types = loader.get_required_data_types()
            
            assert isinstance(required_types, set)
            assert "orderbook" in required_types
            assert "kline" in required_types
            assert "trades" in required_types
            assert "ticker" in required_types
            assert "funding" in required_types
    
    def test_get_required_data_types_subset(self):
        """Test get_required_data_types() with registry containing subset of data types."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: true
          - name: "returns_1m"
            input_sources: ["kline"]
            lookback_window: "1m"
            lookahead_forbidden: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            loader.load()
            
            required_types = loader.get_required_data_types()
            
            assert isinstance(required_types, set)
            assert "orderbook" in required_types
            assert "kline" in required_types
            assert "trades" not in required_types
            assert "ticker" not in required_types
            assert "funding" not in required_types
    
    def test_get_required_data_types_empty_registry(self):
        """Test get_required_data_types() with empty registry."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features: []
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            loader.load()
            
            required_types = loader.get_required_data_types()
            
            assert isinstance(required_types, set)
            assert len(required_types) == 0
    
    def test_get_required_data_types_not_loaded(self):
        """Test get_required_data_types() raises error if registry not loaded."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
        
        with pytest.raises(ValueError, match="not loaded"):
            loader.get_required_data_types()
    
    def test_get_data_type_mapping_all_types(self):
        """Test get_data_type_mapping() with all data types."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: true
          - name: "returns_1m"
            input_sources: ["kline"]
            lookback_window: "1m"
            lookahead_forbidden: true
          - name: "vwap_3s"
            input_sources: ["trades"]
            lookback_window: "3s"
            lookahead_forbidden: true
          - name: "returns_1s"
            input_sources: ["ticker"]
            lookback_window: "1s"
            lookahead_forbidden: true
          - name: "funding_rate"
            input_sources: ["funding"]
            lookback_window: "0s"
            lookahead_forbidden: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            loader.load()
            
            mapping = loader.get_data_type_mapping()
            
            assert isinstance(mapping, dict)
            assert "orderbook" in mapping
            assert mapping["orderbook"] == ["orderbook_snapshots", "orderbook_deltas"]
            assert "kline" in mapping
            assert mapping["kline"] == ["klines"]
            assert "trades" in mapping
            assert mapping["trades"] == ["trades"]
            assert "ticker" in mapping
            assert mapping["ticker"] == ["ticker"]
            assert "funding" in mapping
            assert mapping["funding"] == ["funding"]
    
    def test_get_data_type_mapping_subset(self):
        """Test get_data_type_mapping() with subset of data types."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        yaml_content = """
        version: "1.0.0"
        features:
          - name: "mid_price"
            input_sources: ["orderbook"]
            lookback_window: "0s"
            lookahead_forbidden: true
          - name: "returns_1m"
            input_sources: ["kline"]
            lookback_window: "1m"
            lookahead_forbidden: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
            loader.load()
            
            mapping = loader.get_data_type_mapping()
            
            assert isinstance(mapping, dict)
            assert "orderbook" in mapping
            assert "kline" in mapping
            assert "trades" not in mapping
            assert "ticker" not in mapping
            assert "funding" not in mapping
    
    def test_get_data_type_mapping_not_loaded(self):
        """Test get_data_type_mapping() raises error if registry not loaded."""
        from src.services.feature_registry import FeatureRegistryLoader
        
        loader = FeatureRegistryLoader(config_path="/app/config/feature_registry.yaml")
        
        with pytest.raises(ValueError, match="not loaded"):
            loader.get_data_type_mapping()

