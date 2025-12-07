"""
Feature Registry configuration loader and management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class FeatureRegistryLoader:
    """Feature Registry configuration loader."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Feature Registry loader.
        
        Args:
            config_path: Path to Feature Registry YAML file
        """
        self._config_path = Path(config_path or config.feature_registry_path)
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """
        Load Feature Registry configuration from YAML file.
        
        Returns:
            Dict containing Feature Registry configuration
            
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config is invalid
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Feature Registry config not found: {self._config_path}")
        
        with open(self._config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError("Feature Registry config is empty")
        
        # Validate structure
        if "version" not in config_data:
            raise ValueError("Feature Registry config must include 'version' field")
        
        if "features" not in config_data:
            raise ValueError("Feature Registry config must include 'features' section")
        
        # Validate each feature
        for feature in config_data["features"]:
            self._validate_feature(feature)
        
        self._config = config_data
        logger.info("Feature Registry loaded", version=config_data.get("version"))
        
        return config_data
    
    def _validate_feature(self, feature: Dict[str, Any]) -> None:
        """
        Validate a feature definition.
        
        Args:
            feature: Feature definition dictionary
            
        Raises:
            ValueError: If feature is invalid
        """
        required_fields = ["name", "input_sources", "lookback_window", "lookahead_forbidden"]
        
        for field in required_fields:
            if field not in feature:
                raise ValueError(f"Feature '{feature.get('name', 'unknown')}' missing required field: {field}")
        
        # Validate lookahead_forbidden
        if not feature["lookahead_forbidden"]:
            raise ValueError(
                f"Feature '{feature['name']}' must have lookahead_forbidden=true "
                "to prevent data leakage"
            )
        
        # Validate max_lookback_days if present
        if "max_lookback_days" in feature:
            lookback_window = feature["lookback_window"]
            max_lookback_days = feature["max_lookback_days"]
            
            # Parse lookback_window to days (simplified)
            # This is a basic validation - actual parsing would be more complex
            if "d" in lookback_window.lower() or "day" in lookback_window.lower():
                # Extract days from lookback_window
                # For now, just check that max_lookback_days is reasonable
                if max_lookback_days < 0:
                    raise ValueError(
                        f"Feature '{feature['name']}' max_lookback_days must be non-negative"
                    )
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get loaded configuration.
        
        Returns:
            Configuration dictionary or None if not loaded
        """
        return self._config
    
    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.
        
        Returns:
            Dict containing Feature Registry configuration
        """
        self._config = None
        return self.load()
    
    def get_required_data_types(self) -> set:
        """
        Determine required data types from Feature Registry.
        
        Extracts unique input_sources from all features in registry.
        
        Returns:
            Set of required data types: {"orderbook", "kline", "trades", "ticker", "funding"}
            
        Raises:
            ValueError: If registry is not loaded
        """
        if self._config is None:
            raise ValueError("Feature Registry not loaded. Call load() first.")
        
        required_types = set()
        
        if "features" not in self._config:
            logger.warning("Feature Registry has no features section")
            return required_types
        
        for feature in self._config["features"]:
            if "input_sources" in feature:
                input_sources = feature["input_sources"]
                if isinstance(input_sources, list):
                    required_types.update(input_sources)
                elif isinstance(input_sources, str):
                    required_types.add(input_sources)
        
        logger.debug("Required data types determined", data_types=sorted(required_types))
        return required_types
    
    def get_data_type_mapping(self) -> Dict[str, List[str]]:
        """
        Map Feature Registry input_sources to actual data storage types.
        
        Maps:
        - "orderbook" → ["orderbook_snapshots", "orderbook_deltas"]
        - "kline" → ["klines"]
        - "trades" → ["trades"]
        - "ticker" → ["ticker"]
        - "funding" → ["funding"]
        
        Returns:
            Dict mapping input_source to list of storage types
            
        Raises:
            ValueError: If registry is not loaded
        """
        if self._config is None:
            raise ValueError("Feature Registry not loaded. Call load() first.")
        
        mapping = {
            "orderbook": ["orderbook_snapshots", "orderbook_deltas"],
            "kline": ["klines"],
            "trades": ["trades"],
            "ticker": ["ticker"],
            "funding": ["funding"],
        }
        
        # Get required input sources
        required_sources = self.get_required_data_types()
        
        # Return mapping only for required sources
        result = {source: mapping[source] for source in required_sources if source in mapping}
        
        logger.debug("Data type mapping created", mapping=result)
        return result

