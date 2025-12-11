"""
Feature Registry configuration loader and management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import ValidationError
from src.config import config
from src.logging import get_logger
from src.models.feature_registry import FeatureRegistry

logger = get_logger(__name__)

# Forward declaration to avoid circular import
FeatureRegistryVersionManager = None  # type: ignore


class FeatureRegistryLoader:
    """Feature Registry configuration loader."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        use_db: Optional[bool] = None,
        version_manager: Optional[Any] = None,
    ):
        """
        Initialize Feature Registry loader.
        
        Args:
            config_path: Path to Feature Registry YAML file (for file_mode)
            use_db: Whether to use database-driven mode (default: from config)
            version_manager: FeatureRegistryVersionManager instance (required if use_db=True)
        """
        self._config_path = Path(config_path or config.feature_registry_path)
        self._config: Optional[Dict[str, Any]] = None
        self._registry_model: Optional[FeatureRegistry] = None
        self._use_db = use_db if use_db is not None else config.feature_registry_use_db
        self._version_manager = version_manager
    
    def load(self) -> Dict[str, Any]:
        """
        Load Feature Registry configuration (sync method, for backward compatibility).
        
        If use_db=True, this method cannot be used directly - use load_async() instead.
        If use_db=False, loads from file directly (legacy mode).
        
        Validates:
        - Temporal boundaries (lookback_window format, max_lookback_days)
        - Data leakage prevention (lookahead_forbidden, negative lookback, excessive lookback)
        - Feature structure and required fields
        
        Returns:
            Dict containing Feature Registry configuration
            
        Raises:
            FileNotFoundError: If config file does not exist or active version not found
            ValueError: If config is invalid (structure, temporal boundaries, data leakage)
            RuntimeError: If use_db=True (must use load_async() instead)
        """
        if self._use_db and self._version_manager:
            raise RuntimeError(
                "Cannot use sync load() method with database-driven mode. "
                "Use load_async() instead or set use_db=False."
            )
        else:
            return self._load_from_file()
    
    async def load_async(self) -> Dict[str, Any]:
        """
        Load Feature Registry configuration (async method).
        
        If use_db=True, loads from database (active version) and reads file.
        If use_db=False, loads from file directly (legacy mode).
        
        Validates:
        - Temporal boundaries (lookback_window format, max_lookback_days)
        - Data leakage prevention (lookahead_forbidden, negative lookback, excessive lookback)
        - Feature structure and required fields
        
        Returns:
            Dict containing Feature Registry configuration
            
        Raises:
            FileNotFoundError: If config file does not exist or active version not found
            ValueError: If config is invalid (structure, temporal boundaries, data leakage)
        """
        if self._use_db and self._version_manager:
            return await self.load_active_from_db()
        else:
            return self._load_from_file()
    
    async def load_active_from_db(self) -> Dict[str, Any]:
        """
        Load active version from database and read config from file (async).
        
        Returns:
            Dict containing Feature Registry configuration
            
        Raises:
            FileNotFoundError: If active version not found or file missing
            ValueError: If config is invalid
        """
        if self._version_manager is None:
            raise ValueError("version_manager is required for database-driven mode")
        
        # Load active version from DB (async)
        config_data = await self._version_manager.load_active_version()
        
        # Validate and store
        self._validate_and_store_config(config_data)
        
        return config_data
    
    def _load_from_file(self) -> Dict[str, Any]:
        """
        Load Feature Registry configuration from YAML file (legacy mode).
        
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
        
        # Validate and store
        self._validate_and_store_config(config_data)
        
        return config_data
    
    def _validate_and_store_config(self, config_data: Dict[str, Any]) -> None:
        """
        Validate config using FeatureRegistry model and store it.
        
        Args:
            config_data: Configuration dictionary
            
        Raises:
            ValueError: If config is invalid
        """
        # Validate using FeatureRegistry model (comprehensive validation)
        try:
            self._registry_model = FeatureRegistry(**config_data)
        except ValidationError as e:
            # Convert Pydantic ValidationError to more readable ValueError
            errors = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                errors.append(f"{field}: {msg}")
            
            raise ValueError(
                f"Feature Registry validation failed:\n" + "\n".join(errors)
            ) from e
        
        # Store config dict for backward compatibility
        self._config = config_data
        logger.info(
            "Feature Registry loaded and validated",
            version=config_data.get("version"),
            features_count=len(config_data.get("features", [])),
            use_db=self._use_db,
        )
    
    def set_config(self, config_data: Dict[str, Any]) -> None:
        """
        Set configuration manually (for hot reload).
        
        Args:
            config_data: Configuration dictionary
            
        Raises:
            ValueError: If config is invalid
        """
        self._validate_and_store_config(config_data)
        logger.info("Feature Registry config updated manually (hot reload)")
    
    def validate_version_match(self, file_path: Path, db_version: str) -> bool:
        """
        Validate that file version matches DB version.
        
        Args:
            file_path: Path to YAML file
            db_version: Version from database
            
        Returns:
            True if versions match, False otherwise
        """
        if not file_path.exists():
            return False
        
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        file_version = config_data.get("version") if config_data else None
        
        if file_version != db_version:
            logger.warning(
                "feature_registry_version_mismatch",
                db_version=db_version,
                file_version=file_version,
                file_path=str(file_path),
            )
            return False
        
        return True
    
    def _validate_feature(self, feature: Dict[str, Any]) -> None:
        """
        Validate a feature definition (legacy method, now uses FeatureRegistry model).
        
        This method is kept for backward compatibility but validation is now done
        by the FeatureRegistry model in load() method.
        
        Args:
            feature: Feature definition dictionary
            
        Raises:
            ValueError: If feature is invalid
        """
        # This method is deprecated - validation is now done by FeatureRegistry model
        # Keeping for backward compatibility
        pass
    
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
        if self._registry_model is not None:
            # Use model if available (preferred)
            return self._registry_model.get_required_data_types()
        
        # Fallback to dict-based approach for backward compatibility
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

