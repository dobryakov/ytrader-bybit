"""
Target Registry Version Manager for database-driven version management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from src.config import config
from src.logging import get_logger
from src.storage.metadata_storage import MetadataStorage
from src.models.dataset import TargetConfig
from pydantic import ValidationError

logger = get_logger(__name__)


class TargetRegistryVersionManager:
    """Manages Target Registry versions via database with files as source of truth."""
    
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        versions_dir: Optional[str] = None,
    ):
        """
        Initialize Target Registry Version Manager.
        
        Args:
            metadata_storage: MetadataStorage instance for database operations
            versions_dir: Directory for version files (default: from config)
        """
        self._metadata_storage = metadata_storage
        self._versions_dir = Path(versions_dir or config.feature_registry_versions_dir)
        self._versions_dir.mkdir(parents=True, exist_ok=True)
    
    def get_version_file_path(self, version: str) -> Path:
        """
        Get file path for a version.
        
        Args:
            version: Version identifier
            
        Returns:
            Path to version file
        """
        return self._versions_dir / f"target_registry_v{version}.yaml"
    
    async def load_active_version(self) -> Dict[str, Any]:
        """
        Load active version from database and read config from file.
        
        Returns:
            Config dict from file
            
        Raises:
            FileNotFoundError: If active version not found or file missing
            ValueError: If version mismatch between DB and file
        """
        # Get active version from DB
        version_record = await self._metadata_storage.get_active_target_registry_version()
        
        if version_record is None:
            raise FileNotFoundError("No active Target Registry version found in database")
        
        version = version_record["version"]
        file_path = Path(version_record["file_path"]) if version_record.get("file_path") else None
        
        # If file_path is not set, use config from DB
        if file_path is None or not file_path.exists():
            logger.info(
                "target_registry_loading_from_db",
                version=version,
            )
            return version_record["config"]
        
        # Read config from file
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Target Registry file is empty: {file_path}")
        
        # Validate version match
        file_version = config_data.get("version")
        if file_version != version:
            logger.warning(
                "target_registry_version_mismatch",
                db_version=version,
                file_version=file_version,
                file_path=str(file_path),
            )
        
        # Validate config using TargetConfig model
        try:
            TargetConfig(**config_data.get("config", config_data))
        except ValidationError as e:
            logger.error(
                "target_registry_validation_failed",
                version=version,
                file_path=str(file_path),
                errors=str(e),
            )
            raise ValueError(f"Target Registry validation failed: {e}") from e
        
        logger.info(
            "target_registry_active_version_loaded",
            version=version,
            file_path=str(file_path),
        )
        
        return config_data.get("config", config_data)
    
    async def create_version(
        self,
        version: str,
        config_data: Dict[str, Any],
        created_by: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """
        Create a new version: save config to file and create DB record.
        
        Args:
            version: Version identifier
            config_data: Target configuration dict (type, horizon, threshold)
            created_by: User who created this version (optional)
            description: Description of this version (optional)
            
        Returns:
            Created version record from database
            
        Raises:
            ValueError: If config is invalid or version already exists
        """
        # Validate config before saving
        try:
            TargetConfig(**config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid Target Registry configuration: {e}") from e
        
        # Check if version already exists
        existing = await self._metadata_storage.get_target_registry_version(version)
        if existing is not None:
            raise ValueError(f"Target Registry version already exists: {version}")
        
        # Save to file
        file_path = self.get_version_file_path(version)
        yaml_data = {
            "version": version,
            "config": config_data,
            "description": description,
        }
        with open(file_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(
            "target_registry_version_file_created",
            version=version,
            file_path=str(file_path),
        )
        
        # Create DB record
        version_record = await self._metadata_storage.create_target_registry_version(
            version=version,
            config=config_data,
            is_active=False,
            created_by=created_by,
            description=description,
            file_path=str(file_path),
        )
        
        logger.info(
            "target_registry_version_created",
            version=version,
            file_path=str(file_path),
        )
        
        return version_record
    
    async def activate_version(
        self,
        version: str,
        activated_by: Optional[str] = None,
        activation_reason: Optional[str] = None,
    ) -> dict:
        """
        Activate a version: validate config from file and update DB.
        
        Args:
            version: Version identifier to activate
            activated_by: User who activated this version (optional)
            activation_reason: Reason for activation (optional)
            
        Returns:
            Activated version record from database
            
        Raises:
            ValueError: If version not found or validation fails
        """
        # Get version record
        version_record = await self._metadata_storage.get_target_registry_version(version)
        if version_record is None:
            raise ValueError(f"Target Registry version not found: {version}")
        
        file_path = Path(version_record["file_path"]) if version_record.get("file_path") else None
        
        # Validate config from file if exists, otherwise from DB
        if file_path and file_path.exists():
            with open(file_path, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ValueError(f"Target Registry file is empty: {file_path}")
            
            target_config = config_data.get("config", config_data)
        else:
            target_config = version_record["config"]
        
        # Validate using TargetConfig model
        try:
            TargetConfig(**target_config)
        except ValidationError as e:
            logger.error(
                "target_registry_activation_validation_failed",
                version=version,
                errors=str(e),
            )
            raise ValueError(f"Target Registry validation failed: {e}") from e
        
        # Activate version
        activated_record = await self._metadata_storage.activate_target_registry_version(
            version=version,
            activated_by=activated_by,
            activation_reason=activation_reason,
        )
        
        logger.info(
            "target_registry_version_activated",
            version=version,
            activated_by=activated_by,
        )
        
        return activated_record
    
    async def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific version's config.
        
        Args:
            version: Version identifier
            
        Returns:
            Config dict or None if not found
        """
        version_record = await self._metadata_storage.get_target_registry_version(version)
        if version_record is None:
            return None
        
        file_path = Path(version_record["file_path"]) if version_record.get("file_path") else None
        
        # Try to load from file first, fallback to DB
        if file_path and file_path.exists():
            with open(file_path, "r") as f:
                config_data = yaml.safe_load(f)
            return config_data.get("config", config_data)
        
        return version_record["config"]
    
    async def list_versions(self) -> List[dict]:
        """
        List all versions.
        
        Returns:
            List of version records
        """
        return await self._metadata_storage.list_target_registry_versions()

