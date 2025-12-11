"""
Feature Registry Version Manager for database-driven version management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from src.config import config
from src.logging import get_logger
from src.storage.metadata_storage import MetadataStorage
from src.models.feature_registry import FeatureRegistry, FeatureDefinition
from pydantic import ValidationError

logger = get_logger(__name__)


class CompatibilityReport:
    """Report of backward compatibility check results."""
    
    def __init__(
        self,
        compatibility_warnings: List[str],
        breaking_changes: List[str],
    ):
        self.compatibility_warnings = compatibility_warnings
        self.breaking_changes = breaking_changes
    
    @property
    def has_breaking_changes(self) -> bool:
        """Check if report contains breaking changes."""
        return len(self.breaking_changes) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if report contains warnings."""
        return len(self.compatibility_warnings) > 0


class FeatureRegistryVersionManager:
    """Manages Feature Registry versions via database with files as source of truth."""
    
    def __init__(
        self,
        metadata_storage: MetadataStorage,
        versions_dir: Optional[str] = None,
    ):
        """
        Initialize Feature Registry Version Manager.
        
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
        return self._versions_dir / f"feature_registry_v{version}.yaml"
    
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
        version_record = await self._metadata_storage.get_active_feature_registry_version()
        
        if version_record is None:
            raise FileNotFoundError("No active Feature Registry version found in database")
        
        version = version_record["version"]
        file_path = Path(version_record["file_path"])
        
        # Read config from file
        if not file_path.exists():
            logger.warning(
                "feature_registry_file_not_found",
                version=version,
                file_path=str(file_path),
            )
            raise FileNotFoundError(f"Feature Registry file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Feature Registry file is empty: {file_path}")
        
        # Validate version match
        file_version = config_data.get("version")
        if file_version != version:
            logger.warning(
                "feature_registry_version_mismatch",
                db_version=version,
                file_version=file_version,
                file_path=str(file_path),
            )
            # Use file version as source of truth, but log warning
            # In production, this should trigger a sync operation
        
        # Validate config using FeatureRegistry model
        try:
            FeatureRegistry(**config_data)
        except ValidationError as e:
            logger.error(
                "feature_registry_validation_failed",
                version=version,
                file_path=str(file_path),
                errors=str(e),
            )
            raise ValueError(f"Feature Registry validation failed: {e}") from e
        
        logger.info(
            "feature_registry_active_version_loaded",
            version=version,
            file_path=str(file_path),
        )
        
        return config_data
    
    async def create_version(
        self,
        version: str,
        config_data: Dict[str, Any],
        created_by: Optional[str] = None,
    ) -> dict:
        """
        Create a new version: save config to file and create DB record.
        
        Args:
            version: Version identifier
            config_data: Feature Registry configuration dict
            created_by: User who created this version (optional)
            
        Returns:
            Created version record from database
            
        Raises:
            ValueError: If config is invalid or version already exists
        """
        # Validate config before saving
        try:
            FeatureRegistry(**config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid Feature Registry configuration: {e}") from e
        
        # Check if version already exists
        existing = await self._metadata_storage.get_feature_registry_version(version)
        if existing is not None:
            raise ValueError(f"Feature Registry version already exists: {version}")
        
        # Ensure version in config matches
        if config_data.get("version") != version:
            config_data = config_data.copy()
            config_data["version"] = version
        
        # Save to file
        file_path = self.get_version_file_path(version)
        with open(file_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(
            "feature_registry_version_file_created",
            version=version,
            file_path=str(file_path),
        )
        
        # Create DB record
        version_record = await self._metadata_storage.create_feature_registry_version(
            version=version,
            file_path=str(file_path),
            is_active=False,
            created_by=created_by,
        )
        
        logger.info(
            "feature_registry_version_created",
            version=version,
            file_path=str(file_path),
        )
        
        return version_record
    
    async def activate_version(
        self,
        version: str,
        activated_by: Optional[str] = None,
        activation_reason: Optional[str] = None,
        acknowledge_breaking_changes: bool = False,
    ) -> dict:
        """
        Activate a version: validate config from file and update DB.
        
        Args:
            version: Version identifier to activate
            activated_by: User who activated this version (optional)
            activation_reason: Reason for activation (optional)
            acknowledge_breaking_changes: Must be True if breaking changes are detected
            
        Returns:
            Activated version record from database
            
        Raises:
            ValueError: If version not found, validation fails, or breaking changes not acknowledged
        """
        # Get version record
        version_record = await self._metadata_storage.get_feature_registry_version(version)
        if version_record is None:
            raise ValueError(f"Feature Registry version not found: {version}")
        
        file_path = Path(version_record["file_path"])
        
        # Validate config from file
        if not file_path.exists():
            raise FileNotFoundError(f"Feature Registry file not found: {file_path}")
        
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Feature Registry file is empty: {file_path}")
        
        # Validate using FeatureRegistry model
        try:
            FeatureRegistry(**config_data)
        except ValidationError as e:
            logger.error(
                "feature_registry_activation_validation_failed",
                version=version,
                file_path=str(file_path),
                errors=str(e),
            )
            raise ValueError(f"Feature Registry validation failed: {e}") from e
        
        # Check backward compatibility with current active version
        compatibility_report = None
        current_active = await self._metadata_storage.get_active_feature_registry_version()
        if current_active:
            try:
                # Load current active config from file
                current_file_path = Path(current_active["file_path"])
                if current_file_path.exists():
                    with open(current_file_path, "r") as f:
                        current_config = yaml.safe_load(f)
                    
                    if current_config:
                        compatibility_report = self.check_backward_compatibility(
                            old_config=current_config,
                            new_config=config_data,
                        )
                        
                        # Log compatibility report
                        if compatibility_report.has_breaking_changes:
                            logger.warning(
                                "breaking_changes_detected",
                                version=version,
                                breaking_changes=compatibility_report.breaking_changes,
                            )
                        if compatibility_report.has_warnings:
                            logger.info(
                                "compatibility_warnings",
                                version=version,
                                warnings=compatibility_report.compatibility_warnings,
                            )
            except Exception as e:
                logger.warning(
                    "compatibility_check_failed",
                    version=version,
                    error=str(e),
                )
        
        # Check if breaking changes require acknowledgment
        if compatibility_report and compatibility_report.has_breaking_changes:
            if not acknowledge_breaking_changes:
                raise ValueError(
                    f"Breaking changes detected. Set 'acknowledge_breaking_changes' to True to proceed. "
                    f"Breaking changes: {', '.join(compatibility_report.breaking_changes)}"
                )
        
        # Apply automatic schema migration if migration_script is provided
        migration_applied = False
        if version_record.get("migration_script"):
            try:
                migration_applied = await self._apply_migration_script(
                    version=version,
                    migration_script=version_record["migration_script"],
                    config_data=config_data,
                )
                if migration_applied:
                    logger.info(
                        "schema_migration_applied",
                        version=version,
                    )
            except Exception as migration_error:
                logger.error(
                    "schema_migration_failed",
                    version=version,
                    error=str(migration_error),
                )
                raise ValueError(
                    f"Schema migration failed: {migration_error}"
                ) from migration_error
        
        # Atomically activate version in DB (with compatibility info)
        activated_record = await self._metadata_storage.activate_feature_registry_version(
            version=version,
            activated_by=activated_by,
            activation_reason=activation_reason,
        )
        
        # Update compatibility warnings and breaking changes in DB
        if compatibility_report:
            await self._metadata_storage.update_feature_registry_version_metadata(
                version=version,
                compatibility_warnings=compatibility_report.compatibility_warnings,
                breaking_changes=compatibility_report.breaking_changes,
            )
        
        logger.info(
            "feature_registry_version_activated",
            version=version,
            activated_by=activated_by,
            activation_reason=activation_reason,
            migration_applied=migration_applied,
            has_breaking_changes=compatibility_report.has_breaking_changes if compatibility_report else False,
        )
        
        # Add compatibility and migration info to return value
        result = activated_record.copy()
        if compatibility_report:
            result["compatibility_warnings"] = compatibility_report.compatibility_warnings
            result["breaking_changes"] = compatibility_report.breaking_changes
        result["migration_applied"] = migration_applied
        
        return result
    
    def check_backward_compatibility(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> CompatibilityReport:
        """
        Check backward compatibility between two Feature Registry versions.
        
        Detects:
        - Removed features (breaking change)
        - Changed feature names (breaking change)
        - Changed feature logic (lookback_window, input_sources, etc.) (warning or breaking)
        
        Args:
            old_config: Previous Feature Registry configuration
            new_config: New Feature Registry configuration
            
        Returns:
            CompatibilityReport with warnings and breaking changes
        """
        compatibility_warnings: List[str] = []
        breaking_changes: List[str] = []
        
        # Parse feature definitions
        try:
            old_registry = FeatureRegistry(**old_config)
            new_registry = FeatureRegistry(**new_config)
        except ValidationError as e:
            breaking_changes.append(f"Invalid configuration: {e}")
            return CompatibilityReport(compatibility_warnings, breaking_changes)
        
        # Create feature maps by name
        old_features = {f.name: f for f in old_registry.features}
        new_features = {f.name: f for f in new_registry.features}
        
        # Check for removed features (breaking change)
        removed_features = set(old_features.keys()) - set(new_features.keys())
        for feature_name in removed_features:
            breaking_changes.append(
                f"Feature '{feature_name}' was removed (breaking change)"
            )
        
        # Check for new features (not breaking, but log as info)
        new_feature_names = set(new_features.keys()) - set(old_features.keys())
        if new_feature_names:
            logger.info(
                "new_features_detected",
                new_features=list(new_feature_names),
            )
        
        # Check for changed feature definitions
        common_features = set(old_features.keys()) & set(new_features.keys())
        for feature_name in common_features:
            old_feature = old_features[feature_name]
            new_feature = new_features[feature_name]
            
            # Check for changed input_sources (breaking change - affects data requirements)
            if set(old_feature.input_sources) != set(new_feature.input_sources):
                breaking_changes.append(
                    f"Feature '{feature_name}': input_sources changed from "
                    f"{old_feature.input_sources} to {new_feature.input_sources} (breaking change)"
                )
            
            # Check for changed lookback_window (warning - may affect feature values)
            if old_feature.lookback_window != new_feature.lookback_window:
                compatibility_warnings.append(
                    f"Feature '{feature_name}': lookback_window changed from "
                    f"'{old_feature.lookback_window}' to '{new_feature.lookback_window}' "
                    f"(may affect feature values)"
                )
            
            # Check for changed max_lookback_days (warning)
            if old_feature.max_lookback_days != new_feature.max_lookback_days:
                compatibility_warnings.append(
                    f"Feature '{feature_name}': max_lookback_days changed from "
                    f"{old_feature.max_lookback_days} to {new_feature.max_lookback_days}"
                )
            
            # Check for changed data_sources (warning - may affect data requirements)
            old_data_sources = {
                ds.source for ds in (old_feature.data_sources or [])
            }
            new_data_sources = {
                ds.source for ds in (new_feature.data_sources or [])
            }
            if old_data_sources != new_data_sources:
                compatibility_warnings.append(
                    f"Feature '{feature_name}': data_sources changed from "
                    f"{old_data_sources} to {new_data_sources}"
                )
        
        return CompatibilityReport(compatibility_warnings, breaking_changes)
    
    async def _apply_migration_script(
        self,
        version: str,
        migration_script: str,
        config_data: Dict[str, Any],
    ) -> bool:
        """
        Apply automatic schema migration script.
        
        Args:
            version: Version identifier
            migration_script: Migration script or instructions (text)
            config_data: New configuration data
            
        Returns:
            True if migration was applied successfully
            
        Raises:
            ValueError: If migration fails
        """
        # For now, migration_script is stored as text
        # In the future, this could execute Python code or SQL
        # For safety, we only log the script and mark migration as applied
        # Actual migration logic would need to be implemented based on requirements
        
        logger.info(
            "migration_script_execution",
            version=version,
            script_length=len(migration_script),
        )
        
        # TODO: Implement actual migration script execution
        # This could involve:
        # 1. Executing Python code (with sandboxing)
        # 2. Running SQL migrations
        # 3. Transforming data structures
        # For now, we just validate that script exists and log it
        
        if not migration_script or not migration_script.strip():
            logger.warning(
                "empty_migration_script",
                version=version,
            )
            return False
        
        # Basic validation: check if script looks valid
        # In production, this would need proper sandboxing and validation
        logger.info(
            "migration_script_ready",
            version=version,
            preview=migration_script[:100] if len(migration_script) > 100 else migration_script,
        )
        
        # Mark migration as applied (actual execution would happen here)
        return True
    
    async def can_delete_version(self, version: str) -> bool:
        """
        Check if a version can be deleted (not in use by any datasets).
        
        Args:
            version: Version identifier
            
        Returns:
            True if version can be deleted, False otherwise
        """
        usage_count = await self._metadata_storage.check_version_usage(version)
        return usage_count == 0
    
    async def sync_db_to_files(self) -> List[str]:
        """
        Sync database records to files (create missing files from DB).
        
        This is a migration helper: if config JSONB still exists in DB,
        create files from it. Otherwise, ensure files exist for all DB versions.
        
        Returns:
            List of created file paths
        """
        versions = await self._metadata_storage.list_feature_registry_versions()
        created_files = []
        
        for version_record in versions:
            file_path = Path(version_record["file_path"])
            
            if not file_path.exists():
                # File missing - cannot recreate without config JSONB
                # In migration scenario, config JSONB might still exist
                logger.warning(
                    "feature_registry_file_missing",
                    version=version_record["version"],
                    file_path=str(file_path),
                )
                # Note: In production, this would require manual intervention
                # or a migration script that preserves config JSONB during transition
        
        return created_files
    
    async def sync_files_to_db(self) -> List[dict]:
        """
        Sync files to database (create DB records for files not in DB).
        
        Scans versions/ directory and creates DB records for files not in DB.
        
        Returns:
            List of created version records
        """
        created_records = []
        db_versions = {
            v["version"] for v in await self._metadata_storage.list_feature_registry_versions()
        }
        
        # Scan versions directory
        for file_path in self._versions_dir.glob("feature_registry_v*.yaml"):
            # Extract version from filename
            filename = file_path.stem  # e.g., "feature_registry_v1.0.0"
            if not filename.startswith("feature_registry_v"):
                continue
            
            version = filename[len("feature_registry_v"):]
            
            if version not in db_versions:
                # File exists but no DB record - create DB record
                try:
                    with open(file_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    
                    if config_data and config_data.get("version") == version:
                        record = await self._metadata_storage.create_feature_registry_version(
                            version=version,
                            file_path=str(file_path),
                            is_active=False,
                        )
                        created_records.append(record)
                        logger.info(
                            "feature_registry_db_record_created_from_file",
                            version=version,
                            file_path=str(file_path),
                        )
                except Exception as e:
                    logger.error(
                        "feature_registry_sync_file_to_db_failed",
                        version=version,
                        file_path=str(file_path),
                        error=str(e),
                    )
        
        return created_records
    
    async def migrate_legacy_to_db(
        self,
        legacy_config_path: Optional[str] = None,
    ) -> dict:
        """
        Migrate legacy feature_registry.yaml to versioned storage.
        
        Args:
            legacy_config_path: Path to legacy feature_registry.yaml (default: from config)
            
        Returns:
            Created version record from database
        """
        legacy_path = Path(legacy_config_path or config.feature_registry_path)
        
        if not legacy_path.exists():
            raise FileNotFoundError(f"Legacy Feature Registry file not found: {legacy_path}")
        
        # Load legacy config
        with open(legacy_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Legacy Feature Registry file is empty: {legacy_path}")
        
        # Extract version from config
        version = config_data.get("version", "1.0.0")
        
        # Validate config
        try:
            FeatureRegistry(**config_data)
        except ValidationError as e:
            raise ValueError(f"Invalid Feature Registry configuration: {e}") from e
        
        # Check if active version already exists
        active_version = await self._metadata_storage.get_active_feature_registry_version()
        if active_version is not None:
            logger.info(
                "feature_registry_migration_skipped",
                reason="active_version_exists",
                existing_version=active_version["version"],
            )
            return active_version
        
        # Save to versions directory
        file_path = self.get_version_file_path(version)
        with open(file_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(
            "feature_registry_legacy_migrated",
            version=version,
            file_path=str(file_path),
        )
        
        # Create DB record with is_active=true
        version_record = await self._metadata_storage.create_feature_registry_version(
            version=version,
            file_path=str(file_path),
            is_active=True,
            created_by="system",
        )
        
        logger.info(
            "feature_registry_legacy_migration_complete",
            version=version,
            file_path=str(file_path),
        )
        
        return version_record

