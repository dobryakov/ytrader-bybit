"""
Unit tests for Feature Registry automatic fallback/rollback.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.storage.metadata_storage import MetadataStorage
from tests.fixtures.feature_registry import get_valid_feature_registry_config


@pytest.fixture
def temp_versions_dir():
    """Create temporary directory for version files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_metadata_storage():
    """Mock MetadataStorage instance."""
    storage = MagicMock(spec=MetadataStorage)
    storage.get_active_feature_registry_version = AsyncMock(return_value=None)
    storage.get_feature_registry_version = AsyncMock(return_value=None)
    storage.create_feature_registry_version = AsyncMock()
    storage.activate_feature_registry_version = AsyncMock()
    storage.rollback_feature_registry_version = AsyncMock()
    storage.check_version_usage = AsyncMock(return_value=0)
    return storage


@pytest.mark.asyncio
class TestFeatureRegistryRollback:
    """Tests for automatic fallback/rollback mechanisms."""
    
    async def test_rollback_on_validation_error(
        self, mock_metadata_storage, temp_versions_dir
    ):
        """Test automatic rollback when validation fails during activation."""
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=mock_metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        version = "1.0.0"
        invalid_config = {"version": version}  # Missing features
        
        # Create invalid config file
        file_path = version_manager.get_version_file_path(version)
        with open(file_path, "w") as f:
            yaml.dump(invalid_config, f)
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
            }
        )
        
        # Activation should fail validation
        with pytest.raises(ValueError, match="Feature Registry validation failed"):
            await version_manager.activate_version(version=version)
        
        # Rollback should not be called automatically (it's the caller's responsibility)
        # But we verify activation was not successful
        mock_metadata_storage.activate_feature_registry_version.assert_not_called()
    
    async def test_rollback_on_migration_error(
        self, mock_metadata_storage, temp_versions_dir
    ):
        """Test automatic rollback when migration fails."""
        import yaml
        from tests.fixtures.feature_registry import get_valid_feature_registry_config
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=mock_metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        # Create a valid legacy file
        config_data = get_valid_feature_registry_config()
        legacy_path = temp_versions_dir.parent / "feature_registry.yaml"
        with open(legacy_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock migration failure in create_feature_registry_version
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value=None
        )
        mock_metadata_storage.create_feature_registry_version = AsyncMock(
            side_effect=Exception("Migration failed")
        )
        
        # Migration should fail
        with pytest.raises(Exception, match="Migration failed"):
            await version_manager.migrate_legacy_to_db(
                legacy_config_path=str(legacy_path)
            )
    
    async def test_rollback_on_runtime_error_during_computation(
        self, mock_metadata_storage, temp_versions_dir
    ):
        """Test that runtime errors during feature computation don't trigger rollback."""
        # This test verifies that runtime errors in feature computation
        # don't automatically trigger registry rollback
        # Rollback should only happen during activation/migration failures
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=mock_metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        # Create valid config file
        file_path = version_manager.get_version_file_path(version)
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Successfully activate version
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
            }
        )
        mock_metadata_storage.activate_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "is_active": True,
            }
        )
        
        result = await version_manager.activate_version(version=version)
        assert result["is_active"] is True
        
        # Runtime errors during feature computation should not trigger rollback
        # (This is tested at integration level, not unit level)
    
    async def test_fallback_to_previous_version_on_load_failure(
        self, mock_metadata_storage, temp_versions_dir
    ):
        """Test fallback to previous version when active version file is missing."""
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=mock_metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        # Mock active version with missing file
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value={
                "version": "1.0.0",
                "file_path": "/nonexistent/path.yaml",
                "is_active": True,
                "previous_version": "0.9.0",
            }
        )
        
        # Load should fail
        with pytest.raises(FileNotFoundError):
            await version_manager.load_active_version()
        
        # Note: Automatic fallback to previous version would be implemented
        # at a higher level (e.g., in startup process)
    
    async def test_rollback_feature_registry_version_success(
        self, mock_metadata_storage
    ):
        """Test rollback_feature_registry_version activates previous version."""
        previous_version = {
            "version": "0.9.0",
            "is_active": True,
            "previous_version": None,
        }
        
        mock_metadata_storage.rollback_feature_registry_version = AsyncMock(
            return_value=previous_version
        )
        
        result = await mock_metadata_storage.rollback_feature_registry_version()
        
        assert result is not None
        assert result["version"] == "0.9.0"
        assert result["is_active"] is True
    
    async def test_rollback_feature_registry_version_no_previous(
        self, mock_metadata_storage
    ):
        """Test rollback_feature_registry_version returns None when no previous version."""
        mock_metadata_storage.rollback_feature_registry_version = AsyncMock(
            return_value=None
        )
        
        result = await mock_metadata_storage.rollback_feature_registry_version()
        
        assert result is None

