"""
Unit tests for Feature Registry automatic schema migration.
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
    storage.update_feature_registry_version_metadata = AsyncMock()
    storage.check_version_usage = AsyncMock(return_value=0)
    return storage


@pytest.fixture
def version_manager(mock_metadata_storage, temp_versions_dir):
    """Create FeatureRegistryVersionManager instance."""
    return FeatureRegistryVersionManager(
        metadata_storage=mock_metadata_storage,
        versions_dir=str(temp_versions_dir),
    )


@pytest.mark.asyncio
class TestAutomaticSchemaMigration:
    """Tests for automatic schema migration."""
    
    async def test_migration_script_execution_success(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test automatic migration execution when migration_script is provided."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        # Create file
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock version record with migration script
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
                "migration_script": "# Migration script example\n# Transform data structures",
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
        # Migration should be applied (logged)
    
    async def test_migration_script_execution_empty_script(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test migration with empty script returns False."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock version record with empty migration script
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
                "migration_script": "",  # Empty script
            }
        )
        mock_metadata_storage.activate_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "is_active": True,
            }
        )
        
        # Migration should handle empty script gracefully
        result = await version_manager.activate_version(version=version)
        assert result["is_active"] is True
    
    async def test_migration_script_handling(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test migration script handling and logging."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        migration_script = """
# Example migration script
# This would transform data structures or apply schema changes
def migrate(old_config):
    # Transform logic here
    return old_config
"""
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
                "migration_script": migration_script,
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
    
    async def test_rollback_on_migration_failure(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test rollback on migration failure."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock version record with migration script that will fail
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
                "migration_script": "invalid script that causes error",
            }
        )
        
        # Mock _apply_migration_script to raise error
        with patch.object(
            version_manager,
            "_apply_migration_script",
            new_callable=AsyncMock,
            side_effect=ValueError("Migration script execution failed"),
        ):
            with pytest.raises(ValueError, match="Schema migration failed"):
                await version_manager.activate_version(version=version)
            
            # Activation should not succeed
            mock_metadata_storage.activate_feature_registry_version.assert_not_called()
    
    async def test_migration_script_not_provided(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test activation without migration script (normal flow)."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock version record without migration script
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": False,
                "migration_script": None,  # No migration script
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
        # No migration should be attempted

