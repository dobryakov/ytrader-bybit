"""
Unit tests for FeatureRegistryVersionManager.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

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
class TestFeatureRegistryVersionManager:
    """Tests for FeatureRegistryVersionManager."""
    
    async def test_get_version_file_path(self, version_manager):
        """Test get_version_file_path constructs correct path."""
        path = version_manager.get_version_file_path("1.0.0")
        assert path.name == "feature_registry_v1.0.0.yaml"
        assert path.parent == version_manager._versions_dir
    
    async def test_load_active_version_success(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test load_active_version loads from DB and reads file."""
        # Setup: Create version file
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock DB response
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
                "is_active": True,
            }
        )
        
        # Load active version
        result = await version_manager.load_active_version()
        
        assert result is not None
        assert result["version"] == version
        assert "features" in result
    
    async def test_load_active_version_not_found(self, version_manager, mock_metadata_storage):
        """Test load_active_version raises FileNotFoundError when no active version."""
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value=None
        )
        
        with pytest.raises(FileNotFoundError):
            await version_manager.load_active_version()
    
    async def test_load_active_version_file_missing(
        self, version_manager, mock_metadata_storage
    ):
        """Test load_active_version handles missing file."""
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value={
                "version": "1.0.0",
                "file_path": "/nonexistent/path.yaml",
                "is_active": True,
            }
        )
        
        with pytest.raises(FileNotFoundError):
            await version_manager.load_active_version()
    
    async def test_load_active_version_mismatch(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test load_active_version handles version mismatch between DB and file."""
        config_data = get_valid_feature_registry_config()
        config_data["version"] = "1.0.0"
        file_path = version_manager.get_version_file_path("1.0.0")
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # DB says version is 2.0.0, but file has 1.0.0
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value={
                "version": "2.0.0",
                "file_path": str(file_path),
                "is_active": True,
            }
        )
        
        # Should still load but log warning
        result = await version_manager.load_active_version()
        assert result is not None
        assert result["version"] == "1.0.0"  # File version is source of truth
    
    async def test_create_version_success(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test create_version saves file and creates DB record."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(return_value=None)
        mock_metadata_storage.create_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(version_manager.get_version_file_path(version)),
                "is_active": False,
            }
        )
        
        result = await version_manager.create_version(
            version=version,
            config_data=config_data,
        )
        
        assert result is not None
        assert result["version"] == version
        
        # Verify file was created
        file_path = version_manager.get_version_file_path(version)
        assert file_path.exists()
        
        # Verify file content
        with open(file_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        assert loaded_config["version"] == version
    
    async def test_create_version_duplicate(
        self, version_manager, mock_metadata_storage
    ):
        """Test create_version raises ValueError for duplicate version."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={"version": version}
        )
        
        with pytest.raises(ValueError, match="already exists"):
            await version_manager.create_version(
                version=version,
                config_data=config_data,
            )
    
    async def test_create_version_invalid_config(
        self, version_manager, mock_metadata_storage
    ):
        """Test create_version validates config before saving."""
        invalid_config = {"version": "1.0.0"}  # Missing features
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="Invalid Feature Registry configuration"):
            await version_manager.create_version(
                version="1.0.0",
                config_data=invalid_config,
            )
    
    async def test_activate_version_success(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test activate_version validates and activates version."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        file_path = version_manager.get_version_file_path(version)
        
        # Create file
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock DB responses
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
                "file_path": str(file_path),
                "is_active": True,
            }
        )
        
        result = await version_manager.activate_version(version=version)
        
        assert result is not None
        assert result["is_active"] is True
        mock_metadata_storage.activate_feature_registry_version.assert_called_once()
    
    async def test_activate_version_not_found(self, version_manager, mock_metadata_storage):
        """Test activate_version raises ValueError when version not found."""
        mock_metadata_storage.get_feature_registry_version = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="not found"):
            await version_manager.activate_version(version="1.0.0")
    
    async def test_activate_version_file_missing(
        self, version_manager, mock_metadata_storage
    ):
        """Test activate_version raises FileNotFoundError when file missing."""
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": "1.0.0",
                "file_path": "/nonexistent/path.yaml",
            }
        )
        
        with pytest.raises(FileNotFoundError):
            await version_manager.activate_version(version="1.0.0")
    
    async def test_activate_version_invalid_config(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test activate_version validates config before activation."""
        version = "1.0.0"
        file_path = version_manager.get_version_file_path(version)
        
        # Create invalid config file
        invalid_config = {"version": version}  # Missing features
        with open(file_path, "w") as f:
            yaml.dump(invalid_config, f)
        
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(file_path),
            }
        )
        
        with pytest.raises(ValueError, match="Validation failed"):
            await version_manager.activate_version(version=version)
    
    async def test_can_delete_version_true(
        self, version_manager, mock_metadata_storage
    ):
        """Test can_delete_version returns True when version not in use."""
        mock_metadata_storage.check_version_usage = AsyncMock(return_value=0)
        
        result = await version_manager.can_delete_version("1.0.0")
        assert result is True
    
    async def test_can_delete_version_false(
        self, version_manager, mock_metadata_storage
    ):
        """Test can_delete_version returns False when version in use."""
        mock_metadata_storage.check_version_usage = AsyncMock(return_value=5)
        
        result = await version_manager.can_delete_version("1.0.0")
        assert result is False
    
    async def test_migrate_legacy_to_db_success(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test migrate_legacy_to_db migrates legacy file."""
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        # Create legacy file
        legacy_path = temp_versions_dir.parent / "feature_registry.yaml"
        with open(legacy_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Mock DB responses
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value=None
        )
        mock_metadata_storage.create_feature_registry_version = AsyncMock(
            return_value={
                "version": version,
                "file_path": str(version_manager.get_version_file_path(version)),
                "is_active": True,
            }
        )
        
        result = await version_manager.migrate_legacy_to_db(
            legacy_config_path=str(legacy_path)
        )
        
        assert result is not None
        assert result["version"] == version
        assert result["is_active"] is True
        
        # Verify file was created in versions directory
        version_file = version_manager.get_version_file_path(version)
        assert version_file.exists()
    
    async def test_migrate_legacy_to_db_already_active(
        self, version_manager, mock_metadata_storage
    ):
        """Test migrate_legacy_to_db skips if active version exists."""
        existing_version = {
            "version": "1.0.0",
            "is_active": True,
        }
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value=existing_version
        )
        
        result = await version_manager.migrate_legacy_to_db()
        
        assert result == existing_version
        mock_metadata_storage.create_feature_registry_version.assert_not_called()
    
    async def test_migrate_legacy_to_db_file_not_found(
        self, version_manager, mock_metadata_storage
    ):
        """Test migrate_legacy_to_db raises FileNotFoundError when legacy file missing."""
        with pytest.raises(FileNotFoundError):
            await version_manager.migrate_legacy_to_db(
                legacy_config_path="/nonexistent/path.yaml"
            )

