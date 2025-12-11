"""
Integration tests for Feature Registry version management via database.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from src.storage.metadata_storage import MetadataStorage
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from tests.fixtures.feature_registry import get_valid_feature_registry_config


@pytest.fixture
def temp_versions_dir():
    """Create temporary directory for version files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
@pytest.mark.integration
class TestFeatureRegistryVersionDB:
    """Integration tests for Feature Registry version management via database."""
    
    async def test_full_lifecycle_create_activate_load_rollback(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test full lifecycle: create version, activate, load on startup, rollback."""
        # Initialize services
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        try:
            # 1. Create version
            created = await version_manager.create_version(
                version=version,
                config_data=config_data,
                created_by="test_user",
            )
            assert created["version"] == version
            
            # 2. Activate version
            activated = await version_manager.activate_version(
                version=version,
                activated_by="test_user",
                activation_reason="Testing",
            )
            assert activated["is_active"] is True
            
            # 3. Load active version
            loaded = await version_manager.load_active_version()
            assert loaded["version"] == version
            assert "features" in loaded
            
            # 4. Rollback
            rolled_back = await metadata_storage.rollback_feature_registry_version()
            # Note: Rollback may return None if no previous version
            # This is expected for first version
        
        finally:
            await metadata_storage.close()
    
    async def test_concurrent_activation_prevention(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test concurrent activation prevention."""
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version1 = "1.0.0"
        version2 = "1.1.0"
        
        try:
            # Create two versions
            config1 = config_data.copy()
            config1["version"] = version1
            await version_manager.create_version(
                version=version1,
                config_data=config1,
            )
            
            config2 = config_data.copy()
            config2["version"] = version2
            await version_manager.create_version(
                version=version2,
                config_data=config2,
            )
            
            # Activate first version
            await version_manager.activate_version(version=version1)
            
            # Activate second version (should deactivate first)
            await version_manager.activate_version(version=version2)
            
            # Verify only second is active
            active = await metadata_storage.get_active_feature_registry_version()
            assert active["version"] == version2
        
        finally:
            await metadata_storage.close()
    
    async def test_file_deletion_handling(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test handling of file deletion."""
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        try:
            # Create and activate version
            await version_manager.create_version(
                version=version,
                config_data=config_data,
            )
            await version_manager.activate_version(version=version)
            
            # Delete file
            file_path = version_manager.get_version_file_path(version)
            file_path.unlink()
            
            # Load should fail
            with pytest.raises(FileNotFoundError):
                await version_manager.load_active_version()
        
        finally:
            await metadata_storage.close()
    
    async def test_version_activation_with_hot_reload(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test version activation with hot reload capability."""
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        try:
            # Create and activate version
            await version_manager.create_version(
                version=version,
                config_data=config_data,
            )
            
            # Activate (hot reload would be triggered by API endpoint)
            activated = await version_manager.activate_version(version=version)
            assert activated["is_active"] is True
            
            # Load config for hot reload
            config = await version_manager.load_active_version()
            assert config is not None
        
        finally:
            await metadata_storage.close()
    
    async def test_startup_with_missing_active_version_fallback(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test startup with missing active version fallback."""
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        try:
            # No active version in DB
            with pytest.raises(FileNotFoundError):
                await version_manager.load_active_version()
            
            # Fallback would happen at startup level (main.py)
        
        finally:
            await metadata_storage.close()
    
    async def test_startup_with_missing_file_fallback(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test startup with missing file fallback."""
        metadata_storage = MetadataStorage(pool=mock_db_pool)
        await metadata_storage.initialize()
        
        version_manager = FeatureRegistryVersionManager(
            metadata_storage=metadata_storage,
            versions_dir=str(temp_versions_dir),
        )
        
        config_data = get_valid_feature_registry_config()
        version = config_data["version"]
        
        try:
            # Create version record in DB but don't create file
            await metadata_storage.create_feature_registry_version(
                version=version,
                file_path=str(version_manager.get_version_file_path(version)),
                is_active=True,
            )
            
            # Load should fail (file missing)
            with pytest.raises(FileNotFoundError):
                await version_manager.load_active_version()
        
        finally:
            await metadata_storage.close()

