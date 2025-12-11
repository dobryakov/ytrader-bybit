"""
Integration tests for Feature Registry loading and activation.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.feature_registry import FeatureRegistryLoader
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.storage.metadata_storage import MetadataStorage
from tests.fixtures.feature_registry import get_valid_feature_registry_config


@pytest.fixture
def temp_versions_dir():
    """Create temporary directory for version files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
@pytest.mark.integration
class TestFeatureRegistryIntegration:
    """Integration tests for Feature Registry loading and activation."""
    
    async def test_feature_registry_loading_from_db(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test Feature Registry loading from database."""
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
            
            # Load via FeatureRegistryLoader with DB mode
            loader = FeatureRegistryLoader(
                config_path=str(temp_versions_dir / "feature_registry.yaml"),
                use_db=True,
                version_manager=version_manager,
            )
            
            loaded_config = await loader.load_async()
            assert loaded_config["version"] == version
            assert "features" in loaded_config
        
        finally:
            await metadata_storage.close()
    
    async def test_feature_registry_loading_from_file(
        self, temp_versions_dir
    ):
        """Test Feature Registry loading from file (legacy mode)."""
        config_data = get_valid_feature_registry_config()
        file_path = temp_versions_dir / "feature_registry.yaml"
        
        # Create file
        with open(file_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Load via FeatureRegistryLoader in file mode
        loader = FeatureRegistryLoader(
            config_path=str(file_path),
            use_db=False,
        )
        
        loaded_config = loader.load()
        assert loaded_config["version"] == config_data["version"]
        assert "features" in loaded_config
    
    async def test_feature_registry_activation_workflow(
        self, mock_db_pool, temp_versions_dir
    ):
        """Test complete Feature Registry activation workflow."""
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
            )
            assert created["version"] == version
            
            # 2. Verify file exists
            file_path = version_manager.get_version_file_path(version)
            assert file_path.exists()
            
            # 3. Activate version
            activated = await version_manager.activate_version(version=version)
            assert activated["is_active"] is True
            
            # 4. Verify active version in DB
            active = await metadata_storage.get_active_feature_registry_version()
            assert active["version"] == version
            
            # 5. Load active version
            loaded = await version_manager.load_active_version()
            assert loaded["version"] == version
        
        finally:
            await metadata_storage.close()

