"""
Contract tests for Feature Registry API endpoints.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.main import app
from src.api.feature_registry import (
    set_feature_registry_loader,
    set_feature_registry_version_manager,
    set_metadata_storage_for_registry,
    set_feature_computer_for_registry,
    set_dataset_builder_for_registry,
    set_orderbook_manager_for_registry,
)
from src.services.feature_registry import FeatureRegistryLoader
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.storage.metadata_storage import MetadataStorage
from tests.fixtures.feature_registry import get_valid_feature_registry_config


@pytest.fixture
def mock_feature_registry_loader():
    """Mock FeatureRegistryLoader."""
    loader = MagicMock(spec=FeatureRegistryLoader)
    loader.get_config = MagicMock(return_value=get_valid_feature_registry_config())
    loader.load = MagicMock(return_value=get_valid_feature_registry_config())
    loader.reload = MagicMock(return_value=get_valid_feature_registry_config())
    loader._use_db = False
    loader._version_manager = None
    return loader


@pytest.fixture
def mock_version_manager():
    """Mock FeatureRegistryVersionManager."""
    manager = MagicMock(spec=FeatureRegistryVersionManager)
    manager.load_active_version = AsyncMock(
        return_value=get_valid_feature_registry_config()
    )
    manager.create_version = AsyncMock(
        return_value={
            "version": "1.0.0",
            "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            "is_active": False,
        }
    )
    manager.activate_version = AsyncMock(
        return_value={
            "version": "1.0.0",
            "is_active": True,
            "hot_reload": True,
        }
    )
    manager.can_delete_version = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_metadata_storage():
    """Mock MetadataStorage."""
    storage = MagicMock(spec=MetadataStorage)
    storage.list_feature_registry_versions = AsyncMock(return_value=[])
    storage.get_feature_registry_version = AsyncMock(return_value=None)
    storage.activate_feature_registry_version = AsyncMock(
        return_value={"version": "1.0.0", "is_active": True}
    )
    storage.rollback_feature_registry_version = AsyncMock(
        return_value={"version": "0.9.0", "is_active": True}
    )
    storage.check_version_usage = AsyncMock(return_value=0)
    storage.delete_feature_registry_version = AsyncMock(return_value=True)
    return storage


@pytest.fixture
def api_client(mock_feature_registry_loader, mock_version_manager, mock_metadata_storage):
    """Create test client with mocked dependencies."""
    # Set up mocks
    set_feature_registry_loader(mock_feature_registry_loader)
    set_feature_registry_version_manager(mock_version_manager)
    set_metadata_storage_for_registry(mock_metadata_storage)
    
    # Mock other dependencies
    mock_feature_computer = MagicMock()
    mock_dataset_builder = MagicMock()
    mock_orderbook_manager = MagicMock()
    set_feature_computer_for_registry(mock_feature_computer)
    set_dataset_builder_for_registry(mock_dataset_builder)
    set_orderbook_manager_for_registry(mock_orderbook_manager)
    
    return TestClient(app)


@pytest.mark.asyncio
class TestFeatureRegistryAPI:
    """Contract tests for Feature Registry API endpoints."""
    
    def test_get_feature_registry(self, api_client, mock_feature_registry_loader):
        """Test GET /feature-registry endpoint."""
        response = api_client.get(
            "/feature-registry",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "features" in data
    
    def test_get_feature_registry_not_loaded(self, api_client, mock_feature_registry_loader):
        """Test GET /feature-registry when registry not loaded."""
        mock_feature_registry_loader.get_config = MagicMock(return_value=None)
        
        response = api_client.get(
            "/feature-registry",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 404
    
    async def test_post_feature_registry_reload(
        self, api_client, mock_feature_registry_loader
    ):
        """Test POST /feature-registry/reload endpoint."""
        response = api_client.post(
            "/feature-registry/reload",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
    
    def test_get_feature_registry_validate(
        self, api_client, mock_feature_registry_loader
    ):
        """Test GET /feature-registry/validate endpoint."""
        response = api_client.get(
            "/feature-registry/validate",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["valid", "invalid"]
    
    async def test_get_feature_registry_versions(
        self, api_client, mock_metadata_storage
    ):
        """Test GET /feature-registry/versions endpoint."""
        mock_metadata_storage.list_feature_registry_versions = AsyncMock(
            return_value=[
                {"version": "1.0.0", "is_active": True},
                {"version": "0.9.0", "is_active": False},
            ]
        )
        
        response = api_client.get(
            "/feature-registry/versions",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
    
    async def test_get_feature_registry_version(
        self, api_client, mock_metadata_storage
    ):
        """Test GET /feature-registry/versions/{version} endpoint."""
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={"version": "1.0.0", "is_active": True}
        )
        
        response = api_client.get(
            "/feature-registry/versions/1.0.0",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"
    
    async def test_get_feature_registry_version_not_found(
        self, api_client, mock_metadata_storage
    ):
        """Test GET /feature-registry/versions/{version} when version not found."""
        mock_metadata_storage.get_feature_registry_version = AsyncMock(return_value=None)
        
        response = api_client.get(
            "/feature-registry/versions/1.0.0",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 404
    
    async def test_post_feature_registry_versions_create(
        self, api_client, mock_version_manager
    ):
        """Test POST /feature-registry/versions endpoint."""
        config_data = get_valid_feature_registry_config()
        
        response = api_client.post(
            "/feature-registry/versions",
            headers={"X-API-Key": "test-key"},
            json={
                "version": "1.0.0",
                "config": config_data,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"
    
    async def test_post_feature_registry_versions_create_invalid_config(
        self, api_client, mock_version_manager
    ):
        """Test POST /feature-registry/versions with invalid config."""
        mock_version_manager.create_version = AsyncMock(
            side_effect=ValueError("Invalid configuration")
        )
        
        response = api_client.post(
            "/feature-registry/versions",
            headers={"X-API-Key": "test-key"},
            json={
                "version": "1.0.0",
                "config": {"version": "1.0.0"},  # Invalid: missing features
            },
        )
        
        assert response.status_code == 400
    
    async def test_post_feature_registry_versions_activate(
        self, api_client, mock_version_manager
    ):
        """Test POST /feature-registry/versions/{version}/activate endpoint."""
        mock_version_manager.load_active_version = AsyncMock(
            return_value=get_valid_feature_registry_config()
        )
        
        response = api_client.post(
            "/feature-registry/versions/1.0.0/activate",
            headers={"X-API-Key": "test-key"},
            json={
                "acknowledge_breaking_changes": False,
                "activation_reason": "Testing",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_active"] is True
    
    async def test_post_feature_registry_versions_activate_not_found(
        self, api_client, mock_version_manager
    ):
        """Test POST /feature-registry/versions/{version}/activate when version not found."""
        mock_version_manager.activate_version = AsyncMock(
            side_effect=ValueError("Version not found")
        )
        
        response = api_client.post(
            "/feature-registry/versions/1.0.0/activate",
            headers={"X-API-Key": "test-key"},
            json={"acknowledge_breaking_changes": False},
        )
        
        assert response.status_code == 400
    
    async def test_post_feature_registry_rollback(
        self, api_client, mock_metadata_storage
    ):
        """Test POST /feature-registry/rollback endpoint."""
        response = api_client.post(
            "/feature-registry/rollback",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "0.9.0"
    
    async def test_get_feature_registry_version_usage(
        self, api_client, mock_metadata_storage
    ):
        """Test GET /feature-registry/versions/{version}/usage endpoint."""
        mock_metadata_storage.check_version_usage = AsyncMock(return_value=5)
        
        response = api_client.get(
            "/feature-registry/versions/1.0.0/usage",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"
        assert data["usage_count"] == 5
        assert data["in_use"] is True
    
    async def test_delete_feature_registry_version(
        self, api_client, mock_version_manager, mock_metadata_storage
    ):
        """Test DELETE /feature-registry/versions/{version} endpoint."""
        response = api_client.delete(
            "/feature-registry/versions/1.0.0",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
    
    async def test_delete_feature_registry_version_in_use(
        self, api_client, mock_version_manager, mock_metadata_storage
    ):
        """Test DELETE /feature-registry/versions/{version} when version in use."""
        mock_version_manager.can_delete_version = AsyncMock(return_value=False)
        mock_metadata_storage.check_version_usage = AsyncMock(return_value=3)
        
        response = api_client.delete(
            "/feature-registry/versions/1.0.0",
            headers={"X-API-Key": "test-key"},
        )
        
        assert response.status_code == 409
    
    async def test_post_feature_registry_versions_sync_file(
        self, api_client, mock_version_manager, mock_metadata_storage
    ):
        """Test POST /feature-registry/versions/{version}/sync-file endpoint."""
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": "1.0.0",
                "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            }
        )
        
        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            import yaml
            mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(
                get_valid_feature_registry_config()
            )
            
            response = api_client.post(
                "/feature-registry/versions/1.0.0/sync-file",
                headers={"X-API-Key": "test-key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["version"] == "1.0.0"
            assert "validation_status" in data

