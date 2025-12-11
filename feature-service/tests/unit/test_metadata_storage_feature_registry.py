"""
Unit tests for MetadataStorage feature registry version methods.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from src.storage.metadata_storage import MetadataStorage


@pytest.fixture
def mock_db_pool():
    """Mock asyncpg connection pool."""
    from tests.fixtures.database import mock_db_pool
    return mock_db_pool()


@pytest.fixture
def metadata_storage(mock_db_pool):
    """Create MetadataStorage instance with mocked pool."""
    return MetadataStorage(pool=mock_db_pool)


@pytest.mark.asyncio
class TestMetadataStorageFeatureRegistry:
    """Tests for MetadataStorage feature registry version methods."""
    
    async def test_get_active_feature_registry_version_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test get_active_feature_registry_version returns active version."""
        # Mock connection and result
        mock_conn = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
        }.get(key)
        
        mock_conn.fetchrow = AsyncMock(return_value=mock_record)
        
        # Mock get_connection context manager
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.get_active_feature_registry_version()
        
        assert result is not None
        assert result["version"] == "1.0.0"
        assert result["is_active"] is True
    
    async def test_get_active_feature_registry_version_not_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test get_active_feature_registry_version returns None when no active version."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.get_active_feature_registry_version()
        
        assert result is None
    
    async def test_create_feature_registry_version(
        self, metadata_storage, mock_db_pool
    ):
        """Test create_feature_registry_version creates new version record."""
        mock_conn = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            "is_active": False,
            "created_at": datetime.now(timezone.utc),
            "created_by": "test_user",
        }.get(key)
        
        mock_conn.fetchrow = AsyncMock(return_value=mock_record)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.create_feature_registry_version(
            version="1.0.0",
            file_path="/app/config/versions/feature_registry_v1.0.0.yaml",
            is_active=False,
            created_by="test_user",
        )
        
        assert result is not None
        assert result["version"] == "1.0.0"
        assert result["is_active"] is False
        mock_conn.fetchrow.assert_called_once()
    
    async def test_activate_feature_registry_version(
        self, metadata_storage, mock_db_pool
    ):
        """Test activate_feature_registry_version atomically updates is_active flags."""
        mock_conn = MagicMock()
        
        # Mock current active version
        current_active = MagicMock()
        current_active.__getitem__ = lambda self, key: {"version": "0.9.0"}.get(key)
        mock_conn.fetchrow = AsyncMock(return_value=current_active)
        
        # Mock activated version
        activated_record = MagicMock()
        activated_record.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            "is_active": True,
            "previous_version": "0.9.0",
            "activated_by": "test_user",
            "activation_reason": "Testing",
        }.get(key)
        
        # Mock transaction context manager
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        # First call returns current active, second returns activated version
        mock_conn.fetchrow = AsyncMock(side_effect=[current_active, activated_record])
        
        result = await metadata_storage.activate_feature_registry_version(
            version="1.0.0",
            activated_by="test_user",
            activation_reason="Testing",
        )
        
        assert result is not None
        assert result["version"] == "1.0.0"
        assert result["is_active"] is True
        assert result["previous_version"] == "0.9.0"
    
    async def test_activate_feature_registry_version_not_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test activate_feature_registry_version raises ValueError when version not found."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No current active
        
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        # Second fetchrow (for UPDATE RETURNING) returns None
        mock_conn.fetchrow = AsyncMock(side_effect=[None, None])
        
        with pytest.raises(ValueError, match="not found"):
            await metadata_storage.activate_feature_registry_version(
                version="1.0.0",
            )
    
    async def test_list_feature_registry_versions(
        self, metadata_storage, mock_db_pool
    ):
        """Test list_feature_registry_versions returns all versions ordered by created_at."""
        mock_conn = MagicMock()
        
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "created_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
        }.get(key)
        
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = lambda self, key: {
            "version": "0.9.0",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        }.get(key)
        
        mock_conn.fetch = AsyncMock(return_value=[mock_row1, mock_row2])
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.list_feature_registry_versions()
        
        assert len(result) == 2
        assert result[0]["version"] == "1.0.0"  # Most recent first
        assert result[1]["version"] == "0.9.0"
    
    async def test_get_feature_registry_version_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test get_feature_registry_version returns specific version."""
        mock_conn = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "file_path": "/app/config/versions/feature_registry_v1.0.0.yaml",
            "is_active": False,
        }.get(key)
        
        mock_conn.fetchrow = AsyncMock(return_value=mock_record)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.get_feature_registry_version("1.0.0")
        
        assert result is not None
        assert result["version"] == "1.0.0"
    
    async def test_get_feature_registry_version_not_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test get_feature_registry_version returns None when version not found."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.get_feature_registry_version("1.0.0")
        
        assert result is None
    
    async def test_check_version_usage(
        self, metadata_storage, mock_db_pool
    ):
        """Test check_version_usage returns count of datasets using version."""
        mock_conn = MagicMock()
        mock_conn.fetchval = AsyncMock(return_value=5)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.check_version_usage("1.0.0")
        
        assert result == 5
    
    async def test_check_version_usage_zero(
        self, metadata_storage, mock_db_pool
    ):
        """Test check_version_usage returns 0 when version not in use."""
        mock_conn = MagicMock()
        mock_conn.fetchval = AsyncMock(return_value=0)
        
        async def mock_get_connection():
            yield mock_conn
        
        metadata_storage.get_connection = MagicMock(return_value=mock_get_connection())
        
        result = await metadata_storage.check_version_usage("1.0.0")
        
        assert result == 0
    
    async def test_rollback_feature_registry_version(
        self, metadata_storage, mock_db_pool
    ):
        """Test rollback_feature_registry_version activates previous version."""
        mock_conn = MagicMock()
        
        # Mock current active version with previous_version
        current_active = MagicMock()
        current_active.__getitem__ = lambda self, key: {
            "version": "1.0.0",
            "previous_version": "0.9.0",
        }.get(key)
        
        # Mock previous version to activate
        previous_record = MagicMock()
        previous_record.__getitem__ = lambda self, key: {
            "version": "0.9.0",
            "is_active": True,
            "previous_version": None,
        }.get(key)
        
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        # First call gets current active, second gets previous version
        mock_conn.fetchrow = AsyncMock(side_effect=[current_active, previous_record])
        
        # Mock activate_feature_registry_version call
        with patch.object(
            metadata_storage,
            "activate_feature_registry_version",
            new_callable=AsyncMock,
            return_value=previous_record,
        ) as mock_activate:
            result = await metadata_storage.rollback_feature_registry_version()
            
            assert result is not None
            assert result["version"] == "0.9.0"
            mock_activate.assert_called_once_with(
                "0.9.0",
                activated_by="system",
                activation_reason="rollback",
            )
    
    async def test_rollback_feature_registry_version_no_previous(
        self, metadata_storage, mock_db_pool
    ):
        """Test rollback_feature_registry_version returns None when no previous version."""
        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No active version
        
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        result = await metadata_storage.rollback_feature_registry_version()
        
        assert result is None
    
    async def test_delete_feature_registry_version(
        self, metadata_storage, mock_db_pool
    ):
        """Test delete_feature_registry_version deletes version record."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 1")
        
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        result = await metadata_storage.delete_feature_registry_version("1.0.0")
        
        assert result is True
        mock_conn.execute.assert_called_once()
    
    async def test_delete_feature_registry_version_not_found(
        self, metadata_storage, mock_db_pool
    ):
        """Test delete_feature_registry_version returns False when version not found."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 0")
        
        async def mock_transaction():
            yield mock_conn
        
        metadata_storage.transaction = MagicMock(return_value=mock_transaction())
        
        result = await metadata_storage.delete_feature_registry_version("1.0.0")
        
        assert result is False

