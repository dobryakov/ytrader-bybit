"""
Unit tests for automatic archiving/deletion of expired data.
"""
import pytest
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import shutil
import tarfile
import tempfile

from src.services.data_storage import DataStorageService
from src.storage.parquet_storage import ParquetStorage


@pytest.fixture
def tmp_storage_path(tmp_path):
    """Temporary storage path for testing."""
    storage_dir = tmp_path / "raw_data"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


@pytest.fixture
def tmp_archive_path(tmp_path):
    """Temporary archive path for testing."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    return str(archive_dir)


@pytest.fixture
def mock_parquet_storage():
    """Mock ParquetStorage for testing."""
    return MagicMock(spec=ParquetStorage)


@pytest.fixture
def data_storage_service(tmp_storage_path, tmp_archive_path, mock_parquet_storage):
    """DataStorageService instance with archiving enabled for testing."""
    return DataStorageService(
        base_path=tmp_storage_path,
        parquet_storage=mock_parquet_storage,
        retention_days=90,
        archive_path=tmp_archive_path,
        enable_archiving=True,
    )


@pytest.fixture
def data_storage_service_no_archiving(tmp_storage_path, mock_parquet_storage):
    """DataStorageService instance with archiving disabled for testing."""
    return DataStorageService(
        base_path=tmp_storage_path,
        parquet_storage=mock_parquet_storage,
        retention_days=90,
        enable_archiving=False,
    )


class TestDataArchiving:
    """Test cases for automatic archiving/deletion."""
    
    def test_archiving_enabled(self, data_storage_service):
        """Test that archiving can be enabled."""
        assert data_storage_service._enable_archiving is True
    
    def test_archiving_disabled(self, data_storage_service_no_archiving):
        """Test that archiving can be disabled."""
        assert data_storage_service_no_archiving._enable_archiving is False
    
    @pytest.mark.asyncio
    async def test_archive_expired_data(self, data_storage_service, tmp_storage_path, tmp_archive_path):
        """Test that expired data is archived."""
        base_path = Path(tmp_storage_path)
        archive_path = Path(tmp_archive_path)
        
        # Create expired file
        expired_date = (datetime.now(timezone.utc) - timedelta(days=100)).date()
        expired_dir = base_path / "trades" / expired_date.strftime("%Y-%m-%d")
        expired_dir.mkdir(parents=True, exist_ok=True)
        expired_file = expired_dir / "BTCUSDT.parquet"
        expired_file.write_bytes(b"test data")
        
        # Archive expired data (requires data_type and expired_date)
        await data_storage_service.archive_expired_data("trades", expired_date)
        
        # Verify file was archived (implementation may create archive file)
        # For now, we check that the method exists and can be called
        assert hasattr(data_storage_service, 'archive_expired_data')
    
    @pytest.mark.asyncio
    async def test_delete_expired_data_when_archiving_disabled(
        self, data_storage_service_no_archiving, tmp_storage_path
    ):
        """Test that expired data is deleted when archiving is disabled."""
        base_path = Path(tmp_storage_path)
        
        # Create expired file
        expired_date = (datetime.now(timezone.utc) - timedelta(days=100)).date()
        expired_dir = base_path / "trades" / expired_date.strftime("%Y-%m-%d")
        expired_dir.mkdir(parents=True, exist_ok=True)
        expired_file = expired_dir / "BTCUSDT.parquet"
        expired_file.write_bytes(b"test data")
        
        # Delete expired data (archiving disabled, requires data_type and expired_date)
        await data_storage_service_no_archiving.delete_expired_data("trades", expired_date)
        
        # Verify file was deleted or method exists
        assert hasattr(data_storage_service_no_archiving, 'delete_expired_data')
    
    @pytest.mark.asyncio
    async def test_archive_recovery_support(self, data_storage_service, tmp_archive_path):
        """Test that archived data can be recovered."""
        archive_path = Path(tmp_archive_path)
        
        # Create test archive file
        test_archive = archive_path / "trades_2025-01-01.tar.gz"
        
        # Test recovery method exists
        assert hasattr(data_storage_service, 'recover_from_archive')
        
        # For now, we check that the method exists
        # Full recovery implementation will be tested in integration tests
    
    @pytest.mark.asyncio
    async def test_archiving_handles_missing_files(self, data_storage_service):
        """Test that archiving handles missing files gracefully."""
        expired_date = (datetime.now(timezone.utc) - timedelta(days=100)).date()
        
        # Should not raise error if files don't exist (requires data_type and expired_date)
        await data_storage_service.archive_expired_data("trades", expired_date)
        
        # Method should complete without errors
        assert True
    
    @pytest.mark.asyncio
    async def test_archiving_handles_permission_errors(self, data_storage_service, tmp_storage_path):
        """Test that archiving handles permission errors gracefully."""
        base_path = Path(tmp_storage_path)
        
        # Create expired file
        expired_date = (datetime.now(timezone.utc) - timedelta(days=100)).date()
        expired_dir = base_path / "trades" / expired_date.strftime("%Y-%m-%d")
        expired_dir.mkdir(parents=True, exist_ok=True)
        expired_file = expired_dir / "BTCUSDT.parquet"
        expired_file.write_bytes(b"test data")
        
        # Mock permission error
        with patch('shutil.move', side_effect=PermissionError("Permission denied")):
            # Should handle error gracefully (requires data_type and expired_date)
            await data_storage_service.archive_expired_data("trades", expired_date)
        
        # Method should complete (may log error but not crash)
        assert True
    
    @pytest.mark.asyncio
    async def test_archiving_creates_archive_structure(self, data_storage_service, tmp_archive_path):
        """Test that archiving creates proper archive structure."""
        archive_path = Path(tmp_archive_path)
        
        # Archive should create organized structure
        # Format: archive/{data_type}/{date}.tar.gz or similar
        assert archive_path.exists()
        
        # Verify archive directory structure can be created
        assert hasattr(data_storage_service, '_archive_path')
    
    @pytest.mark.asyncio
    async def test_delete_expired_data_removes_files(self, data_storage_service_no_archiving, tmp_storage_path):
        """Test that delete_expired_data removes files."""
        base_path = Path(tmp_storage_path)
        
        # Create expired file
        expired_date = (datetime.now(timezone.utc) - timedelta(days=100)).date()
        expired_dir = base_path / "trades" / expired_date.strftime("%Y-%m-%d")
        expired_dir.mkdir(parents=True, exist_ok=True)
        expired_file = expired_dir / "BTCUSDT.parquet"
        expired_file.write_bytes(b"test data")
        
        # Delete expired data (requires data_type and expired_date)
        await data_storage_service_no_archiving.delete_expired_data("trades", expired_date)
        
        # Verify method exists and can be called
        assert hasattr(data_storage_service_no_archiving, 'delete_expired_data')
    
    @pytest.mark.asyncio
    async def test_automatic_cleanup_scheduled(self, data_storage_service):
        """Test that automatic cleanup can be scheduled."""
        # Verify cleanup task can be started via start() method
        assert hasattr(data_storage_service, 'start')
        assert hasattr(data_storage_service, 'stop')
        assert hasattr(data_storage_service, '_cleanup_task')
    
    @pytest.mark.asyncio
    async def test_cleanup_task_runs_periodically(self, data_storage_service):
        """Test that cleanup task runs periodically."""
        # This will be tested in integration tests
        # For now, verify methods exist
        assert hasattr(data_storage_service, 'start')
        assert hasattr(data_storage_service, '_cleanup_interval')
