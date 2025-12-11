"""
Unit tests for Feature Registry backward compatibility checking.
"""
import pytest
import tempfile
import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.services.feature_registry_version_manager import (
    FeatureRegistryVersionManager,
    CompatibilityReport,
)
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


class TestBackwardCompatibility:
    """Tests for backward compatibility checking."""
    
    def test_compatibility_report_no_changes(self, version_manager):
        """Test compatibility report with no changes."""
        config = get_valid_feature_registry_config()
        
        report = version_manager.check_backward_compatibility(
            old_config=config,
            new_config=config,
        )
        
        assert not report.has_breaking_changes
        assert not report.has_warnings
    
    def test_compatibility_report_removed_feature(self, version_manager):
        """Test compatibility report detects removed features."""
        old_config = get_valid_feature_registry_config()
        new_config = old_config.copy()
        
        # Remove a feature
        new_config["features"] = new_config["features"][:1]  # Keep only first feature
        
        report = version_manager.check_backward_compatibility(
            old_config=old_config,
            new_config=new_config,
        )
        
        assert report.has_breaking_changes
        assert len(report.breaking_changes) > 0
        assert any("removed" in change.lower() for change in report.breaking_changes)
    
    def test_compatibility_report_changed_input_sources(self, version_manager):
        """Test compatibility report detects changed input_sources."""
        old_config = get_valid_feature_registry_config()
        new_config = copy.deepcopy(old_config)  # Deep copy to avoid mutating original
        
        # Change input_sources for first feature
        if new_config["features"]:
            new_config["features"][0]["input_sources"] = ["kline"]  # Changed from original
        
        report = version_manager.check_backward_compatibility(
            old_config=old_config,
            new_config=new_config,
        )
        
        assert report.has_breaking_changes
        assert any("input_sources" in change for change in report.breaking_changes)
    
    def test_compatibility_report_changed_lookback_window(self, version_manager):
        """Test compatibility report detects changed lookback_window (warning)."""
        old_config = get_valid_feature_registry_config()
        new_config = copy.deepcopy(old_config)  # Deep copy to avoid mutating original
        
        # Change lookback_window for first feature
        if new_config["features"]:
            new_config["features"][0]["lookback_window"] = "5m"  # Changed from original
        
        report = version_manager.check_backward_compatibility(
            old_config=old_config,
            new_config=new_config,
        )
        
        assert not report.has_breaking_changes  # Lookback window change is warning, not breaking
        assert report.has_warnings
        assert any("lookback_window" in warning for warning in report.compatibility_warnings)
    
    def test_compatibility_report_changed_max_lookback_days(self, version_manager):
        """Test compatibility report detects changed max_lookback_days (warning)."""
        old_config = get_valid_feature_registry_config()
        new_config = copy.deepcopy(old_config)  # Deep copy to avoid mutating original
        
        # Change max_lookback_days for first feature
        if new_config["features"]:
            new_config["features"][0]["max_lookback_days"] = 5  # Changed from original
        
        report = version_manager.check_backward_compatibility(
            old_config=old_config,
            new_config=new_config,
        )
        
        assert not report.has_breaking_changes
        assert report.has_warnings
        assert any("max_lookback_days" in warning for warning in report.compatibility_warnings)
    
    def test_compatibility_report_new_feature(self, version_manager):
        """Test compatibility report with new feature (not breaking)."""
        old_config = get_valid_feature_registry_config()
        new_config = copy.deepcopy(old_config)  # Deep copy to avoid mutating original
        
        # Add new feature
        new_feature = {
            "name": "new_feature",
            "input_sources": ["orderbook"],
            "lookback_window": "0s",
            "lookahead_forbidden": True,
            "max_lookback_days": 0,
            "data_sources": [{"source": "orderbook", "timestamp_required": True}],
        }
        new_config["features"].append(new_feature)
        
        report = version_manager.check_backward_compatibility(
            old_config=old_config,
            new_config=new_config,
        )
        
        # New features are not breaking changes
        assert not report.has_breaking_changes
    
    @pytest.mark.asyncio
    async def test_activate_version_with_breaking_changes_requires_acknowledgment(
        self, version_manager, mock_metadata_storage, temp_versions_dir
    ):
        """Test activate_version requires acknowledgment for breaking changes."""
        import yaml
        
        # Create old config
        old_config = get_valid_feature_registry_config()
        old_version = old_config["version"]
        old_file_path = version_manager.get_version_file_path(old_version)
        with open(old_file_path, "w") as f:
            yaml.dump(old_config, f)
        
        # Create new config with removed feature
        new_config = old_config.copy()
        new_config["version"] = "1.1.0"
        new_config["features"] = new_config["features"][:1]  # Remove feature
        
        new_version = new_config["version"]
        new_file_path = version_manager.get_version_file_path(new_version)
        with open(new_file_path, "w") as f:
            yaml.dump(new_config, f)
        
        # Mock DB responses
        mock_metadata_storage.get_active_feature_registry_version = AsyncMock(
            return_value={
                "version": old_version,
                "file_path": str(old_file_path),
                "is_active": True,
            }
        )
        mock_metadata_storage.get_feature_registry_version = AsyncMock(
            return_value={
                "version": new_version,
                "file_path": str(new_file_path),
                "is_active": False,
            }
        )
        
        # Try to activate without acknowledgment
        with pytest.raises(ValueError, match="Breaking changes detected"):
            await version_manager.activate_version(
                version=new_version,
                acknowledge_breaking_changes=False,
            )
        
        # Activate with acknowledgment
        mock_metadata_storage.activate_feature_registry_version = AsyncMock(
            return_value={
                "version": new_version,
                "is_active": True,
            }
        )
        
        result = await version_manager.activate_version(
            version=new_version,
            acknowledge_breaking_changes=True,
        )
        
        assert result["is_active"] is True
        # Verify metadata was updated
        mock_metadata_storage.update_feature_registry_version_metadata.assert_called_once()

