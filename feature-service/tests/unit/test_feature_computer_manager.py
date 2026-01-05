"""
Unit tests for FeatureComputerManager.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.services.feature_computer_manager import FeatureComputerManager
from src.services.orderbook_manager import OrderbookManager
from src.models.rolling_windows import RollingWindows


class TestFeatureComputerManager:
    """Test FeatureComputerManager service."""
    
    @pytest.fixture
    def orderbook_manager(self):
        """Create orderbook manager."""
        return OrderbookManager()
    
    @pytest.fixture
    def mock_version_manager(self):
        """Create mocked FeatureRegistryVersionManager."""
        manager = MagicMock()
        
        # Minimal valid Feature Registry config
        minimal_config = {
            "version": "1.0.0",
            "features": [
                {
                    "name": "mid_price",
                    "input_sources": ["orderbook"],
                    "lookback_window": "0s",
                    "lookahead_forbidden": True,
                }
            ]
        }
        
        # Mock active version
        async def load_active_version():
            return minimal_config.copy()
        manager.load_active_version = AsyncMock(side_effect=load_active_version)
        
        # Mock load_version
        async def load_version(version):
            config = minimal_config.copy()
            config["version"] = version
            return config
        manager.load_version = AsyncMock(side_effect=load_version)
        
        return manager
    
    @pytest.fixture
    def shared_rolling_windows(self):
        """Create shared rolling windows dict."""
        return {}
    
    @pytest.fixture
    def manager(self, orderbook_manager, mock_version_manager, shared_rolling_windows):
        """Create FeatureComputerManager instance."""
        return FeatureComputerManager(
            orderbook_manager=orderbook_manager,
            feature_registry_version_manager=mock_version_manager,
            shared_rolling_windows=shared_rolling_windows,
        )
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_active_version(self, manager, mock_version_manager):
        """Test getting computer for active version (None)."""
        computer = await manager.get_or_create_computer(feature_registry_version=None)
        
        assert computer is not None
        assert computer._feature_registry_version == "1.0.0"
        mock_version_manager.load_active_version.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_specific_version(self, manager, mock_version_manager):
        """Test getting computer for specific version."""
        computer = await manager.get_or_create_computer(feature_registry_version="1.2.0")
        
        assert computer is not None
        assert computer._feature_registry_version == "1.2.0"
        mock_version_manager.load_version.assert_called_once_with("1.2.0")
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_caching(self, manager, mock_version_manager):
        """Test that computers are cached."""
        computer1 = await manager.get_or_create_computer(feature_registry_version="1.2.0")
        computer2 = await manager.get_or_create_computer(feature_registry_version="1.2.0")
        
        # Should be the same instance
        assert computer1 is computer2
        # load_version should be called only once
        assert mock_version_manager.load_version.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_different_versions(self, manager, mock_version_manager):
        """Test getting different versions creates different computers."""
        computer1 = await manager.get_or_create_computer(feature_registry_version="1.2.0")
        computer2 = await manager.get_or_create_computer(feature_registry_version="1.3.0")
        
        # Should be different instances
        assert computer1 is not computer2
        assert computer1._feature_registry_version == "1.2.0"
        assert computer2._feature_registry_version == "1.3.0"
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_uses_shared_windows(self, manager, shared_rolling_windows):
        """Test that computers use shared rolling windows."""
        computer = await manager.get_or_create_computer(feature_registry_version="1.2.0")
        
        # Computer should use shared windows
        assert computer._uses_shared_windows is True
        assert computer._rolling_windows is shared_rolling_windows
    
    @pytest.mark.asyncio
    async def test_get_shared_rolling_windows(self, manager, shared_rolling_windows):
        """Test getting shared rolling windows."""
        # Initially empty
        assert manager.get_shared_rolling_windows("BTCUSDT") is None
        
        # Create a rolling window
        from datetime import datetime, timezone
        import pandas as pd
        
        windows = {
            "1m": pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]),
        }
        rolling_window = RollingWindows(
            symbol="BTCUSDT",
            windows=windows,
            last_update=datetime.now(timezone.utc),
            window_intervals={"1m"},
            max_lookback_minutes_1m=30,
        )
        shared_rolling_windows["BTCUSDT"] = rolling_window
        
        # Should return the window
        result = manager.get_shared_rolling_windows("BTCUSDT")
        assert result is rolling_window
    
    @pytest.mark.asyncio
    async def test_get_or_create_computer_version_not_found(self, manager, mock_version_manager):
        """Test that version not found raises ValueError."""
        async def load_version_fail(version):
            raise FileNotFoundError(f"Version {version} not found")
        mock_version_manager.load_version = AsyncMock(side_effect=load_version_fail)
        
        with pytest.raises(ValueError, match="Feature Registry version 1.5.0 not found"):
            await manager.get_or_create_computer(feature_registry_version="1.5.0")

