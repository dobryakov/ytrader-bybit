"""Integration tests for sync-bybit endpoint with asset filtering."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.positions import get_position_manager
from src.config.settings import settings as app_settings
from src.services.position_manager import PositionManager


def make_test_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create test client with mocked dependencies."""
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    
    app = create_app()
    return TestClient(app)


@pytest.mark.asyncio
async def test_sync_bybit_without_asset_filter(monkeypatch: pytest.MonkeyPatch):
    """Test sync-bybit endpoint without asset filter (syncs all positions)."""
    client = make_test_client(monkeypatch)
    
    # Mock PositionManager.sync_positions_with_bybit
    mock_report = {
        "bybit_positions_count": 2,
        "local_positions_count": 2,
        "comparisons": [
            {
                "asset": "BTCUSDT",
                "bybit_exists": True,
                "local_exists": True,
                "size_match": True,
            },
            {
                "asset": "ETHUSDT",
                "bybit_exists": True,
                "local_exists": True,
                "size_match": True,
            },
        ],
        "updated": [],
        "created": [],
        "errors": [],
    }
    
    async def mock_sync(force=False, asset=None, trace_id=None):
        return mock_report
    
    app = create_app()
    mock_pm = MagicMock(spec=PositionManager)
    mock_pm.sync_positions_with_bybit = AsyncMock(side_effect=mock_sync)
    app.dependency_overrides[get_position_manager] = lambda: mock_pm
    
    test_client = TestClient(app)
    
    response = test_client.post(
        "/api/v1/positions/sync-bybit?force=true",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["bybit_positions_count"] == 2
    assert data["local_positions_count"] == 2
    assert len(data["comparisons"]) == 2
    
    # Verify sync was called without asset filter
    mock_pm.sync_positions_with_bybit.assert_called_once()
    call_kwargs = mock_pm.sync_positions_with_bybit.call_args[1]
    assert call_kwargs["force"] is True
    assert call_kwargs.get("asset") is None


@pytest.mark.asyncio
async def test_sync_bybit_with_asset_filter(monkeypatch: pytest.MonkeyPatch):
    """Test sync-bybit endpoint with asset filter (syncs only specific asset)."""
    client = make_test_client(monkeypatch)
    
    # Mock PositionManager.sync_positions_with_bybit
    mock_report = {
        "bybit_positions_count": 1,
        "local_positions_count": 1,
        "comparisons": [
            {
                "asset": "BTCUSDT",
                "bybit_exists": True,
                "local_exists": True,
                "size_match": True,
            },
        ],
        "updated": [],
        "created": [],
        "errors": [],
    }
    
    async def mock_sync(force=False, asset=None, trace_id=None):
        return mock_report
    
    app = create_app()
    mock_pm = MagicMock(spec=PositionManager)
    mock_pm.sync_positions_with_bybit = AsyncMock(side_effect=mock_sync)
    app.dependency_overrides[get_position_manager] = lambda: mock_pm
    
    test_client = TestClient(app)
    
    response = test_client.post(
        "/api/v1/positions/sync-bybit?force=true&asset=BTCUSDT",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["bybit_positions_count"] == 1
    assert data["local_positions_count"] == 1
    assert len(data["comparisons"]) == 1
    assert data["comparisons"][0]["asset"] == "BTCUSDT"
    
    # Verify sync was called with asset filter
    mock_pm.sync_positions_with_bybit.assert_called_once()
    call_kwargs = mock_pm.sync_positions_with_bybit.call_args[1]
    assert call_kwargs["force"] is True
    assert call_kwargs.get("asset") == "BTCUSDT"


@pytest.mark.asyncio
async def test_sync_bybit_without_force(monkeypatch: pytest.MonkeyPatch):
    """Test sync-bybit endpoint without force (only comparison, no updates)."""
    client = make_test_client(monkeypatch)
    
    # Mock PositionManager.sync_positions_with_bybit
    mock_report = {
        "bybit_positions_count": 1,
        "local_positions_count": 1,
        "comparisons": [
            {
                "asset": "BTCUSDT",
                "bybit_exists": True,
                "local_exists": True,
                "size_match": False,
                "bybit_size": "1.5",
                "local_size": "1.0",
            },
        ],
        "updated": [],
        "created": [],
        "errors": [],
    }
    
    async def mock_sync(force=False, asset=None, trace_id=None):
        return mock_report
    
    app = create_app()
    mock_pm = MagicMock(spec=PositionManager)
    mock_pm.sync_positions_with_bybit = AsyncMock(side_effect=mock_sync)
    app.dependency_overrides[get_position_manager] = lambda: mock_pm
    
    test_client = TestClient(app)
    
    response = test_client.post(
        "/api/v1/positions/sync-bybit?asset=BTCUSDT",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["bybit_positions_count"] == 1
    assert len(data["updated"]) == 0  # No updates when force=False
    
    # Verify sync was called without force
    mock_pm.sync_positions_with_bybit.assert_called_once()
    call_kwargs = mock_pm.sync_positions_with_bybit.call_args[1]
    assert call_kwargs["force"] is False
    assert call_kwargs.get("asset") == "BTCUSDT"

