from datetime import datetime
from decimal import Decimal
from typing import List

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.positions import get_position_manager
from src.config.settings import settings as app_settings
from src.models import Position, PositionSnapshot
from src.services.position_manager import PositionManager


class DummySnapshotPositionManager:
    """Stub PositionManager for snapshot API tests (no real DB)."""

    def __init__(self, positions: List[Position], snapshots: List[PositionSnapshot]) -> None:
        self._positions = positions
        self._snapshots = snapshots
        self.created_snapshots: List[PositionSnapshot] = []

    async def get_position(self, asset: str, mode: str = "one-way") -> Position | None:
        for p in self._positions:
            if p.asset == asset.upper() and p.mode == mode:
                return p
        return None

    async def get_position_snapshots(
        self,
        position_id,
        limit: int,
        offset: int,
    ) -> List[PositionSnapshot]:
        # Simple slice-based pagination for test data.
        relevant = [s for s in self._snapshots if s.position_id == position_id]
        return relevant[offset : offset + limit]

    async def create_position_snapshot(self, position: Position, trace_id: str | None = None) -> PositionSnapshot:
        """Simulate snapshot creation by appending to in-memory list."""
        snapshot = make_snapshot(position)
        self._snapshots.append(snapshot)
        self.created_snapshots.append(snapshot)
        return snapshot


def make_position(asset: str = "BTCUSDT") -> Position:
    now = datetime.utcnow()
    return Position(
        asset=asset,
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        current_price=Decimal("110.0"),
        unrealized_pnl=Decimal("10.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now,
        last_updated=now,
    )


def make_snapshot(position: Position) -> PositionSnapshot:
    return PositionSnapshot(
        position_id=position.id,
        asset=position.asset,
        mode=position.mode,
        snapshot_data=position.to_db_dict(),
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Ensure required DB env variables exist even in isolated test runs.
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")

    # Prepare in-memory positions and snapshots.
    position = make_position()
    snapshots = [make_snapshot(position) for _ in range(3)]

    dummy_pm = DummySnapshotPositionManager([position], snapshots)

    app = create_app()
    # Override dependency so that tests do not hit the real database.
    app.dependency_overrides[get_position_manager] = lambda: dummy_pm  # type: ignore[return-value]
    return TestClient(app)


def test_snapshot_history_query(client: TestClient) -> None:
    response = client.get(
        "/api/v1/positions/BTCUSDT/snapshots?limit=2&offset=0",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 200
    data = response.json()

    assert "snapshots" in data
    assert "count" in data
    assert len(data["snapshots"]) == 2


def test_snapshot_history_position_not_found(client: TestClient) -> None:
    response = client.get(
        "/api/v1/positions/ETHUSDT/snapshots",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 404


def test_snapshot_history_invalid_mode(client: TestClient) -> None:
    response = client.get(
        "/api/v1/positions/BTCUSDT/snapshots?mode=invalid",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 400


def test_snapshot_creation_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /snapshot should create a snapshot when position exists."""
    # Minimal env for app startup.
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")

    position = make_position()
    snapshots: List[PositionSnapshot] = []
    dummy_pm = DummySnapshotPositionManager([position], snapshots)

    app = create_app()
    app.dependency_overrides[get_position_manager] = lambda: dummy_pm  # type: ignore[return-value]
    client = TestClient(app)

    response = client.post(
        "/api/v1/positions/BTCUSDT/snapshot",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["position_id"] == str(position.id)
    assert len(dummy_pm.created_snapshots) == 1


def test_snapshot_creation_position_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """POST /snapshot should return 404 when position does not exist."""
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")

    dummy_pm = DummySnapshotPositionManager([], [])

    app = create_app()
    app.dependency_overrides[get_position_manager] = lambda: dummy_pm  # type: ignore[return-value]
    client = TestClient(app)

    response = client.post(
        "/api/v1/positions/BTCUSDT/snapshot",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 404


