from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.positions import get_position_manager
from src.config.settings import settings as app_settings
from src.models import Position
from src.services.position_manager import PositionManager


class InMemoryPositionManager(PositionManager):
    """In-memory PositionManager stub for E2E-style API flows."""

    def __init__(self) -> None:
        super().__init__()
        self._positions: dict[tuple[str, str], Position] = {}

    async def get_position(self, asset: str, mode: str = "one-way") -> Position | None:
        return self._positions.get((asset.upper(), mode))

    async def get_all_positions(self):
        return list(self._positions.values())

    async def update_position_from_order_fill(
        self,
        asset: str,
        size_delta: Decimal,
        execution_price: Decimal,
        execution_fees: Decimal | None = None,
        mode: str = "one-way",
    ) -> Position:
        key = (asset.upper(), mode)
        pos = self._positions.get(key)
        if not pos:
            pos = Position(
                asset=asset.upper(),
                mode=mode,
                size=Decimal("0"),
                average_entry_price=execution_price,
                current_price=execution_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
            )
        pos.size += size_delta
        pos.current_price = execution_price
        self._positions[key] = pos
        return pos


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Minimal env for app startup
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")

    pm = InMemoryPositionManager()
    app = create_app()
    app.dependency_overrides[get_position_manager] = lambda: pm  # type: ignore[return-value]
    return TestClient(app)


def test_order_execution_update_flow(client: TestClient) -> None:
    """E2E-style flow: after synthetic order fill, position appears in API."""
    # Simulate that PositionManager received an order fill update.
    pm: InMemoryPositionManager = client.app.dependency_overrides[get_position_manager]()  # type: ignore[call-arg]
    client.loop = None  # type: ignore[attr-defined]

    import asyncio

    asyncio.get_event_loop().run_until_complete(
        pm.update_position_from_order_fill(
            asset="BTCUSDT",
            size_delta=Decimal("1.0"),
            execution_price=Decimal("30000.0"),
            execution_fees=None,
            mode="one-way",
        )
    )

    # Now query the position via API
    resp = client.get(
        "/api/v1/positions/BTCUSDT?mode=one-way",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "BTCUSDT"
    assert data["size"] == "1.0"


