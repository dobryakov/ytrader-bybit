from datetime import datetime
from decimal import Decimal
from typing import List

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes.portfolio import get_portfolio_manager
from src.config.settings import settings as app_settings
from src.models import Position
from src.services.portfolio_manager import PortfolioManager


class DummyPositionManager:
    """Stub PositionManager that returns a fixed list of positions."""

    def __init__(self, positions: List[Position]) -> None:
        self._positions = positions

    async def get_all_positions(self) -> List[Position]:
        return self._positions


def make_position(asset: str, size: str, current_price: str) -> Position:
    return Position(
        asset=asset,
        mode="one-way",
        size=Decimal(size),
        average_entry_price=Decimal("100.0"),
        current_price=Decimal(current_price),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Ensure required DB env variables exist even in isolated test runs.
    monkeypatch.setenv("POSTGRES_DB", "test")
    monkeypatch.setenv("POSTGRES_USER", "test")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")

    # Use dummy positions for risk-management style queries.
    positions = [
        make_position("BTCUSDT", "1.0", "100.0"),
        make_position("ETHUSDT", "2.0", "200.0"),
    ]
    dummy_pm = DummyPositionManager(positions)
    portfolio_manager = PortfolioManager(position_manager=dummy_pm)

    app = create_app()

    # Override dependency so integration tests do not touch the real DB.
    app.dependency_overrides[get_portfolio_manager] = lambda: portfolio_manager
    return TestClient(app)


def test_portfolio_exposure_endpoint_for_risk_management(client: TestClient) -> None:
    response = client.get(
        "/api/v1/portfolio/exposure",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 200
    data = response.json()

    # total_exposure_usdt should be numeric and calculated_at must be present.
    assert "total_exposure_usdt" in data
    assert "calculated_at" in data


def test_portfolio_metrics_for_limit_checking(client: TestClient) -> None:
    response = client.get(
        "/api/v1/portfolio",
        headers={"X-API-Key": app_settings.position_manager_api_key},
    )
    assert response.status_code == 200
    data = response.json()

    # Core metrics available for limit checking.
    assert "total_exposure_usdt" in data
    assert "total_unrealized_pnl_usdt" in data
    assert "total_realized_pnl_usdt" in data
    assert "by_asset" in data


