from datetime import datetime
from decimal import Decimal
from typing import List

import pytest

from src.config import settings
from src.models import Position
from src.services.portfolio_manager import PortfolioManager


class DummyPositionManager:
    """Simple stub for PositionManager that returns a preconfigured list of positions."""

    def __init__(self, positions: List[Position]) -> None:
        self._positions = positions
        self.calls = 0

    async def get_all_positions(self) -> List[Position]:
        self.calls += 1
        return self._positions


def make_position(
    asset: str,
    size: str,
    current_price: str | None,
    unrealized: str = "0",
    realized: str = "0",
) -> Position:
    return Position(
        asset=asset,
        mode="one-way",
        size=Decimal(size),
        average_entry_price=Decimal("100.0"),
        current_price=Decimal(current_price) if current_price is not None else None,
        unrealized_pnl=Decimal(unrealized),
        realized_pnl=Decimal(realized),
    )


def test_calculate_metrics_empty_portfolio() -> None:
    manager = PortfolioManager(position_manager=DummyPositionManager([]))
    metrics = manager.calculate_metrics_from_positions([])

    assert metrics.total_exposure_usdt == Decimal("0")
    assert metrics.total_unrealized_pnl_usdt == Decimal("0")
    assert metrics.total_realized_pnl_usdt == Decimal("0")
    assert metrics.portfolio_value_usdt == Decimal("0")
    assert metrics.open_positions_count == 0
    assert metrics.long_positions_count == 0
    assert metrics.short_positions_count == 0
    assert metrics.net_exposure_usdt == Decimal("0")
    assert metrics.by_asset == {}


def test_calculate_metrics_with_positions_and_null_prices() -> None:
    positions = [
        make_position("BTCUSDT", "1.0", "100.0", unrealized="5", realized="2"),
        make_position("ETHUSDT", "-0.5", "200.0", unrealized="3", realized="-1"),
        # No current_price: should not affect exposure/portfolio_value
        make_position("XRPUSDT", "2.0", None, unrealized="1", realized="0"),
    ]
    dummy_pm = DummyPositionManager(positions)
    manager = PortfolioManager(position_manager=dummy_pm)

    metrics = manager.calculate_metrics_from_positions(positions)

    # Exposure: |1*100| + |(-0.5)*200| = 100 + 100 = 200
    assert metrics.total_exposure_usdt == Decimal("200")
    # Unrealized PnL: 5 + 3 + 1 = 9
    assert metrics.total_unrealized_pnl_usdt == Decimal("9")
    # Realized PnL: 2 - 1 + 0 = 1
    assert metrics.total_realized_pnl_usdt == Decimal("1")
    # Portfolio value: 1*100 + (-0.5)*200 = 100 - 100 = 0
    assert metrics.portfolio_value_usdt == Decimal("0")
    # Open positions: all sizes != 0
    assert metrics.open_positions_count == 3
    assert metrics.long_positions_count == 2  # BTC (1.0), XRP (2.0)
    assert metrics.short_positions_count == 1  # ETH (-0.5)

    # By-asset breakdown
    assert "BTCUSDT" in metrics.by_asset
    btc = metrics.by_asset["BTCUSDT"]
    assert btc.exposure_usdt == Decimal("100")
    assert btc.unrealized_pnl_usdt == Decimal("5")
    assert btc.realized_pnl_usdt == Decimal("2")
    assert btc.size == Decimal("1.0")

    assert "XRPUSDT" in metrics.by_asset
    # XRP exposure is 0 because current_price is NULL
    assert metrics.by_asset["XRPUSDT"].exposure_usdt == Decimal("0")


@pytest.mark.asyncio
async def test_portfolio_metrics_cache_hits_and_misses(monkeypatch) -> None:
    positions = [make_position("BTCUSDT", "1.0", "100.0")]
    dummy_pm = DummyPositionManager(positions)
    manager = PortfolioManager(position_manager=dummy_pm)

    # Force a small TTL to make test deterministic
    original_ttl = settings.settings.position_manager_metrics_cache_ttl
    settings.settings.position_manager_metrics_cache_ttl = 60

    try:
        # First call should hit underlying PositionManager
        metrics1 = await manager.get_portfolio_metrics(include_positions=False)
        assert dummy_pm.calls == 1

        # Second call with same parameters should be served from cache
        metrics2 = await manager.get_portfolio_metrics(include_positions=False)
        assert dummy_pm.calls == 1  # no new DB call
        assert metrics1.total_exposure_usdt == metrics2.total_exposure_usdt

        # Different cache key (include_positions=True) should trigger new fetch
        metrics3 = await manager.get_portfolio_metrics(include_positions=True)
        assert dummy_pm.calls == 2
        assert hasattr(metrics3, "_positions")
    finally:
        settings.settings.position_manager_metrics_cache_ttl = original_ttl


def test_portfolio_limit_exceeded_flag() -> None:
    """T073: limit_exceeded should reflect configured exposure soft limit."""
    positions = [make_position("BTCUSDT", "1.0", "100.0")]  # exposure 100
    dummy_pm = DummyPositionManager(positions)
    manager = PortfolioManager(position_manager=dummy_pm)

    original_limit = settings.settings.position_manager_portfolio_max_exposure_usdt
    settings.settings.position_manager_portfolio_max_exposure_usdt = 50.0

    try:
        metrics = manager.calculate_metrics_from_positions(positions)
        assert metrics.total_exposure_usdt == Decimal("100")
        assert metrics.limit_exceeded is True
    finally:
        settings.settings.position_manager_portfolio_max_exposure_usdt = original_limit

