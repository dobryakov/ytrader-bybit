from datetime import datetime, timedelta
from decimal import Decimal

from src.models import Position
from src.services.position_manager import PositionManager


def make_position(
    asset: str = "BTCUSDT",
    mode: str = "one-way",
    size: str = "1.0",
    current_price: str | None = "110.0",
) -> Position:
    now = datetime.utcnow()
    return Position(
        asset=asset,
        mode=mode,
        size=Decimal(size),
        average_entry_price=Decimal("100.0"),
        current_price=Decimal(current_price) if current_price is not None else None,
        unrealized_pnl=Decimal("10.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
    )


def test_calculate_unrealized_pnl_pct_basic() -> None:
    manager = PositionManager()
    position = make_position(size="1.0", current_price="110.0")

    pct = manager.calculate_unrealized_pnl_pct(position)
    assert pct is not None
    assert round(pct, 2) == Decimal("10.00")


def test_calculate_unrealized_pnl_pct_handles_zero_size() -> None:
    manager = PositionManager()
    position = make_position(size="0.0", current_price="110.0")

    pct = manager.calculate_unrealized_pnl_pct(position)
    assert pct is None


def test_calculate_time_held_minutes() -> None:
    manager = PositionManager()
    now = datetime.utcnow()
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("100.0"),
        current_price=Decimal("110.0"),
        unrealized_pnl=Decimal("10.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=42),
        last_updated=now,
    )

    minutes = manager.calculate_time_held_minutes(position)
    assert minutes is not None
    assert 40 <= minutes <= 45


def test_calculate_position_size_norm() -> None:
    manager = PositionManager()
    position = make_position(size="2.0", current_price="150.0")

    norm = manager.calculate_position_size_norm(position, total_exposure=Decimal("1000.0"))
    assert norm is not None
    # exposure = 2 * 150 = 300; 300/1000 = 0.3
    assert round(norm, 2) == Decimal("0.30")


def test_calculate_position_size_norm_no_price_or_zero_exposure() -> None:
    manager = PositionManager()

    # No current price -> None
    position_no_price = make_position(size="2.0", current_price=None)
    assert manager.calculate_position_size_norm(position_no_price, total_exposure=Decimal("1000.0")) is None

    # Zero total exposure -> None
    position = make_position(size="2.0", current_price="150.0")
    assert manager.calculate_position_size_norm(position, total_exposure=Decimal("0")) is None


def test_position_filters() -> None:
    manager = PositionManager()
    positions = [
        make_position(asset="BTCUSDT", mode="one-way", size="1.0"),
        make_position(asset="ETHUSDT", mode="hedge", size="-0.5"),
        make_position(asset="BTCUSDT", mode="hedge", size="0.1"),
    ]

    by_asset = manager.filter_by_asset(positions, "BTCUSDT")
    assert all(p.asset == "BTCUSDT" for p in by_asset)

    by_mode = manager.filter_by_mode(positions, "hedge")
    assert all(p.mode == "hedge" for p in by_mode)

    by_size = manager.filter_by_size(positions, size_min=Decimal("0.0"), size_max=Decimal("1.0"))
    assert all(Decimal("0.0") <= p.size <= Decimal("1.0") for p in by_size)

