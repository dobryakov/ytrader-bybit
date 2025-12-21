from datetime import datetime, timedelta
from decimal import Decimal

import pytest

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


@pytest.mark.asyncio
async def test_external_price_api_success(monkeypatch) -> None:
    """PositionManager should parse markPrice from external API response."""
    manager = PositionManager()

    class DummyResponse:
        def __init__(self, json_data: dict) -> None:
            self._json = json_data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._json

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "DummyClient":
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def get(self, url, params=None):
            return DummyResponse(
                {
                    "result": {
                        "list": [
                            {
                                "markPrice": "12345.67",
                            }
                        ]
                    }
                }
            )

    import src.services.position_manager as pm_module

    # Patch module-level `httpx` symbol used by PositionManager.
    monkeypatch.setitem(pm_module.__dict__, "httpx", type("M", (), {"AsyncClient": DummyClient}))

    price = await manager._get_current_price_from_api("BTCUSDT", trace_id="test-trace")
    assert price == Decimal("12345.67")


@pytest.mark.asyncio
async def test_external_price_api_retry_and_fallback(monkeypatch) -> None:
    """On repeated failures, external price API helper should return None."""
    manager = PositionManager()

    class FailingClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FailingClient":
            return self

        async def __aexit__(self, *exc_info) -> None:
            return None

        async def get(self, url, params=None):
            raise RuntimeError("network failure")

    import src.services.position_manager as pm_module

    monkeypatch.setitem(pm_module.__dict__, "httpx", type("M", (), {"AsyncClient": FailingClient}))

    price = await manager._get_current_price_from_api("BTCUSDT", trace_id="test-trace")
    assert price is None


def test_resolve_avg_price_under_threshold_keeps_existing(monkeypatch) -> None:
    """_resolve_avg_price should keep existing value when diff below threshold."""
    manager = PositionManager()
    existing = Decimal("50000.00")
    ws_avg = Decimal("50049.00")  # < 0.1% diff

    # Ensure threshold is 0.1%
    from src.config.settings import settings

    monkeypatch.setattr(settings, "position_manager_avg_price_diff_threshold", 0.001)

    resolved = manager._resolve_avg_price(existing_avg_price=existing, ws_avg_price=ws_avg)  # type: ignore[attr-defined]
    assert resolved == existing


def test_resolve_avg_price_over_threshold_uses_ws(monkeypatch) -> None:
    """_resolve_avg_price should switch to ws_avg when diff above threshold."""
    manager = PositionManager()
    existing = Decimal("50000.00")
    ws_avg = Decimal("50500.00")  # 1% diff

    from src.config.settings import settings

    monkeypatch.setattr(settings, "position_manager_avg_price_diff_threshold", 0.001)

    resolved = manager._resolve_avg_price(existing_avg_price=existing, ws_avg_price=ws_avg)  # type: ignore[attr-defined]
    assert resolved == ws_avg


def test_has_size_discrepancy() -> None:
    """_has_size_discrepancy should respect POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD."""
    manager = PositionManager()

    db_size = Decimal("1.0")
    ws_close = Decimal("1.00005")  # below default 0.0001 threshold
    ws_far = Decimal("1.001")  # above threshold

    assert not manager._has_size_discrepancy(db_size, ws_close)  # type: ignore[attr-defined]
    assert manager._has_size_discrepancy(db_size, ws_far)  # type: ignore[attr-defined]


def test_timestamp_conflict_resolution_ws_fresher(monkeypatch) -> None:
    """_should_update_size_from_ws should return True - WebSocket is always source of truth."""
    manager = PositionManager()

    from src.config.settings import settings

    monkeypatch.setattr(settings, "position_manager_size_validation_threshold", 0.0001)

    db_size = Decimal("1.0")
    ws_size = Decimal("1.5")
    order_ts = datetime.utcnow()
    ws_ts = order_ts + timedelta(seconds=1)

    # NOTE: WebSocket is always the source of truth, regardless of timestamps
    assert manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=ws_size,
        ws_timestamp=ws_ts,
        order_timestamp=order_ts,
    )


def test_timestamp_conflict_resolution_order_fresher(monkeypatch) -> None:
    """_should_update_size_from_ws should return True - WebSocket is always source of truth."""
    manager = PositionManager()

    from src.config.settings import settings

    monkeypatch.setattr(settings, "position_manager_size_validation_threshold", 0.0001)

    db_size = Decimal("1.0")
    ws_size = Decimal("1.5")
    ws_ts = datetime.utcnow()
    order_ts = ws_ts + timedelta(seconds=1)

    # NOTE: WebSocket is always the source of truth, regardless of order_timestamp
    assert manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=ws_size,
        ws_timestamp=ws_ts,
        order_timestamp=order_ts,
    )


def test_timestamp_conflict_resolution_missing_timestamps(monkeypatch) -> None:
    """_should_update_size_from_ws handles missing timestamps - WebSocket is source of truth."""
    manager = PositionManager()

    from src.config.settings import settings

    monkeypatch.setattr(settings, "position_manager_size_validation_threshold", 0.0001)

    db_size = Decimal("1.0")
    ws_size = Decimal("1.5")
    now = datetime.utcnow()

    # Missing WS timestamp (ws_size is None effectively means no update)
    # If ws_size is None, method returns False
    assert not manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=None,  # No WebSocket size means no update
        ws_timestamp=None,
        order_timestamp=now,
    )

    # Missing Order Manager timestamp - doesn't matter, WebSocket is source of truth
    # With ws_size available and significant discrepancy, should return True
    assert manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=ws_size,
        ws_timestamp=now,
        order_timestamp=None,  # order_timestamp is deprecated and ignored
    )


def test_timestamp_conflict_resolution_disabled_feature(monkeypatch) -> None:
    """_should_update_size_from_ws always uses WebSocket as source of truth (feature flag removed)."""
    manager = PositionManager()

    from src.config.settings import settings

    # NOTE: position_manager_enable_timestamp_resolution is no longer used
    monkeypatch.setattr(settings, "position_manager_size_validation_threshold", 0.0001)

    db_size = Decimal("1.0")
    ws_size = Decimal("1.5")
    order_ts = datetime.utcnow()
    ws_ts = order_ts + timedelta(seconds=1)

    # WebSocket is always source of truth, regardless of feature flags
    assert manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=ws_size,
        ws_timestamp=ws_ts,
        order_timestamp=order_ts,
    )
    
    # Test case: no significant discrepancy (below threshold)
    small_ws_size = db_size + Decimal("0.00001")  # Very small difference
    assert not manager._should_update_size_from_ws(  # type: ignore[attr-defined]
        db_size=db_size,
        ws_size=small_ws_size,
        ws_timestamp=ws_ts,
        order_timestamp=order_ts,
    )

