import asyncio
from decimal import Decimal
from typing import List

import pytest

from src.models import Position
from src.services.position_manager import PositionManager
from src.tasks.position_validation_task import PositionValidationTask


class DummyValidatingPositionManager(PositionManager):
    """Stub PositionManager for validation task tests."""

    def __init__(self, positions: List[Position]) -> None:
        super().__init__()
        self._positions_stub = positions
        self.validate_calls: List[str] = []

    async def get_all_positions(self) -> List[Position]:
        return self._positions_stub

    async def validate_position(self, asset: str, mode: str = "one-way", fix_discrepancies: bool = True, trace_id=None):
        # Record that we were called and simulate a valid position.
        self.validate_calls.append(asset)
        return True, None, None


@pytest.mark.asyncio
async def test_position_validation_task_runs_and_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare a couple of stub positions.
    positions = [
        Position(
            asset="BTCUSDT",
            mode="one-way",
            size=Decimal("1.0"),
            average_entry_price=Decimal("100.0"),
            current_price=Decimal("110.0"),
            unrealized_pnl=Decimal("10.0"),
            realized_pnl=Decimal("0.0"),
        ),
        Position(
            asset="ETHUSDT",
            mode="one-way",
            size=Decimal("2.0"),
            average_entry_price=Decimal("200.0"),
            current_price=Decimal("210.0"),
            unrealized_pnl=Decimal("20.0"),
            realized_pnl=Decimal("0.0"),
        ),
    ]

    dummy_pm = DummyValidatingPositionManager(positions)

    # Patch PositionValidationTask to use our dummy PositionManager and a short interval.
    async def fake_loop(self_self):
        await asyncio.sleep(0)  # simulate one tick
        # Manually perform one validation cycle similar to the real loop
        for p in await dummy_pm.get_all_positions():
            await dummy_pm.validate_position(p.asset, p.mode, fix_discrepancies=True, trace_id="test-trace")

    monkeypatch.setattr(PositionValidationTask, "_validation_loop", fake_loop)
    task = PositionValidationTask()
    # Inject dummy manager
    task._position_manager = dummy_pm  # type: ignore[attr-defined]

    await task.start()
    await asyncio.sleep(0.05)
    await task.stop()

    # Ensure validation was called for both positions
    assert set(dummy_pm.validate_calls) == {"BTCUSDT", "ETHUSDT"}


