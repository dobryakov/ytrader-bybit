"""E2E tests for unified positions architecture.

Tests the complete flow of position management with unified positions table:
- Active and closed positions in single table
- Position creation, update, and closure
- Position reopening (creates new record)
- Race condition protection
"""

from decimal import Decimal
from datetime import datetime
import pytest
import asyncio

from src.models import Position
from src.services.position_manager import PositionManager
from src.config.database import DatabaseConnection


@pytest.mark.asyncio
async def test_create_active_position():
    """E2E test: Create new active position via update_position_from_websocket."""
    manager = PositionManager()
    
    # Create new position
    await manager.update_position_from_websocket(
        asset="BTCUSDT",
        mode="one-way",
        size_from_ws=Decimal("1.5"),
        avg_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("150.0"),
        trace_id="test-e2e-001",
    )
    
    # Verify position was created
    position = await manager.get_active_position("BTCUSDT", "one-way")
    assert position is not None
    assert position.asset == "BTCUSDT"
    assert position.mode == "one-way"
    assert position.size == Decimal("1.5")
    assert position.average_entry_price == Decimal("50000.0")
    assert position.unrealized_pnl == Decimal("150.0")
    assert position.closed_at is None  # Active position
    assert position.created_at is not None


@pytest.mark.asyncio
async def test_close_position():
    """E2E test: Close position by setting size to 0."""
    manager = PositionManager()
    
    # Create active position first
    await manager.update_position_from_websocket(
        asset="ETHUSDT",
        mode="one-way",
        size_from_ws=Decimal("2.0"),
        avg_price=Decimal("3000.0"),
        trace_id="test-e2e-002",
    )
    
    # Close position
    await manager.update_position_from_websocket(
        asset="ETHUSDT",
        mode="one-way",
        size_from_ws=Decimal("0.0"),
        avg_price=None,
        unrealized_pnl=Decimal("0.0"),
        trace_id="test-e2e-002",
    )
    
    # Verify position is closed
    active_position = await manager.get_active_position("ETHUSDT", "one-way")
    assert active_position is None  # No active position
    
    # Verify closed position exists in history
    closed_positions = await manager.get_position_history("ETHUSDT", "one-way", limit=1)
    assert len(closed_positions) > 0
    closed = closed_positions[0]
    assert closed.asset == "ETHUSDT"
    assert closed.size == Decimal("0")
    assert closed.closed_at is not None
    assert closed.is_closed is True


@pytest.mark.asyncio
async def test_reopen_position_creates_new_record():
    """E2E test: Reopening closed position creates new record (not updates existing)."""
    manager = PositionManager()
    
    # Create and close position
    await manager.update_position_from_websocket(
        asset="SOLUSDT",
        mode="one-way",
        size_from_ws=Decimal("10.0"),
        avg_price=Decimal("100.0"),
        trace_id="test-e2e-003",
    )
    
    first_position = await manager.get_active_position("SOLUSDT", "one-way")
    first_id = first_position.id
    
    # Close position
    await manager.update_position_from_websocket(
        asset="SOLUSDT",
        mode="one-way",
        size_from_ws=Decimal("0.0"),
        trace_id="test-e2e-003",
    )
    
    # Reopen position
    await manager.update_position_from_websocket(
        asset="SOLUSDT",
        mode="one-way",
        size_from_ws=Decimal("5.0"),
        avg_price=Decimal("105.0"),
        trace_id="test-e2e-003",
    )
    
    # Verify new position was created (different ID)
    new_position = await manager.get_active_position("SOLUSDT", "one-way")
    assert new_position is not None
    assert new_position.id != first_id  # New record created
    assert new_position.size == Decimal("5.0")
    assert new_position.closed_at is None  # Active
    
    # Verify old position is still closed
    closed_positions = await manager.get_position_history("SOLUSDT", "one-way", limit=10)
    old_closed = [p for p in closed_positions if p.id == first_id]
    assert len(old_closed) == 1
    assert old_closed[0].closed_at is not None


@pytest.mark.asyncio
async def test_unique_index_prevents_duplicate_active_positions():
    """E2E test: Unique index prevents creating duplicate active positions."""
    manager = PositionManager()
    
    # Create active position
    await manager.update_position_from_websocket(
        asset="ADAUSDT",
        mode="one-way",
        size_from_ws=Decimal("100.0"),
        avg_price=Decimal("0.5"),
        trace_id="test-e2e-004",
    )
    
    # Try to create duplicate (should update existing instead)
    # This tests the unique index protection
    await manager.update_position_from_websocket(
        asset="ADAUSDT",
        mode="one-way",
        size_from_ws=Decimal("150.0"),
        avg_price=Decimal("0.55"),
        trace_id="test-e2e-004",
    )
    
    # Verify only one active position exists
    position = await manager.get_active_position("ADAUSDT", "one-way")
    assert position is not None
    assert position.size == Decimal("150.0")  # Updated, not duplicated
    
    # Verify only one active position in total
    all_active = await manager.get_all_active_positions()
    ada_active = [p for p in all_active if p.asset == "ADAUSDT" and p.mode == "one-way"]
    assert len(ada_active) == 1


@pytest.mark.asyncio
async def test_get_position_history():
    """E2E test: Retrieve position history for closed positions."""
    manager = PositionManager()
    
    # Create and close multiple positions
    for i in range(3):
        await manager.update_position_from_websocket(
            asset="DOTUSDT",
            mode="one-way",
            size_from_ws=Decimal("20.0"),
            avg_price=Decimal("7.0"),
            trace_id=f"test-e2e-005-{i}",
        )
        await manager.update_position_from_websocket(
            asset="DOTUSDT",
            mode="one-way",
            size_from_ws=Decimal("0.0"),
            trace_id=f"test-e2e-005-{i}",
        )
    
    # Get history
    history = await manager.get_position_history("DOTUSDT", "one-way", limit=10)
    
    # Verify we have closed positions
    assert len(history) >= 3
    for pos in history:
        assert pos.is_closed is True
        assert pos.closed_at is not None
        assert pos.size == Decimal("0")


@pytest.mark.asyncio
async def test_position_computed_properties():
    """E2E test: Verify computed properties (is_active, is_closed, total_pnl) work correctly."""
    manager = PositionManager()
    
    # Create active position
    await manager.update_position_from_websocket(
        asset="LINKUSDT",
        mode="one-way",
        size_from_ws=Decimal("50.0"),
        avg_price=Decimal("15.0"),
        unrealized_pnl=Decimal("25.0"),
        realized_pnl=Decimal("10.0"),
        trace_id="test-e2e-006",
    )
    
    position = await manager.get_active_position("LINKUSDT", "one-way")
    assert position.is_active is True
    assert position.is_closed is False
    assert position.total_pnl == Decimal("35.0")  # unrealized + realized
    assert position.holding_time_minutes is not None
    
    # Close position
    await manager.update_position_from_websocket(
        asset="LINKUSDT",
        mode="one-way",
        size_from_ws=Decimal("0.0"),
        unrealized_pnl=Decimal("0.0"),
        realized_pnl=Decimal("35.0"),
        trace_id="test-e2e-006",
    )
    
    # Verify closed position properties
    history = await manager.get_position_history("LINKUSDT", "one-way", limit=1)
    closed = history[0]
    assert closed.is_active is False
    assert closed.is_closed is True
    assert closed.total_pnl == Decimal("35.0")


@pytest.mark.asyncio
async def test_concurrent_updates_use_optimistic_locking():
    """E2E test: Concurrent updates use optimistic locking with retry logic."""
    manager = PositionManager()
    
    # Create position
    await manager.update_position_from_websocket(
        asset="MATICUSDT",
        mode="one-way",
        size_from_ws=Decimal("1000.0"),
        avg_price=Decimal("0.8"),
        trace_id="test-e2e-007",
    )
    
    # Simulate concurrent updates (in real scenario, these would come from different processes)
    # Both try to update the same position
    async def update_1():
        return await manager.update_position_from_websocket(
            asset="MATICUSDT",
            mode="one-way",
            size_from_ws=Decimal("1100.0"),
            trace_id="test-e2e-007-1",
        )
    
    async def update_2():
        return await manager.update_position_from_websocket(
            asset="MATICUSDT",
            mode="one-way",
            size_from_ws=Decimal("1200.0"),
            trace_id="test-e2e-007-2",
        )
    
    # Execute concurrently
    results = await asyncio.gather(update_1(), update_2(), return_exceptions=True)
    
    # Verify no exceptions (retry logic should handle conflicts)
    for result in results:
        assert not isinstance(result, Exception)
    
    # Verify final state is consistent (one of the updates succeeded)
    final_position = await manager.get_active_position("MATICUSDT", "one-way")
    assert final_position is not None
    assert final_position.size in [Decimal("1100.0"), Decimal("1200.0")]


@pytest.mark.asyncio
async def test_get_all_active_positions():
    """E2E test: Get all active positions excludes closed ones."""
    manager = PositionManager()
    
    # Create multiple active positions
    assets = ["AVAXUSDT", "ATOMUSDT", "ALGOUSDT"]
    for asset in assets:
        await manager.update_position_from_websocket(
            asset=asset,
            mode="one-way",
            size_from_ws=Decimal("10.0"),
            avg_price=Decimal("20.0"),
            trace_id=f"test-e2e-008-{asset}",
        )
    
    # Close one
    await manager.update_position_from_websocket(
        asset="AVAXUSDT",
        mode="one-way",
        size_from_ws=Decimal("0.0"),
        trace_id="test-e2e-008-AVAXUSDT",
    )
    
    # Get all active positions
    active = await manager.get_all_active_positions()
    
    # Verify closed position is not in active list
    avax_active = [p for p in active if p.asset == "AVAXUSDT"]
    assert len(avax_active) == 0
    
    # Verify other positions are active
    atom_active = [p for p in active if p.asset == "ATOMUSDT"]
    algo_active = [p for p in active if p.asset == "ALGOUSDT"]
    assert len(atom_active) == 1
    assert len(algo_active) == 1

