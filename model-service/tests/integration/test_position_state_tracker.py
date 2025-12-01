"""
Integration tests for PositionStateTracker.
"""

import pytest
from datetime import datetime
from src.services.position_state_tracker import position_state_tracker
from src.models.position_state_tracker import PositionState


@pytest.mark.asyncio
async def test_create_and_get_position_state(db_pool):
    """Test creating and retrieving position state."""
    asset = "BTCUSDT"
    position_data = {
        "asset": asset,
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,
        "avg_price": 50000.0,
    }

    # Create state
    state = await position_state_tracker.get_or_create_state(asset, position_data)

    assert state.asset == asset
    assert state.entry_price > 0
    assert state.entry_time is not None
    assert state.peak_price > 0

    # Clean up
    await position_state_tracker._remove_from_database(asset)


@pytest.mark.asyncio
async def test_update_position_state(db_pool):
    """Test updating position state."""
    asset = "BTCUSDT"
    position_data = {
        "asset": asset,
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,
        "avg_price": 50000.0,
    }

    # Create state
    state1 = await position_state_tracker.get_or_create_state(asset, position_data)
    initial_peak = state1.peak_price

    # Update with higher PnL
    position_data2 = {
        "asset": asset,
        "size": 1000.0,
        "unrealized_pnl_pct": 3.5,  # Higher PnL
        "avg_price": 50000.0,
    }

    state2 = await position_state_tracker.update_state(asset, position_data2)

    assert state2.asset == asset
    # Peak price should be updated if current price is higher
    assert state2.peak_price >= initial_peak

    # Clean up
    await position_state_tracker._remove_from_database(asset)


@pytest.mark.asyncio
async def test_mark_exit_signal_sent(db_pool):
    """Test marking exit signal as sent."""
    asset = "BTCUSDT"
    position_data = {
        "asset": asset,
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,
        "avg_price": 50000.0,
    }

    # Create state
    await position_state_tracker.get_or_create_state(asset, position_data)

    # Mark exit signal sent
    await position_state_tracker.mark_exit_signal_sent(asset)

    # Retrieve state and verify
    state = await position_state_tracker._load_from_database(asset)
    assert state is not None
    assert state.last_exit_signal_time is not None

    # Clean up
    await position_state_tracker._remove_from_database(asset)


@pytest.mark.asyncio
async def test_remove_position_state(db_pool):
    """Test removing position state."""
    asset = "BTCUSDT"
    position_data = {
        "asset": asset,
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,
        "avg_price": 50000.0,
    }

    # Create state
    await position_state_tracker.get_or_create_state(asset, position_data)

    # Verify it exists
    state = await position_state_tracker._load_from_database(asset)
    assert state is not None

    # Remove state
    await position_state_tracker.remove_state(asset)

    # Verify it's gone
    state = await position_state_tracker._load_from_database(asset)
    assert state is None

