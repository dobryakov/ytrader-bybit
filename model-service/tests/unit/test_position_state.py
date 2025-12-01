"""
Unit tests for PositionState data model.
"""

import pytest
from datetime import datetime, timedelta
from src.models.position_state_tracker import PositionState


def test_position_state_creation():
    """Test creating a PositionState."""
    entry_time = datetime.utcnow()
    state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
        last_exit_signal_time=None,
    )

    assert state.asset == "BTCUSDT"
    assert state.entry_price == 50000.0
    assert state.entry_time == entry_time
    assert state.peak_price == 51000.0
    assert state.highest_unrealized_pnl == 2.0
    assert state.last_exit_signal_time is None


def test_update_peak_price():
    """Test updating peak price."""
    state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    # Update with higher price
    updated = state.update_peak_price(52000.0)
    assert updated is True
    assert state.peak_price == 52000.0

    # Try to update with lower price (should not update)
    updated = state.update_peak_price(51000.0)
    assert updated is False
    assert state.peak_price == 52000.0  # Still the higher value


def test_update_highest_pnl():
    """Test updating highest unrealized PnL."""
    state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    # Update with higher PnL
    updated = state.update_highest_pnl(3.5)
    assert updated is True
    assert state.highest_unrealized_pnl == 3.5

    # Try to update with lower PnL (should not update)
    updated = state.update_highest_pnl(2.5)
    assert updated is False
    assert state.highest_unrealized_pnl == 3.5  # Still the higher value


def test_get_time_held_minutes():
    """Test calculating time held in minutes."""
    entry_time = datetime.utcnow() - timedelta(hours=2)
    state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    time_held = state.get_time_held_minutes()
    assert time_held is not None
    assert 115 <= time_held <= 125  # Approximately 2 hours (120 minutes) with some tolerance


def test_to_dict():
    """Test converting PositionState to dictionary."""
    entry_time = datetime.utcnow()
    state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    result = state.to_dict()

    assert result["asset"] == "BTCUSDT"
    assert result["entry_price"] == 50000.0
    assert "entry_time" in result
    assert result["peak_price"] == 51000.0
    assert result["highest_unrealized_pnl"] == 2.0
    assert result["last_exit_signal_time"] is None


def test_from_dict():
    """Test creating PositionState from dictionary."""
    entry_time = datetime.utcnow()
    data = {
        "asset": "BTCUSDT",
        "entry_price": 50000.0,
        "entry_time": entry_time.isoformat() + "Z",
        "peak_price": 51000.0,
        "highest_unrealized_pnl": 2.0,
        "last_exit_signal_time": None,
    }

    state = PositionState.from_dict(data)

    assert state.asset == "BTCUSDT"
    assert state.entry_price == 50000.0
    assert state.peak_price == 51000.0
    assert state.highest_unrealized_pnl == 2.0

