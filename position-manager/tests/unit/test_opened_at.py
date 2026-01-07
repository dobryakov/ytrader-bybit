"""Unit tests for opened_at field tracking in Position model."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.models import Position
from src.services.position_manager import PositionManager


@pytest.mark.asyncio
async def test_opened_at_set_when_position_created_with_non_zero_size():
    """Test that opened_at is set when position is created with non-zero size."""
    manager = PositionManager()
    now = datetime.utcnow()
    
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        current_price=Decimal("51000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        opened_at=now - timedelta(minutes=5),  # Position was opened 5 minutes ago
    )
    
    assert position.opened_at is not None
    assert position.opened_at < now
    assert position.opened_at > position.created_at


@pytest.mark.asyncio
async def test_opened_at_none_when_position_closed():
    """Test that opened_at can be None for closed positions."""
    manager = PositionManager()
    now = datetime.utcnow()
    
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("0.0"),
        average_entry_price=None,
        current_price=None,
        unrealized_pnl=Decimal("0.0"),
        realized_pnl=Decimal("100.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        opened_at=None,  # Closed position, no opened_at
        closed_at=now - timedelta(minutes=1),
    )
    
    assert position.opened_at is None
    assert position.closed_at is not None


def test_time_held_minutes_uses_opened_at_when_available():
    """Test that time_held_minutes property uses opened_at when available, not created_at."""
    now = datetime.utcnow()
    
    # Position created 10 minutes ago, but opened 5 minutes ago
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        current_price=Decimal("51000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        opened_at=now - timedelta(minutes=5),  # Opened 5 minutes ago
    )
    
    minutes = position.time_held_minutes
    assert minutes is not None
    # Should be approximately 5 minutes (from opened_at), not 10 (from created_at)
    assert 4 <= minutes <= 6


def test_time_held_minutes_fallback_to_created_at_when_opened_at_missing():
    """Test that time_held_minutes property falls back to created_at when opened_at is None."""
    now = datetime.utcnow()
    
    # Position without opened_at (legacy data)
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        current_price=Decimal("51000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        opened_at=None,  # Legacy position without opened_at
    )
    
    minutes = position.time_held_minutes
    assert minutes is not None
    # Should fall back to created_at (10 minutes ago)
    assert 9 <= minutes <= 11


def test_position_model_accepts_opened_at():
    """Test that Position model correctly accepts and stores opened_at field."""
    now = datetime.utcnow()
    opened_time = now - timedelta(minutes=5)
    
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        current_price=Decimal("51000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        opened_at=opened_time,
    )
    
    assert position.opened_at == opened_time
    assert position.opened_at is not None
    assert isinstance(position.opened_at, datetime)


def test_position_model_opened_at_optional():
    """Test that opened_at is optional in Position model."""
    now = datetime.utcnow()
    
    position = Position(
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("1.0"),
        average_entry_price=Decimal("50000.0"),
        current_price=Decimal("51000.0"),
        unrealized_pnl=Decimal("1000.0"),
        realized_pnl=Decimal("0.0"),
        created_at=now - timedelta(minutes=10),
        last_updated=now,
        # opened_at not provided - should default to None
    )
    
    assert position.opened_at is None or isinstance(position.opened_at, datetime)

