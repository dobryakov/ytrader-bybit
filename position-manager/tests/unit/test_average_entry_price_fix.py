"""Unit tests for average_entry_price calculation and auto-fix functionality."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Position
from src.services.position_manager import PositionManager


@pytest.mark.asyncio
async def test_calculate_average_entry_price_from_orders_long_position():
    """Test calculation of average_entry_price from order history for long position."""
    manager = PositionManager()
    
    # Mock database pool and query result
    # asyncpg returns records that support dict-like access
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
    
    mock_row1 = MockRow({"side": "Buy", "total_qty": Decimal("1.0"), "total_value": Decimal("50000.0")})
    mock_row2 = MockRow({"side": "Sell", "total_qty": Decimal("0.5"), "total_value": Decimal("51000.0")})
    
    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[mock_row1, mock_row2])
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
        avg_price = await manager._calculate_average_entry_price_from_orders(
            "BTCUSDT", Decimal("1.0")  # Long position
        )
    
    # For long position, should use Buy orders: 50000.0 / 1.0 = 50000.0
    assert avg_price == Decimal("50000.0")


@pytest.mark.asyncio
async def test_calculate_average_entry_price_from_orders_short_position():
    """Test calculation of average_entry_price from order history for short position."""
    manager = PositionManager()
    
    # Mock database pool and query result
    class MockRow:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, key):
            return self._data[key]
    
    mock_row1 = MockRow({"side": "Buy", "total_qty": Decimal("0.5"), "total_value": Decimal("50000.0")})
    mock_row2 = MockRow({"side": "SELL", "total_qty": Decimal("2.0"), "total_value": Decimal("98000.0")})
    
    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[mock_row1, mock_row2])
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
        avg_price = await manager._calculate_average_entry_price_from_orders(
            "BTCUSDT", Decimal("-2.0")  # Short position
        )
    
    # For short position, should use Sell orders: 98000.0 / 2.0 = 49000.0
    assert avg_price == Decimal("49000.0")


@pytest.mark.asyncio
async def test_calculate_average_entry_price_from_orders_no_orders():
    """Test calculation returns None when no orders found."""
    manager = PositionManager()
    
    mock_pool = AsyncMock()
    mock_pool.fetch = AsyncMock(return_value=[])
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
        avg_price = await manager._calculate_average_entry_price_from_orders(
            "BTCUSDT", Decimal("1.0")
        )
    
    assert avg_price is None


@pytest.mark.asyncio
async def test_get_position_auto_fixes_missing_average_entry_price():
    """Test that get_position automatically fixes missing average_entry_price."""
    manager = PositionManager()
    
    # Mock database row with NULL average_entry_price but non-zero size
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("-0.542"),
        "average_entry_price": None,  # NULL in database
        "current_price": Decimal("50000.0"),
        "unrealized_pnl": Decimal("0.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 1,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=mock_row)
    mock_pool.execute = AsyncMock()
    
    # Mock calculation to return a price
    with patch.object(
        manager,
        "_calculate_average_entry_price_from_orders",
        return_value=Decimal("49000.0"),
    ):
        with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
            position = await manager.get_position("BTCUSDT", "one-way")
    
    assert position is not None
    assert position.size == Decimal("-0.542")
    assert position.average_entry_price == Decimal("49000.0")
    # Verify that database was updated
    mock_pool.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_position_no_fix_when_size_zero():
    """Test that get_position does not fix average_entry_price when size is zero."""
    manager = PositionManager()
    
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("0"),
        "average_entry_price": None,
        "current_price": Decimal("50000.0"),
        "unrealized_pnl": Decimal("0.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 1,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=mock_row)
    mock_pool.execute = AsyncMock()
    
    with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
        position = await manager.get_position("BTCUSDT", "one-way")
    
    assert position is not None
    assert position.size == Decimal("0")
    assert position.average_entry_price is None
    # Verify that database was not updated
    mock_pool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_update_position_from_websocket_calculates_missing_avg_price():
    """Test that update_position_from_websocket calculates avg_price when missing."""
    manager = PositionManager()
    
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # Position doesn't exist yet
    
    # Mock insert to return created position
    mock_insert_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("1.0"),
        "average_entry_price": Decimal("50000.0"),
        "current_price": Decimal("51000.0"),
        "unrealized_pnl": Decimal("1000.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 1,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    mock_pool.fetchrow.side_effect = [
        None,  # First call: position doesn't exist
        mock_insert_row,  # Second call: after insert
    ]
    
    # Mock calculation to return a price
    with patch.object(
        manager,
        "_calculate_average_entry_price_from_orders",
        return_value=Decimal("50000.0"),
    ):
        with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
            position = await manager.update_position_from_websocket(
                asset="BTCUSDT",
                mode="one-way",
                size_from_ws=Decimal("1.0"),
                avg_price=None,  # Missing avg_price
                mark_price=Decimal("51000.0"),
            )
    
    assert position is not None
    assert position.size == Decimal("1.0")
    # Verify that calculated price was used
    assert position.average_entry_price == Decimal("50000.0")


@pytest.mark.asyncio
async def test_update_position_from_websocket_uses_mark_price_fallback():
    """Test that update_position_from_websocket uses mark_price as fallback."""
    manager = PositionManager()
    
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)
    
    mock_insert_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("1.0"),
        "average_entry_price": Decimal("51000.0"),  # Should use mark_price
        "current_price": Decimal("51000.0"),
        "unrealized_pnl": Decimal("0.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 1,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    mock_pool.fetchrow.side_effect = [None, mock_insert_row]
    
    # Mock calculation to return None (no orders)
    with patch.object(
        manager,
        "_calculate_average_entry_price_from_orders",
        return_value=None,
    ):
        with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
            position = await manager.update_position_from_websocket(
                asset="BTCUSDT",
                mode="one-way",
                size_from_ws=Decimal("1.0"),
                avg_price=None,
                mark_price=Decimal("51000.0"),  # Should be used as fallback
            )
    
    assert position is not None
    assert position.size == Decimal("1.0")
    # Verify that mark_price was used as fallback
    assert position.average_entry_price == Decimal("51000.0")


@pytest.mark.asyncio
async def test_update_position_from_order_fill_fixes_missing_avg_price():
    """Test that update_position_from_order_fill fixes missing average_entry_price."""
    manager = PositionManager()
    
    # Mock existing position with NULL average_entry_price
    # Use model_construct to bypass Pydantic validation (simulating DB state before fix)
    existing_position = Position.model_construct(
        id="123e4567-e89b-12d3-a456-426614174000",
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("-0.542"),
        average_entry_price=None,  # Missing!
        current_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("0.0"),
        realized_pnl=Decimal("0.0"),
        long_size=None,
        short_size=None,
        version=1,
        last_updated=datetime.utcnow(),
        closed_at=None,
        created_at=datetime.utcnow(),
    )
    
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_pool = AsyncMock()
    
    # Mock update query result
    mock_update_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("-0.542"),
        "average_entry_price": Decimal("49000.0"),  # Fixed
        "current_price": Decimal("50000.0"),
        "unrealized_pnl": Decimal("0.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 2,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    # Mock get_position to return existing position (bypassing DB query)
    async def mock_get_position(asset: str, mode: str = "one-way"):
        return existing_position
    
    mock_pool.fetchrow = AsyncMock(return_value=mock_update_row)
    
    # Mock calculation to return a price
    with patch.object(manager, "get_position", side_effect=mock_get_position):
        with patch.object(
            manager,
            "_calculate_average_entry_price_from_orders",
            return_value=Decimal("49000.0"),
        ):
            with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
                position = await manager.update_position_from_order_fill(
                    asset="BTCUSDT",
                    size_delta=Decimal("0"),
                    execution_price=Decimal("50000.0"),
                    execution_fees=None,
                    mode="one-way",
                )
    
    assert position is not None
    # Verify that average_entry_price was fixed
    assert position.average_entry_price == Decimal("49000.0")


@pytest.mark.asyncio
async def test_update_position_from_order_fill_uses_execution_price_fallback():
    """Test that update_position_from_order_fill uses execution_price as fallback."""
    manager = PositionManager()
    
    # Mock existing position with NULL average_entry_price
    # Use model_construct to bypass Pydantic validation (simulating DB state before fix)
    existing_position = Position.model_construct(
        id="123e4567-e89b-12d3-a456-426614174000",
        asset="BTCUSDT",
        mode="one-way",
        size=Decimal("-0.542"),
        average_entry_price=None,  # Missing!
        current_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("0.0"),
        realized_pnl=Decimal("0.0"),
        long_size=None,
        short_size=None,
        version=1,
        last_updated=datetime.utcnow(),
        closed_at=None,
        created_at=datetime.utcnow(),
    )
    
    # asyncpg records support dict() conversion
    class MockRow(dict):
        def __init__(self, data):
            super().__init__(data)
            self._data = data
        def __getitem__(self, key):
            return self._data.get(key)
    
    now = datetime.utcnow()
    mock_pool = AsyncMock()
    
    mock_update_row = MockRow({
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "size": Decimal("-0.542"),
        "average_entry_price": Decimal("50000.0"),  # Should use execution_price
        "current_price": Decimal("50000.0"),
        "unrealized_pnl": Decimal("0.0"),
        "realized_pnl": Decimal("0.0"),
        "long_size": None,
        "short_size": None,
        "version": 2,
        "last_updated": now,
        "closed_at": None,
        "created_at": now,
    })
    
    mock_pool.fetchrow = AsyncMock(return_value=mock_update_row)
    
    with patch.object(manager, "get_position", return_value=existing_position):
        # Mock calculation to return None
        with patch.object(
            manager,
            "_calculate_average_entry_price_from_orders",
            return_value=None,
        ):
            with patch("src.services.position_manager.DatabaseConnection.get_pool", return_value=mock_pool):
                position = await manager.update_position_from_order_fill(
                    asset="BTCUSDT",
                    size_delta=Decimal("0"),
                    execution_price=Decimal("50000.0"),  # Should be used as fallback
                    execution_fees=None,
                    mode="one-way",
                )
    
    assert position is not None
    # Verify that execution_price was used as fallback
    assert position.average_entry_price == Decimal("50000.0")

