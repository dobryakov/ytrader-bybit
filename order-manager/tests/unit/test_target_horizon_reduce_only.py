"""Unit tests for TargetHorizonCloseTask with reduce-only orders and balance checks."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.target_horizon_close_task import TargetHorizonCloseTask
from src.models.position import Position
from src.models.order import Order
from src.exceptions import OrderExecutionError


@pytest.fixture
def mock_position_long():
    """Create a mock long position."""
    return Position(
        id=uuid4(),
        asset="BTCUSDT",
        size=Decimal("0.01"),
        average_entry_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("50.0"),
        realized_pnl=Decimal("0.0"),
        mode="one-way",
        last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_position_short():
    """Create a mock short position."""
    return Position(
        id=uuid4(),
        asset="BTCUSDT",
        size=Decimal("-0.01"),
        average_entry_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("-50.0"),
        realized_pnl=Decimal("0.0"),
        mode="one-way",
        last_updated=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_close_position_skips_balance_check_for_reduce_only(mock_position_long):
    """Test that balance check is skipped for reduce-only orders."""
    task = TargetHorizonCloseTask()
    
    # Mock database connection to avoid real DB calls
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # No existing orders
    
    # Mock order executor (imported inside the method)
    with patch('src.services.order_executor.OrderExecutor') as mock_order_executor_class:
        mock_order_executor = MagicMock()
        mock_order_executor_class.return_value = mock_order_executor
        
        # Mock instrument info manager (imported inside the method)
        with patch('src.services.instrument_info_manager.InstrumentInfoManager') as mock_instrument_class:
            mock_instrument = MagicMock()
            mock_instrument_class.return_value = mock_instrument
            
            # Mock instrument info
            mock_instrument_info = MagicMock()
            mock_instrument_info.last_price = Decimal("50000.0")
            mock_instrument.get_instrument = AsyncMock(return_value=mock_instrument_info)
            
            # Mock position manager to return same position
            with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_long
                
                # Mock database connection
                with patch('src.services.target_horizon_close_task.DatabaseConnection.get_pool', new_callable=AsyncMock) as mock_get_pool:
                    mock_get_pool.return_value = mock_pool
                    
                    # Mock order creation to return success
                    mock_order = Order(
                        id=uuid4(),
                        order_id="test-bybit-123",
                        signal_id=uuid4(),
                        asset="BTCUSDT",
                        side="SELL",
                        order_type="Market",
                        quantity=Decimal("0.01"),
                        price=None,
                        status="pending",
                        filled_quantity=Decimal("0"),
                        average_price=None,
                        fees=None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        executed_at=None,
                        trace_id="test-trace",
                        is_dry_run=False,
                        rejection_reason=None,
                        target_timestamp=None,
                    )
                    mock_order_executor.create_order = AsyncMock(return_value=mock_order)
                    
                    # Call _close_position_for_order
                    result = await task._close_position_for_order(
                        asset="BTCUSDT",
                        position=mock_position_long,
                        order_id="test-order-123",
                        trace_id="test-trace",
                    )
                    
                    # Should succeed
                    assert result is True
                    
                    # Verify order was created with force_reduce_only=True
                    mock_order_executor.create_order.assert_called_once()
                    call_args = mock_order_executor.create_order.call_args
                    assert call_args.kwargs['force_reduce_only'] is True


@pytest.mark.asyncio
async def test_close_position_verifies_position_before_creating_order(mock_position_long):
    """Test that position is verified before creating close order."""
    task = TargetHorizonCloseTask()
    
    # Mock database connection to avoid real DB calls
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # No existing orders
    
    # Mock order executor (imported inside the method)
    with patch('src.services.order_executor.OrderExecutor') as mock_order_executor_class:
        mock_order_executor = MagicMock()
        mock_order_executor_class.return_value = mock_order_executor
        
        # Mock instrument info manager (imported inside the method)
        with patch('src.services.instrument_info_manager.InstrumentInfoManager') as mock_instrument_class:
            mock_instrument = MagicMock()
            mock_instrument_class.return_value = mock_instrument
            
            # Mock instrument info
            mock_instrument_info = MagicMock()
            mock_instrument_info.last_price = Decimal("50000.0")
            mock_instrument.get_instrument = AsyncMock(return_value=mock_instrument_info)
            
            # Mock position manager to return position that changed size
            updated_position = Position(
                id=mock_position_long.id,
                asset="BTCUSDT",
                size=Decimal("0.005"),  # Size changed
                average_entry_price=Decimal("50000.0"),
                unrealized_pnl=Decimal("25.0"),
                realized_pnl=Decimal("0.0"),
                mode="one-way",
                last_updated=datetime.now(timezone.utc),
            )
            
            with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = updated_position
                
                # Mock database connection
                with patch('src.services.target_horizon_close_task.DatabaseConnection.get_pool', new_callable=AsyncMock) as mock_get_pool:
                    mock_get_pool.return_value = mock_pool
                    
                    # Mock order creation
                    mock_order = Order(
                        id=uuid4(),
                        order_id="test-bybit-123",
                        signal_id=uuid4(),
                        asset="BTCUSDT",
                        side="SELL",
                        order_type="Market",
                        quantity=Decimal("0.005"),  # Updated quantity
                        price=None,
                        status="pending",
                        filled_quantity=Decimal("0"),
                        average_price=None,
                        fees=None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        executed_at=None,
                        trace_id="test-trace",
                        is_dry_run=False,
                        rejection_reason=None,
                        target_timestamp=None,
                    )
                    mock_order_executor.create_order = AsyncMock(return_value=mock_order)
                    
                    # Call _close_position_for_order
                    result = await task._close_position_for_order(
                        asset="BTCUSDT",
                        position=mock_position_long,
                        order_id="test-order-123",
                        trace_id="test-trace",
                    )
                    
                    # Should succeed
                    assert result is True
                    
                    # Verify position was checked before creating order
                    mock_get_position.assert_called_once_with(
                        asset="BTCUSDT",
                        mode="one-way",
                        trace_id="test-trace",
                    )
                    
                    # Verify order was created with updated quantity
                    mock_order_executor.create_order.assert_called_once()
                    call_args = mock_order_executor.create_order.call_args
                    assert call_args.kwargs['quantity'] == Decimal("0.005")  # Updated quantity


@pytest.mark.asyncio
async def test_close_position_handles_position_already_closed(mock_position_long):
    """Test that position already closed is handled as success."""
    task = TargetHorizonCloseTask()
    
    # Mock database connection to avoid real DB calls
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # No existing orders
    
    # Mock order executor (imported inside the method)
    with patch('src.services.order_executor.OrderExecutor') as mock_order_executor_class:
        mock_order_executor = MagicMock()
        mock_order_executor_class.return_value = mock_order_executor
        
        # Mock instrument info manager (imported inside the method)
        with patch('src.services.instrument_info_manager.InstrumentInfoManager') as mock_instrument_class:
            mock_instrument = MagicMock()
            mock_instrument_class.return_value = mock_instrument
            
            # Mock position manager to return None (position closed)
            with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = None  # Position already closed
                
                # Mock database connection
                with patch('src.services.target_horizon_close_task.DatabaseConnection.get_pool', new_callable=AsyncMock) as mock_get_pool:
                    mock_get_pool.return_value = mock_pool
                    
                    # Call _close_position_for_order
                    result = await task._close_position_for_order(
                        asset="BTCUSDT",
                        position=mock_position_long,
                        order_id="test-order-123",
                        trace_id="test-trace",
                    )
                    
                    # Should return True (success - position already closed)
                    assert result is True
                    
                    # Verify position was checked
                    mock_get_position.assert_called_once_with(
                        asset="BTCUSDT",
                        mode="one-way",
                        trace_id="test-trace",
                    )
                    
                    # Verify order was NOT created (position already closed)
                    mock_order_executor.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_close_position_handles_order_returns_none_for_closed_position(mock_position_long):
    """Test that when create_order returns None (position closed during creation), it's handled as success."""
    task = TargetHorizonCloseTask()
    
    # Mock database connection to avoid real DB calls
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # No existing orders
    
    # Mock order executor (imported inside the method)
    with patch('src.services.order_executor.OrderExecutor') as mock_order_executor_class:
        mock_order_executor = MagicMock()
        mock_order_executor_class.return_value = mock_order_executor
        
        # Mock instrument info manager (imported inside the method)
        with patch('src.services.instrument_info_manager.InstrumentInfoManager') as mock_instrument_class:
            mock_instrument = MagicMock()
            mock_instrument_class.return_value = mock_instrument
            
            # Mock instrument info
            mock_instrument_info = MagicMock()
            mock_instrument_info.last_price = Decimal("50000.0")
            mock_instrument.get_instrument = AsyncMock(return_value=mock_instrument_info)
            
            # Mock position manager to return position
            with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_long
                
                # Mock database connection
                with patch('src.services.target_horizon_close_task.DatabaseConnection.get_pool', new_callable=AsyncMock) as mock_get_pool:
                    mock_get_pool.return_value = mock_pool
                    
                    # Mock order creation to return None (position closed during creation)
                    mock_order_executor.create_order = AsyncMock(return_value=None)
                    
                    # Call _close_position_for_order
                    result = await task._close_position_for_order(
                        asset="BTCUSDT",
                        position=mock_position_long,
                        order_id="test-order-123",
                        trace_id="test-trace",
                    )
                    
                    # Should return True (success - position closed during order creation)
                    assert result is True
                    
                    # Verify order creation was attempted
                    mock_order_executor.create_order.assert_called_once()
                    call_args = mock_order_executor.create_order.call_args
                    assert call_args.kwargs['force_reduce_only'] is True


@pytest.mark.asyncio
async def test_close_position_handles_position_check_failure(mock_position_long):
    """Test that position check failure is handled gracefully."""
    task = TargetHorizonCloseTask()
    
    # Mock database connection to avoid real DB calls
    mock_pool = AsyncMock()
    mock_pool.fetchrow = AsyncMock(return_value=None)  # No existing orders
    
    # Mock order executor (imported inside the method)
    with patch('src.services.order_executor.OrderExecutor') as mock_order_executor_class:
        mock_order_executor = MagicMock()
        mock_order_executor_class.return_value = mock_order_executor
        
        # Mock instrument info manager (imported inside the method)
        with patch('src.services.instrument_info_manager.InstrumentInfoManager') as mock_instrument_class:
            mock_instrument = MagicMock()
            mock_instrument_class.return_value = mock_instrument
            
            # Mock instrument info
            mock_instrument_info = MagicMock()
            mock_instrument_info.last_price = Decimal("50000.0")
            mock_instrument.get_instrument = AsyncMock(return_value=mock_instrument_info)
            
            # Mock position manager to raise exception
            with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.side_effect = Exception("Network error")
                
                # Mock database connection
                with patch('src.services.target_horizon_close_task.DatabaseConnection.get_pool', new_callable=AsyncMock) as mock_get_pool:
                    mock_get_pool.return_value = mock_pool
                    
                    # Mock order creation to return success
                    mock_order = Order(
                        id=uuid4(),
                        order_id="test-bybit-123",
                        signal_id=uuid4(),
                        asset="BTCUSDT",
                        side="SELL",
                        order_type="Market",
                        quantity=Decimal("0.01"),
                        price=None,
                        status="pending",
                        filled_quantity=Decimal("0"),
                        average_price=None,
                        fees=None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                        executed_at=None,
                        trace_id="test-trace",
                        is_dry_run=False,
                        rejection_reason=None,
                        target_timestamp=None,
                    )
                    mock_order_executor.create_order = AsyncMock(return_value=mock_order)
                    
                    # Call _close_position_for_order
                    result = await task._close_position_for_order(
                        asset="BTCUSDT",
                        position=mock_position_long,
                        order_id="test-order-123",
                        trace_id="test-trace",
                    )
                    
                    # Should continue and create order anyway
                    assert result is True
                    
                    # Verify order was created despite position check failure
                    mock_order_executor.create_order.assert_called_once()

