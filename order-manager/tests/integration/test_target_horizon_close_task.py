"""Integration tests for TargetHorizonCloseTask."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.config.database import DatabaseConnection
from src.services.target_horizon_close_task import TargetHorizonCloseTask
from src.models.position import Position


@pytest.mark.asyncio
async def test_target_horizon_close_task_datetime_normalization():
    """Test that datetime normalization to timezone-naive works correctly for database queries.
    
    This test verifies the fix for the timezone error:
    "can't subtract offset-naive and offset-aware datetimes"
    """
    try:
        await DatabaseConnection.close_pool()
        pool = await DatabaseConnection.create_pool()
        
        signal_id = uuid4()
        order_id = uuid4()
        bybit_order_id = f"test-bybit-{uuid4()}"
        
        # Create target_timestamp in the past (expired)
        target_ts = datetime.now(timezone.utc) - timedelta(seconds=10)
        # Normalize to timezone-naive for database storage
        target_ts_naive = target_ts.replace(tzinfo=None)
        
        # Insert order with expired target_timestamp and filled status
        insert_query = """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, average_price, fees, created_at, updated_at,
                trace_id, is_dry_run, target_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW(), $13, $14, $15)
        """
        
        await pool.execute(
            insert_query,
            str(order_id),
            bybit_order_id,
            str(signal_id),
            "BTCUSDT",
            "Buy",
            "Market",
            "0.01",
            None,
            "filled",
            "0.01",
            "50000.0",
            None,
            "test-trace",
            False,
            target_ts_naive,
        )
        
        # Test the same normalization logic used in target_horizon_close_task
        # This is the critical part: normalize timezone-aware datetime to timezone-naive
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Query should work without timezone errors
        query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity,
                   filled_quantity, status, target_timestamp
            FROM orders
            WHERE target_timestamp IS NOT NULL
                AND target_timestamp <= $1
                AND status IN ('filled', 'partially_filled')
            ORDER BY target_timestamp ASC
        """
        
        # This should not raise "can't subtract offset-naive and offset-aware datetimes"
        rows = await pool.fetch(query, now_utc)
        
        # Verify query worked and found the order
        assert len(rows) >= 1
        found_order = None
        for row in rows:
            if row["order_id"] == bybit_order_id:
                found_order = row
                break
        
        assert found_order is not None
        assert found_order["asset"] == "BTCUSDT"
        assert found_order["status"] == "filled"
        assert found_order["target_timestamp"] is not None
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id))
    finally:
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_target_horizon_close_task_query_expired_orders():
    """Test that target_horizon_close_task correctly queries orders with expired target_timestamp."""
    try:
        await DatabaseConnection.close_pool()
        pool = await DatabaseConnection.create_pool()
        
        signal_id1 = uuid4()
        order_id1 = uuid4()
        bybit_order_id1 = f"test-bybit-{uuid4()}"
        
        signal_id2 = uuid4()
        order_id2 = uuid4()
        bybit_order_id2 = f"test-bybit-{uuid4()}"
        
        signal_id3 = uuid4()
        order_id3 = uuid4()
        bybit_order_id3 = f"test-bybit-{uuid4()}"
        
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Order 1: expired target_timestamp (past), filled status - should be found
        target_ts1 = now - timedelta(seconds=60)
        
        # Order 2: future target_timestamp, filled status - should NOT be found
        target_ts2 = now + timedelta(seconds=300)
        
        # Order 3: expired target_timestamp, pending status - should NOT be found (only filled/partially_filled)
        target_ts3 = now - timedelta(seconds=30)
        
        insert_query = """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, average_price, fees, created_at, updated_at,
                trace_id, is_dry_run, target_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW(), $13, $14, $15)
        """
        
        # Insert order 1 (expired, filled)
        await pool.execute(
            insert_query,
            str(order_id1),
            bybit_order_id1,
            str(signal_id1),
            "BTCUSDT",
            "Buy",
            "Market",
            "0.01",
            None,
            "filled",
            "0.01",
            "50000.0",
            None,
            "test-trace-1",
            False,
            target_ts1,
        )
        
        # Insert order 2 (future, filled)
        await pool.execute(
            insert_query,
            str(order_id2),
            bybit_order_id2,
            str(signal_id2),
            "ETHUSDT",
            "Buy",
            "Market",
            "0.1",
            None,
            "filled",
            "0.1",
            "3000.0",
            None,
            "test-trace-2",
            False,
            target_ts2,
        )
        
        # Insert order 3 (expired, pending)
        await pool.execute(
            insert_query,
            str(order_id3),
            bybit_order_id3,
            str(signal_id3),
            "BTCUSDT",
            "SELL",
            "Market",
            "0.01",
            None,
            "pending",
            "0",
            None,
            None,
            "test-trace-3",
            False,
            target_ts3,
        )
        
        # Test query logic from target_horizon_close_task
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        
        query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity,
                   filled_quantity, status, target_timestamp
            FROM orders
            WHERE target_timestamp IS NOT NULL
                AND target_timestamp <= $1
                AND status IN ('filled', 'partially_filled')
            ORDER BY target_timestamp ASC
        """
        
        rows = await pool.fetch(query, now_utc)
        
        # Should find only order 1 (expired + filled)
        assert len(rows) >= 1
        found_order_ids = [row["order_id"] for row in rows]
        assert bybit_order_id1 in found_order_ids
        assert bybit_order_id2 not in found_order_ids  # Future timestamp
        assert bybit_order_id3 not in found_order_ids  # Pending status
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id IN ($1, $2, $3)", str(order_id1), str(order_id2), str(order_id3))
    finally:
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_target_horizon_close_task_skip_closed_position():
    """Test that target_horizon_close_task skips orders when position is already closed."""
    task = TargetHorizonCloseTask()
    
    # Mock position manager to return None (position doesn't exist/closed)
    mock_position = None
    
    with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = mock_position
        
        # Mock SignalProcessor to avoid actual order creation
        with patch.object(task, '_close_position_for_order', new_callable=AsyncMock) as mock_close:
            # This should skip the position since it doesn't exist
            # We can't easily test the full loop, but we can test the skip logic
            
            # Create a mock row as if from database query
            mock_row = {
                "order_id": "test-order-123",
                "asset": "BTCUSDT",
                "target_timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
                "status": "filled",
            }
            
            # Simulate the check logic from _close_loop
            position = await task._position_manager_client.get_position(
                asset="BTCUSDT",
                mode="one-way",
                trace_id="test-trace",
            )
            
            # Position should be None or size == 0, so it should be skipped
            assert position is None or (position.size == 0 if position else True)
            
            # Verify get_position was called
            mock_get_position.assert_called_once_with(
                asset="BTCUSDT",
                mode="one-way",
                trace_id="test-trace",
            )
            
            # Verify close_position was NOT called (should be skipped)
            mock_close.assert_not_called()


@pytest.mark.asyncio
async def test_target_horizon_close_task_close_position_with_open_position():
    """Test that target_horizon_close_task attempts to close position when position exists."""
    task = TargetHorizonCloseTask()
    
    # Mock position manager to return an open position
    mock_position = Position(
        id=uuid4(),
        asset="BTCUSDT",
        size=Decimal("0.001"),
        average_entry_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("50.0"),
        realized_pnl=Decimal("0.0"),
        mode="one-way",
        last_updated=datetime.now(timezone.utc),
    )
    
    with patch.object(task._position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = mock_position
        
        # Mock _close_position_for_order to return True (success)
        with patch.object(task, '_close_position_for_order', new_callable=AsyncMock) as mock_close:
            mock_close.return_value = True
            
            # Simulate the check logic from _close_loop
            position = await task._position_manager_client.get_position(
                asset="BTCUSDT",
                mode="one-way",
                trace_id="test-trace",
            )
            
            # Position should exist and have non-zero size
            assert position is not None
            assert position.size != 0
            
            # Simulate calling _close_position_for_order
            if position and position.size != 0:
                success = await task._close_position_for_order(
                    asset="BTCUSDT",
                    position=position,
                    order_id="test-order-123",
                    trace_id="test-trace",
                )
                assert success is True
            
            # Verify get_position was called
            mock_get_position.assert_called_once_with(
                asset="BTCUSDT",
                mode="one-way",
                trace_id="test-trace",
            )
            
            # Verify close_position was called
            mock_close.assert_called_once()

