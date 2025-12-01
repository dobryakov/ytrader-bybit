"""Integration tests for database operations (orders, signal-order relationships)."""

import pytest
from decimal import Decimal
from datetime import datetime
from uuid import uuid4, UUID

from src.config.database import DatabaseConnection
from src.models.order import Order
from src.models.signal_order_rel import SignalOrderRelationship


@pytest.mark.asyncio
async def test_create_and_read_order():
    """Test creating and reading an order from database."""
    # Create test order data
    signal_id = uuid4()
    order_id = uuid4()
    bybit_order_id = f"test-bybit-{uuid4()}"
    
    try:
        # Close any existing pool to ensure clean state
        await DatabaseConnection.close_pool()
        # Create fresh pool for this test
        pool = await DatabaseConnection.create_pool()
        
        # Insert order
        insert_query = """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, average_price, fees, created_at, updated_at,
                trace_id, is_dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW(), $13, $14)
        """
        
        await pool.execute(
            insert_query,
            str(order_id),
            bybit_order_id,
            str(signal_id),
            "BTCUSDT",
            "Buy",
            "Limit",
            "0.01",
            "50000.0",
            "pending",
            "0",
            None,
            None,
            "test-trace-001",
            False,
        )
        
        # Read order back
        select_query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                   status, filled_quantity, average_price, fees, created_at, updated_at,
                   executed_at, trace_id, is_dry_run, rejection_reason
            FROM orders
            WHERE id = $1
        """
        
        row = await pool.fetchrow(select_query, str(order_id))
        
        assert row is not None
        assert row["order_id"] == bybit_order_id
        # PostgreSQL returns UUID as UUID object, compare appropriately
        assert str(row["signal_id"]) == str(signal_id)
        assert row["asset"] == "BTCUSDT"
        assert row["side"] == "Buy"
        assert row["order_type"] == "Limit"
        assert Decimal(str(row["quantity"])) == Decimal("0.01")
        assert Decimal(str(row["price"])) == Decimal("50000.0")
        assert row["status"] == "pending"
        
        # Convert to Order model
        order = Order.from_dict(dict(row))
        assert order.id == order_id
        assert order.order_id == bybit_order_id
        assert order.asset == "BTCUSDT"
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id))
    finally:
        # Close pool after test
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_create_signal_order_relationship():
    """Test creating and reading signal-order relationship."""
    signal_id = uuid4()
    order_id = uuid4()
    rel_id = uuid4()
    
    try:
        # Close any existing pool to ensure clean state
        await DatabaseConnection.close_pool()
        # Create fresh pool for this test
        pool = await DatabaseConnection.create_pool()
        
        # First create an order
        order_insert = """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, created_at, updated_at, trace_id, is_dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12)
        """
        await pool.execute(
            order_insert,
            str(order_id),
            f"bybit-{uuid4()}",
            str(signal_id),
            "BTCUSDT",
            "Buy",
            "Limit",
            "0.01",
            "50000.0",
            "pending",
            "0",
            "test-trace",
            False,
        )
        
        # Create relationship
        rel_insert = """
            INSERT INTO signal_order_relationships (
                id, signal_id, order_id, relationship_type, execution_sequence,
                allocation_amount, allocation_quantity, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        """
        
        await pool.execute(
            rel_insert,
            str(rel_id),
            str(signal_id),
            str(order_id),
            "one_to_one",
            1,
            "500.0",
            "0.01",
        )
        
        # Read relationship back
        rel_select = """
            SELECT id, signal_id, order_id, relationship_type, execution_sequence,
                   allocation_amount, allocation_quantity, created_at
            FROM signal_order_relationships
            WHERE id = $1
        """
        
        row = await pool.fetchrow(rel_select, str(rel_id))
        
        assert row is not None
        # PostgreSQL returns UUID as UUID object, compare appropriately
        assert str(row["signal_id"]) == str(signal_id)
        assert str(row["order_id"]) == str(order_id)
        assert row["relationship_type"] == "one_to_one"
        
        # Convert to model
        rel = SignalOrderRelationship.from_dict(dict(row))
        assert rel.id == rel_id
        assert rel.signal_id == signal_id
        assert rel.order_id == order_id
        
        # Cleanup
        await pool.execute("DELETE FROM signal_order_relationships WHERE id = $1", str(rel_id))
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id))
    finally:
        # Close pool after test
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_query_orders_by_asset():
    """Test querying orders filtered by asset."""
    try:
        # Close any existing pool to ensure clean state
        await DatabaseConnection.close_pool()
        # Create fresh pool for this test
        pool = await DatabaseConnection.create_pool()
        
        # Create test orders
        signal_id_1 = uuid4()
        order_id_1 = uuid4()
        signal_id_2 = uuid4()
        order_id_2 = uuid4()
        
        # Insert BTCUSDT order
        await pool.execute(
            """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, created_at, updated_at, trace_id, is_dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12)
            """,
            str(order_id_1),
            f"bybit-{uuid4()}",
            str(signal_id_1),
            "BTCUSDT",
            "Buy",
            "Limit",
            "0.01",
            "50000.0",
            "pending",
            "0",
            "test-trace",
            False,
        )
        
        # Insert ETHUSDT order
        await pool.execute(
            """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, created_at, updated_at, trace_id, is_dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12)
            """,
            str(order_id_2),
            f"bybit-{uuid4()}",
            str(signal_id_2),
            "ETHUSDT",
            "Buy",
            "Limit",
            "0.1",
            "3000.0",
            "pending",
            "0",
            "test-trace",
            False,
        )
        
        # Query BTCUSDT orders only
        query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                   status, filled_quantity, average_price, fees, created_at, updated_at,
                   executed_at, trace_id, is_dry_run, rejection_reason
            FROM orders
            WHERE asset = $1
            ORDER BY created_at DESC
        """
        
        rows = await pool.fetch(query, "BTCUSDT")
        
        # Should only get BTCUSDT order
        assert len(rows) >= 1
        btc_orders = [row for row in rows if row["asset"] == "BTCUSDT"]
        assert len(btc_orders) >= 1
        assert btc_orders[0]["asset"] == "BTCUSDT"
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id_1))
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id_2))
    finally:
        # Close pool after test
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_update_order_status():
    """Test updating order status in database."""
    try:
        # Close any existing pool to ensure clean state
        await DatabaseConnection.close_pool()
        # Create fresh pool for this test
        pool = await DatabaseConnection.create_pool()
        
        signal_id = uuid4()
        order_id = uuid4()
        
        # Create order
        await pool.execute(
            """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, created_at, updated_at, trace_id, is_dry_run
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW(), $11, $12)
            """,
            str(order_id),
            f"bybit-{uuid4()}",
            str(signal_id),
            "BTCUSDT",
            "Buy",
            "Limit",
            "0.01",
            "50000.0",
            "pending",
            "0",
            "test-trace",
            False,
        )
        
        # Update order status
        update_query = """
            UPDATE orders
            SET status = $1, filled_quantity = $2, average_price = $3, updated_at = NOW()
            WHERE id = $4
        """
        
        await pool.execute(
            update_query,
            "filled",
            "0.01",
            "50000.0",
            str(order_id),
        )
        
        # Verify update
        select_query = """
            SELECT status, filled_quantity, average_price
            FROM orders
            WHERE id = $1
        """
        
        row = await pool.fetchrow(select_query, str(order_id))
        
        assert row is not None
        assert row["status"] == "filled"
        assert Decimal(str(row["filled_quantity"])) == Decimal("0.01")
        assert Decimal(str(row["average_price"])) == Decimal("50000.0")
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id))
    finally:
        # Close pool after test
        await DatabaseConnection.close_pool()
