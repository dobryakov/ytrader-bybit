"""Integration test for order creation with target_timestamp."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.config.database import DatabaseConnection
from src.models.order import Order
from src.models.trading_signal import TradingSignal, MarketDataSnapshot


@pytest.mark.asyncio
async def test_order_creation_with_target_timestamp():
    """Test that order creation saves target_timestamp from signal metadata."""
    try:
        await DatabaseConnection.close_pool()
        pool = await DatabaseConnection.create_pool()
        
        signal_id = uuid4()
        order_id = uuid4()
        bybit_order_id = f"test-bybit-{uuid4()}"
        target_ts = datetime.now(timezone.utc) + timedelta(seconds=180)
        # Normalize to timezone-naive UTC for asyncpg
        target_ts_naive = target_ts.astimezone(timezone.utc).replace(tzinfo=None)
        
        # Insert order with target_timestamp
        insert_query = """
            INSERT INTO orders (
                id, order_id, signal_id, asset, side, order_type, quantity, price,
                status, filled_quantity, average_price, fees, created_at, updated_at,
                trace_id, is_dry_run, target_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW(), $13, $14, $15::timestamptz)
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
        
        # Read order back
        select_query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                   status, filled_quantity, average_price, fees, created_at, updated_at,
                   executed_at, trace_id, is_dry_run, rejection_reason, target_timestamp
            FROM orders
            WHERE id = $1
        """
        
        row = await pool.fetchrow(select_query, str(order_id))
        
        assert row is not None
        assert row["target_timestamp"] is not None
        
        # Convert to Order model
        order = Order.from_dict(dict(row))
        assert order.target_timestamp is not None
        # Compare timestamps (normalize both to UTC for comparison)
        order_ts_utc = order.target_timestamp.astimezone(timezone.utc) if order.target_timestamp.tzinfo else order.target_timestamp.replace(tzinfo=timezone.utc)
        target_ts_utc = target_ts.astimezone(timezone.utc) if target_ts.tzinfo else target_ts.replace(tzinfo=timezone.utc)
        assert abs((order_ts_utc - target_ts_utc).total_seconds()) < 1
        
        # Cleanup
        await pool.execute("DELETE FROM orders WHERE id = $1", str(order_id))
    finally:
        await DatabaseConnection.close_pool()


@pytest.mark.asyncio
async def test_trading_signal_get_target_timestamp():
    """Test TradingSignal.get_target_timestamp() method."""
    from datetime import datetime, timedelta, timezone
    
    # Test with direct target_timestamp
    signal = TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="BTCUSDT",
        amount=Decimal("1000.0"),
        confidence=Decimal("0.75"),
        timestamp=datetime.now(timezone.utc),
        strategy_id="test-strategy",
        model_version="v1",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("50000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata={
            "target_timestamp": (datetime.now(timezone.utc) + timedelta(seconds=180)).isoformat(),
        },
        trace_id=None,
    )
    
    target_ts = signal.get_target_timestamp()
    assert target_ts is not None
    assert isinstance(target_ts, datetime)
    
    # Test with prediction_horizon_seconds
    signal2 = TradingSignal(
        signal_id=uuid4(),
        signal_type="sell",
        asset="ETHUSDT",
        amount=Decimal("500.0"),
        confidence=Decimal("0.80"),
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        strategy_id="test-strategy",
        model_version="v1",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("3000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("500000.0"),
            volatility=Decimal("0.015"),
        ),
        metadata={
            "prediction_horizon_seconds": 300,  # 5 minutes
        },
        trace_id=None,
    )
    
    target_ts2 = signal2.get_target_timestamp()
    assert target_ts2 is not None
    expected_ts = signal2.timestamp + timedelta(seconds=300)
    assert target_ts2 == expected_ts

