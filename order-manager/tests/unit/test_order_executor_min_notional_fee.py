"""Unit tests for fee-based minimal notional check in OrderExecutor."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from src.services.order_executor import OrderExecutor
from src.services.fee_rate_manager import FeeRate
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.config.settings import settings
from src.exceptions import OrderExecutionError


@pytest.fixture
def order_executor():
    executor = OrderExecutor()
    # Replace fee_rate_manager with mock
    executor.fee_rate_manager = MagicMock()
    executor.fee_rate_manager.get_fee_rate = AsyncMock()
    # Also avoid hitting DB when saving rejected order
    executor._save_rejected_order = AsyncMock(return_value=None)
    return executor


@pytest.fixture
def sample_signal():
    return TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="BTCUSDT",
        amount=Decimal("10.0"),
        confidence=Decimal("0.8"),
        timestamp=datetime.now(timezone.utc),
        strategy_id="test-strategy",
        model_version="v1",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("100.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id=None,
    )


@pytest.mark.asyncio
async def test_min_notional_fee_check_rejects_when_notional_leq_fee(order_executor, sample_signal):
    """Order should be rejected when notional is not greater than expected fee."""
    # notional = 1 * 100 = 100
    quantity = Decimal("1.0")
    price = Decimal("100.0")

    fee_rate = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("1.0"),
        taker_fee_rate=Decimal("1.0"),  # 100% fee so notional <= expected_fee
        last_synced_at=datetime.now(timezone.utc),
    )
    order_executor.fee_rate_manager.get_fee_rate.return_value = fee_rate

    with patch.object(settings, "order_manager_enable_min_notional_fee_check", True):
        with patch.object(settings, "order_manager_max_fallback_fee_rate", 0.001):
            with pytest.raises(OrderExecutionError):
                await order_executor._validate_order_notional_vs_fee(
                    signal=sample_signal,
                    order_type="Market",
                    quantity=quantity,
                    price=price,
                    trace_id="test-trace",
                )


@pytest.mark.asyncio
async def test_min_notional_fee_check_allows_when_notional_much_bigger(order_executor, sample_signal):
    """Order should pass when notional is much larger than expected fee."""
    quantity = Decimal("1.0")
    price = Decimal("100.0")

    fee_rate = FeeRate(
        symbol="BTCUSDT",
        market_type="linear",
        maker_fee_rate=Decimal("0.0001"),
        taker_fee_rate=Decimal("0.0001"),  # 0.01% fee
        last_synced_at=datetime.now(timezone.utc),
    )
    order_executor.fee_rate_manager.get_fee_rate.return_value = fee_rate

    with patch.object(settings, "order_manager_enable_min_notional_fee_check", True):
        # Should not raise
        await order_executor._validate_order_notional_vs_fee(
            signal=sample_signal,
            order_type="Market",
            quantity=quantity,
            price=price,
            trace_id="test-trace",
        )


@pytest.mark.asyncio
async def test_min_notional_fee_check_uses_fallback_when_no_fee_data(order_executor, sample_signal):
    """When fee data is unavailable, fallback fee rate from settings is used."""
    quantity = Decimal("1.0")
    price = Decimal("100.0")

    order_executor.fee_rate_manager.get_fee_rate.return_value = None

    # Set very high fallback fee to force rejection (100% of notional)
    with patch.object(settings, "order_manager_enable_min_notional_fee_check", True):
        with patch.object(settings, "order_manager_max_fallback_fee_rate", 1.0):  # 100%
            with pytest.raises(OrderExecutionError):
                await order_executor._validate_order_notional_vs_fee(
                    signal=sample_signal,
                    order_type="Market",
                    quantity=quantity,
                    price=price,
                    trace_id="test-trace",
                )



