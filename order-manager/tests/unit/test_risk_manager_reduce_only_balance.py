"""Unit tests for RiskManager balance check with reduce-only orders."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.risk_manager import RiskManager
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.models.position import Position


@pytest.fixture
def buy_signal():
    """Create a buy signal for testing."""
    return TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="BTCUSDT",
        amount=Decimal("1000.0"),
        confidence=Decimal("0.85"),
        timestamp=datetime.now(timezone.utc),
        strategy_id="test-strategy",
        model_version="v1.0",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("50000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id="test-trace-001",
    )


@pytest.fixture
def sell_signal():
    """Create a sell signal for testing."""
    return TradingSignal(
        signal_id=uuid4(),
        signal_type="sell",
        asset="BTCUSDT",
        amount=Decimal("1000.0"),
        confidence=Decimal("0.85"),
        timestamp=datetime.now(timezone.utc),
        strategy_id="test-strategy",
        model_version="v1.0",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("50000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("1000000.0"),
            volatility=Decimal("0.02"),
        ),
        metadata=None,
        trace_id="test-trace-001",
    )


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
async def test_check_balance_buy_with_short_position_checks_commission(buy_signal, mock_position_short):
    """Test that balance check verifies commission for buy orders with short position (reduce-only)."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient for commission
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("100.0")  # Sufficient for commission
            
            # Mock position manager to return short position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_short
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance with is_reduce_only=True
                    result = await risk_manager.check_balance(
                        signal=buy_signal,
                        order_quantity=Decimal("0.01"),
                        order_price=Decimal("50000.0"),
                        is_reduce_only=True,
                    )
                    
                    # Should return True (commission balance sufficient)
                    assert result is True
                    
                    # Verify position was checked
                    mock_get_position.assert_called_once_with(
                        buy_signal.asset,
                        mode="one-way",
                        trace_id=buy_signal.trace_id,
                    )
                    
                    # Verify commission was calculated
                    mock_get_fee.assert_called_once()


@pytest.mark.asyncio
async def test_check_balance_sell_with_long_position_checks_commission(sell_signal, mock_position_long):
    """Test that balance check verifies commission for sell orders with long position (reduce-only)."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient for commission
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("100.0")  # Sufficient for commission
            
            # Mock position manager to return long position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_long
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance
                    result = await risk_manager.check_balance(
                        signal=sell_signal,
                        order_quantity=Decimal("0.01"),
                        order_price=Decimal("50000.0"),
                    )
                    
                    # Should return True (commission balance sufficient)
                    assert result is True
                    
                    # Verify position was checked
                    mock_get_position.assert_called_once_with(
                        sell_signal.asset,
                        mode="one-way",
                        trace_id=sell_signal.trace_id,
                    )
                    
                    # Verify commission was calculated
                    mock_get_fee.assert_called_once()


@pytest.mark.asyncio
async def test_check_balance_buy_without_position_checks_balance(buy_signal):
    """Test that balance check is performed for buy orders without position."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient balance
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("10000.0")  # Sufficient balance
            
            # Mock position manager to return None (no position)
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = None
                
                # Call check_balance
                result = await risk_manager.check_balance(
                    signal=buy_signal,
                    order_quantity=Decimal("0.01"),
                    order_price=Decimal("50000.0"),
                )
                
                # Should return True (balance sufficient)
                assert result is True
                
                # Verify position was checked
                mock_get_position.assert_called_once_with(
                    buy_signal.asset,
                    mode="one-way",
                    trace_id=buy_signal.trace_id,
                )


@pytest.mark.asyncio
async def test_check_balance_sell_without_position_checks_margin(sell_signal):
    """Test that balance check is performed for sell orders without position (needs margin)."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("10000.0")
            
            # Mock position manager to return None (no position)
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = None
                
                # Mock database query for margin
                with patch('src.services.risk_manager.DatabaseConnection.get_pool') as mock_pool:
                    mock_pool_instance = AsyncMock()
                    mock_pool.return_value = mock_pool_instance
                    
                    # Mock fetchrow to return margin data
                    mock_row = MagicMock()
                    mock_row.__getitem__ = lambda self, key: {
                        "total_available_balance": Decimal("10000.0"),
                        "base_currency": "USDT",
                        "received_at": datetime.now(timezone.utc),
                    }[key]
                    mock_pool_instance.fetchrow = AsyncMock(return_value=mock_row)
                    
                    # Call check_balance
                    result = await risk_manager.check_balance(
                        signal=sell_signal,
                        order_quantity=Decimal("0.01"),
                        order_price=Decimal("50000.0"),
                    )
                    
                    # Should return True (margin sufficient)
                    assert result is True

