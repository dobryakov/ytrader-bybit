"""Unit tests for RiskManager commission balance check with reduce-only orders."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.risk_manager import RiskManager
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.models.position import Position
from src.exceptions import RiskLimitError


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
async def test_check_balance_reduce_only_insufficient_commission_buy(buy_signal, mock_position_short):
    """Test that reduce-only buy order fails when commission balance is insufficient."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - insufficient for commission
        # Order value: 0.01 * 50000 = 500 USDT
        # Commission (0.1%): 500 * 0.001 = 0.5 USDT
        # Required with buffer (10%): 0.5 * 1.1 = 0.55 USDT
        # Available: 0.1 USDT (insufficient)
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("0.1")  # Insufficient for commission
            
            # Mock position manager to return short position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_short
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance - should raise RiskLimitError
                    with pytest.raises(RiskLimitError) as exc_info:
                        await risk_manager.check_balance(
                            signal=buy_signal,
                            order_quantity=Decimal("0.01"),
                            order_price=Decimal("50000.0"),
                            is_reduce_only=True,
                        )
                    
                    # Verify error message mentions commission
                    assert "commission" in str(exc_info.value).lower() or "insufficient balance" in str(exc_info.value).lower()
                    
                    # Verify position was checked
                    mock_get_position.assert_called_once_with(
                        buy_signal.asset,
                        mode="one-way",
                        trace_id=buy_signal.trace_id,
                    )
                    
                    # Verify commission was calculated
                    mock_get_fee.assert_called_once()


@pytest.mark.asyncio
async def test_check_balance_reduce_only_insufficient_commission_sell(sell_signal, mock_position_long):
    """Test that reduce-only sell order fails when commission balance is insufficient."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - insufficient for commission
        # Order value: 0.01 * 50000 = 500 USDT
        # Commission (0.1%): 500 * 0.001 = 0.5 USDT
        # Required with buffer (10%): 0.5 * 1.1 = 0.55 USDT
        # Available: 0.1 USDT (insufficient)
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("0.1")  # Insufficient for commission
            
            # Mock position manager to return long position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_long
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance - should raise RiskLimitError
                    with pytest.raises(RiskLimitError) as exc_info:
                        await risk_manager.check_balance(
                            signal=sell_signal,
                            order_quantity=Decimal("0.01"),
                            order_price=Decimal("50000.0"),
                            is_reduce_only=True,
                        )
                    
                    # Verify error message mentions commission
                    assert "commission" in str(exc_info.value).lower() or "insufficient balance" in str(exc_info.value).lower()
                    
                    # Verify position was checked
                    mock_get_position.assert_called_once_with(
                        sell_signal.asset,
                        mode="one-way",
                        trace_id=sell_signal.trace_id,
                    )
                    
                    # Verify commission was calculated
                    mock_get_fee.assert_called_once()


@pytest.mark.asyncio
async def test_check_balance_reduce_only_sufficient_commission_buy(buy_signal, mock_position_short):
    """Test that reduce-only buy order passes when commission balance is sufficient."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient for commission
        # Order value: 0.01 * 50000 = 500 USDT
        # Commission (0.1%): 500 * 0.001 = 0.5 USDT
        # Required with buffer (10%): 0.5 * 1.1 = 0.55 USDT
        # Available: 1.0 USDT (sufficient)
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("1.0")  # Sufficient for commission
            
            # Mock position manager to return short position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_short
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance - should pass
                    result = await risk_manager.check_balance(
                        signal=buy_signal,
                        order_quantity=Decimal("0.01"),
                        order_price=Decimal("50000.0"),
                        is_reduce_only=True,
                    )
                    
                    # Should return True
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
async def test_check_balance_reduce_only_sufficient_commission_sell(sell_signal, mock_position_long):
    """Test that reduce-only sell order passes when commission balance is sufficient."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient for commission
        # Order value: 0.01 * 50000 = 500 USDT
        # Commission (0.1%): 500 * 0.001 = 0.5 USDT
        # Required with buffer (10%): 0.5 * 1.1 = 0.55 USDT
        # Available: 1.0 USDT (sufficient)
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("1.0")  # Sufficient for commission
            
            # Mock position manager to return long position
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = mock_position_long
                
                # Mock fee rate manager
                mock_fee_info = MagicMock()
                mock_fee_info.taker_fee_rate = Decimal("0.001")  # 0.1% fee
                with patch.object(risk_manager.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_get_fee:
                    mock_get_fee.return_value = mock_fee_info
                    
                    # Call check_balance - should pass
                    result = await risk_manager.check_balance(
                        signal=sell_signal,
                        order_quantity=Decimal("0.01"),
                        order_price=Decimal("50000.0"),
                        is_reduce_only=True,
                    )
                    
                    # Should return True
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
async def test_check_balance_reduce_only_with_is_reduce_only_flag(buy_signal):
    """Test that is_reduce_only flag triggers commission check even without position."""
    risk_manager = RiskManager()
    
    # Mock balance sync
    with patch.object(risk_manager, '_trigger_balance_sync', new_callable=AsyncMock) as mock_sync:
        mock_sync.return_value = True
        
        # Mock get balance from DB - sufficient for commission
        with patch.object(risk_manager, '_get_latest_usdt_balance_from_db', new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = Decimal("1.0")  # Sufficient for commission
            
            # Mock position manager to return None (no position)
            with patch.object(risk_manager.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_position:
                mock_get_position.return_value = None
                
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
                    
                    # Verify commission was calculated
                    mock_get_fee.assert_called_once()

