"""Unit tests for OrderExecutor handling of error 110007 with reduce-only orders."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.order_executor import OrderExecutor
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.models.position import Position
from src.models.order import Order
from src.exceptions import OrderExecutionError


@pytest.fixture
def sample_signal():
    """Create a sample trading signal for testing."""
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
async def test_110007_reduce_only_position_already_closed(sample_signal):
    """Test that error 110007 for reduce-only order returns None when position is already closed."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110007
    mock_bybit_response = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock position manager client class (new instance is created inside the method)
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=None)  # Position is closed
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response)
        
        # Mock PositionManagerClient class to return our mock (patch at the import location inside the method)
        # The import happens inside the method: from ..services.position_manager_client import PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Call create_order with force_reduce_only=True
            result = await executor.create_order(
                signal=sample_signal,
                order_type="Market",
                quantity=Decimal("0.01"),
                price=None,
                trace_id="test-trace",
                force_reduce_only=True,
            )
            
            # Should return None (position already closed - success case)
            assert result is None
            
            # Verify position was checked
            mock_position_client.get_position_from_bybit.assert_called_once_with(
                asset="BTCUSDT",
                trace_id="test-trace",
            )


@pytest.mark.asyncio
async def test_110007_reduce_only_position_exists_insufficient_commission(sample_signal, mock_position_long):
    """Test that error 110007 for reduce-only order checks commission balance when position exists."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110007
    mock_bybit_response = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock position manager client class (new instance is created inside the method)
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)  # Position still exists
    
    # Mock risk manager for commission check
    mock_risk_manager = MagicMock()
    from src.exceptions import RiskLimitError
    mock_risk_manager.check_balance = AsyncMock(side_effect=RiskLimitError("Insufficient balance for commission"))
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response)
        
        # Mock PositionManagerClient class to return our mock
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager class to return our mock (imported inside the method)
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Call create_order with force_reduce_only=True
                with pytest.raises(OrderExecutionError) as exc_info:
                    await executor.create_order(
                        signal=sample_signal,
                        order_type="Market",
                        quantity=Decimal("0.01"),
                        price=None,
                        trace_id="test-trace",
                        force_reduce_only=True,
                    )
                
                # Should raise OrderExecutionError with commission-related message
                error_msg = str(exc_info.value).lower()
                assert ("110007" in str(exc_info.value) or "ab not enough" in error_msg or 
                        "commission" in error_msg or "insufficient balance" in error_msg)
                
                # Verify position was checked
                mock_position_client.get_position_from_bybit.assert_called_once_with(
                    asset="BTCUSDT",
                    trace_id="test-trace",
                )
                
                # Verify commission balance was checked
                mock_risk_manager.check_balance.assert_called_once()


@pytest.mark.asyncio
async def test_110007_reduce_only_position_exists_sufficient_commission(sample_signal, mock_position_long):
    """Test that error 110007 for reduce-only order raises unexpected error when commission balance is sufficient."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110007
    mock_bybit_response = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock position manager client class (new instance is created inside the method)
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)  # Position still exists
    
    # Mock risk manager for commission check - balance is sufficient
    mock_risk_manager = MagicMock()
    mock_risk_manager.check_balance = AsyncMock(return_value=True)  # Commission balance sufficient
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response)
        
        # Mock PositionManagerClient class to return our mock
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager class to return our mock (imported inside the method)
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Call create_order with force_reduce_only=True
                with pytest.raises(OrderExecutionError) as exc_info:
                    await executor.create_order(
                        signal=sample_signal,
                        order_type="Market",
                        quantity=Decimal("0.01"),
                        price=None,
                        trace_id="test-trace",
                        force_reduce_only=True,
                    )
                
                # Should raise OrderExecutionError with unexpected error message
                error_msg = str(exc_info.value).lower()
                assert ("110007" in str(exc_info.value) or "ab not enough" in error_msg or 
                        "unexpected" in error_msg or "api issue" in error_msg)
                
                # Verify position was checked
                mock_position_client.get_position_from_bybit.assert_called_once_with(
                    asset="BTCUSDT",
                    trace_id="test-trace",
                )
                
                # Verify commission balance was checked
                mock_risk_manager.check_balance.assert_called_once()


@pytest.mark.asyncio
async def test_110007_reduce_only_position_check_fails(sample_signal):
    """Test that error 110007 for reduce-only order handles position check failure gracefully."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110007
    mock_bybit_response = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock position manager client class (new instance is created inside the method)
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(side_effect=Exception("Network error"))
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response)
        
        # Mock PositionManagerClient class to return our mock
        # The import happens inside the method: from ..services.position_manager_client import PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Call create_order with force_reduce_only=True
            # Should continue with normal error handling
            with pytest.raises(OrderExecutionError):
                await executor.create_order(
                    signal=sample_signal,
                    order_type="Market",
                    quantity=Decimal("0.01"),
                    price=None,
                    trace_id="test-trace",
                    force_reduce_only=True,
                )


@pytest.mark.asyncio
async def test_110007_non_reduce_only_normal_handling(sample_signal):
    """Test that error 110007 for non-reduce-only orders uses normal error handling."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110007
    mock_bybit_response = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock order size reduction to return None (reduction disabled or failed)
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response)
        
        with patch.object(executor, '_handle_insufficient_balance', new_callable=AsyncMock) as mock_reduce:
            mock_reduce.return_value = None  # Reduction failed/disabled
            
            with patch.object(executor, '_save_rejected_order', new_callable=AsyncMock) as mock_save:
                mock_save.return_value = None
                
                # Call create_order without force_reduce_only
                with pytest.raises(OrderExecutionError) as exc_info:
                    await executor.create_order(
                        signal=sample_signal,
                        order_type="Market",
                        quantity=Decimal("0.01"),
                        price=None,
                        trace_id="test-trace",
                        force_reduce_only=False,
                    )
                
                # Should raise OrderExecutionError
                assert "110007" in str(exc_info.value) or "ab not enough" in str(exc_info.value).lower()
                
                # Verify reduction was attempted (if enabled)
                # Note: This depends on settings.order_manager_enable_order_size_reduction

