"""Unit tests for OrderExecutor balance check after error 110017 (retry without reduceOnly)."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from src.services.order_executor import OrderExecutor
from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.models.position import Position
from src.models.order import Order
from src.exceptions import OrderExecutionError, RiskLimitError


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


@pytest.mark.asyncio
async def test_110017_retry_without_reduce_only_insufficient_balance(sample_signal, mock_position_long):
    """Test that after error 110017, balance is checked before retry without reduceOnly, and fails if insufficient."""
    executor = OrderExecutor()
    
    # Mock Bybit API responses:
    # 1. First attempt with reduceOnly -> error 110017
    # 2. Retry without reduceOnly -> error 110007 (but we should catch it before)
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "reduce-only order has same side with current position",
        "result": {},
    }
    
    # Mock position manager client
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock risk manager - balance insufficient
    mock_risk_manager = MagicMock()
    mock_risk_manager.check_balance = AsyncMock(
        side_effect=RiskLimitError("Insufficient balance: required=500.0 USDT, available=100.0 USDT")
    )
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        # First call returns 110017
        mock_client.post = AsyncMock(return_value=mock_bybit_response_110017)
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager (imported inside the method)
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Mock position manager for _prepare_bybit_order_params (to set reduceOnly)
                with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                    mock_get_pos.return_value = mock_position_long
                    
                    # Mock settings to enable balance check
                    with patch('src.services.order_executor.settings') as mock_settings:
                        mock_settings.order_manager_enable_dry_run = False
                        mock_settings.order_manager_enable_balance_check = True
                        mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                        mock_settings.bybit_market_category = "linear"
                        mock_settings.order_manager_tp_sl_enabled = False
                        mock_settings.order_manager_enable_min_notional_fee_check = False
                        
                        # Mock fee_rate_manager to avoid DB calls
                        with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                            mock_fee.return_value = None
                            
                            # Mock order validator
                            with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                                mock_validator = MagicMock()
                                mock_validator.validate_order_against_instruments_info = AsyncMock()
                                mock_validator_class.return_value = mock_validator
                                
                                # Mock save_rejected_order
                                with patch.object(executor, '_save_rejected_order', new_callable=AsyncMock) as mock_save:
                                    mock_save.return_value = None
                                    
                                    # Call create_order
                                    with pytest.raises(OrderExecutionError) as exc_info:
                                        await executor.create_order(
                                            signal=sample_signal,
                                            order_type="Market",
                                            quantity=Decimal("0.01"),
                                            price=None,
                                            trace_id="test-trace",
                                            force_reduce_only=False,  # Will be set by _prepare_bybit_order_params
                                        )
                                    
                                    # Should raise OrderExecutionError about insufficient balance
                                    error_msg = str(exc_info.value).lower()
                                    assert ("insufficient balance" in error_msg or 
                                            "cannot retry order without reduceonly" in error_msg or
                                            "balance" in error_msg)
                                    
                                    # Verify balance was checked
                                    mock_risk_manager.check_balance.assert_called_once()
                                    # Verify it was called with is_reduce_only=False (regular order)
                                    call_args = mock_risk_manager.check_balance.call_args
                                    assert call_args[1]['is_reduce_only'] is False


@pytest.mark.asyncio
async def test_110017_retry_without_reduce_only_sufficient_balance(sample_signal, mock_position_long):
    """Test that after error 110017, balance check passes and retry without reduceOnly succeeds."""
    executor = OrderExecutor()
    
    # Mock Bybit API responses:
    # 1. First attempt with reduceOnly -> error 110017
    # 2. Retry without reduceOnly -> success
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "reduce-only order has same side with current position",
        "result": {},
    }
    
    mock_bybit_response_success = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "orderId": "test-order-id-123",
        },
    }
    
    # Mock position manager client
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock risk manager - balance sufficient
    mock_risk_manager = MagicMock()
    mock_risk_manager.check_balance = AsyncMock(return_value=True)
    
    # Mock Bybit API - first call returns 110017, second call returns success
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=[
            mock_bybit_response_110017,  # First call with reduceOnly
            mock_bybit_response_success,  # Second call without reduceOnly
        ])
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager (imported inside the method)
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Mock position manager for _prepare_bybit_order_params
                with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                    mock_get_pos.return_value = mock_position_long
                    
                    # Mock settings
                    with patch('src.services.order_executor.settings') as mock_settings:
                        mock_settings.order_manager_enable_dry_run = False
                        mock_settings.order_manager_enable_balance_check = True
                        mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                        mock_settings.bybit_market_category = "linear"
                        mock_settings.order_manager_tp_sl_enabled = False
                        mock_settings.order_manager_enable_min_notional_fee_check = False
                        
                        # Mock fee_rate_manager to avoid DB calls
                        with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                            mock_fee.return_value = None
                            
                            # Mock order validator (imported inside the method)
                            with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                                mock_validator = MagicMock()
                                mock_validator.validate_order_against_instruments_info = AsyncMock()
                                mock_validator_class.return_value = mock_validator
                                
                                # Mock save_order_to_database
                                mock_order = MagicMock()
                                mock_order.id = uuid4()
                                mock_order.order_id = "test-order-id-123"
                                with patch.object(executor, '_save_order_to_database', new_callable=AsyncMock) as mock_save:
                                    mock_save.return_value = mock_order
                                    
                                    # Call create_order
                                    result = await executor.create_order(
                                        signal=sample_signal,
                                        order_type="Market",
                                        quantity=Decimal("0.01"),
                                        price=None,
                                        trace_id="test-trace",
                                        force_reduce_only=False,
                                    )
                                    
                                    # Should succeed
                                    assert result is not None
                                    
                                    # Verify balance was checked
                                    mock_risk_manager.check_balance.assert_called_once()
                                    # Verify it was called with is_reduce_only=False
                                    call_args = mock_risk_manager.check_balance.call_args
                                    assert call_args[1]['is_reduce_only'] is False
                                    
                                    # Verify two API calls were made (first with reduceOnly, second without)
                                    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_110017_retry_without_reduce_only_balance_check_disabled(sample_signal, mock_position_long):
    """Test that balance check is skipped if disabled when retrying without reduceOnly after 110017."""
    executor = OrderExecutor()
    
    # Mock Bybit API responses:
    # 1. First attempt with reduceOnly -> error 110017
    # 2. Retry without reduceOnly -> error 110007 (balance check disabled, so it goes through)
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "reduce-only order has same side with current position",
        "result": {},
    }
    
    mock_bybit_response_110007 = {
        "retCode": 110007,
        "retMsg": "ab not enough for new order",
        "result": {},
    }
    
    # Mock position manager client
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock risk manager (should not be called)
    mock_risk_manager = MagicMock()
    mock_risk_manager.check_balance = AsyncMock(return_value=True)
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=[
            mock_bybit_response_110017,  # First call
            mock_bybit_response_110007,  # Second call (retry without reduceOnly)
        ])
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager (should not be called)
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Mock position manager for _prepare_bybit_order_params
                with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                    mock_get_pos.return_value = mock_position_long
                    
                    # Mock settings - balance check disabled
                    with patch('src.services.order_executor.settings') as mock_settings:
                        mock_settings.order_manager_enable_dry_run = False
                        mock_settings.order_manager_enable_balance_check = False  # Disabled
                        mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                        mock_settings.bybit_market_category = "linear"
                        mock_settings.order_manager_tp_sl_enabled = False
                        mock_settings.order_manager_enable_order_size_reduction = False
                        mock_settings.order_manager_enable_min_notional_fee_check = False
                        
                        # Mock fee_rate_manager to avoid DB calls
                        with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                            mock_fee.return_value = None
                            
                            # Mock order validator (imported inside the method)
                            with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                                mock_validator = MagicMock()
                                mock_validator.validate_order_against_instruments_info = AsyncMock()
                                mock_validator_class.return_value = mock_validator
                                
                                # Mock save_rejected_order
                                with patch.object(executor, '_save_rejected_order', new_callable=AsyncMock) as mock_save:
                                    mock_save.return_value = None
                                    
                                    # Call create_order
                                    with pytest.raises(OrderExecutionError):
                                        await executor.create_order(
                                            signal=sample_signal,
                                            order_type="Market",
                                            quantity=Decimal("0.01"),
                                            price=None,
                                            trace_id="test-trace",
                                            force_reduce_only=False,
                                        )
                                    
                                    # Verify balance check was NOT called (disabled)
                                    mock_risk_manager.check_balance.assert_not_called()


@pytest.mark.asyncio
async def test_110017_retry_without_reduce_only_balance_check_error(sample_signal, mock_position_long):
    """Test that balance check error is handled gracefully when retrying without reduceOnly after 110017."""
    executor = OrderExecutor()
    
    # Mock Bybit API responses:
    # 1. First attempt with reduceOnly -> error 110017
    # 2. Retry without reduceOnly -> success (balance check failed but we continue)
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "reduce-only order has same side with current position",
        "result": {},
    }
    
    mock_bybit_response_success = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "orderId": "test-order-id-123",
        },
    }
    
    # Mock position manager client
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_long)
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock risk manager - balance check raises exception (not RiskLimitError)
    mock_risk_manager = MagicMock()
    mock_risk_manager.check_balance = AsyncMock(side_effect=Exception("Network error during balance check"))
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=[
            mock_bybit_response_110017,  # First call
            mock_bybit_response_success,  # Second call (retry succeeds despite balance check error)
        ])
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock RiskManager
            with patch('src.services.risk_manager.RiskManager', return_value=mock_risk_manager):
                # Mock position manager for _prepare_bybit_order_params
                with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                    mock_get_pos.return_value = mock_position_long
                    
                    # Mock settings
                    with patch('src.services.order_executor.settings') as mock_settings:
                        mock_settings.order_manager_enable_dry_run = False
                        mock_settings.order_manager_enable_balance_check = True
                        mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                        mock_settings.bybit_market_category = "linear"
                        mock_settings.order_manager_tp_sl_enabled = False
                        mock_settings.order_manager_enable_min_notional_fee_check = False
                        
                        # Mock fee_rate_manager to avoid DB calls
                        with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                            mock_fee.return_value = None
                            
                            # Mock order validator (imported inside the method)
                            with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                                mock_validator = MagicMock()
                                mock_validator.validate_order_against_instruments_info = AsyncMock()
                                mock_validator_class.return_value = mock_validator
                                
                                # Mock save_order_to_database
                                mock_order = MagicMock()
                                mock_order.id = uuid4()
                                mock_order.order_id = "test-order-id-123"
                                with patch.object(executor, '_save_order_to_database', new_callable=AsyncMock) as mock_save:
                                    mock_save.return_value = mock_order
                                    
                                    # Call create_order - should succeed despite balance check error
                                    result = await executor.create_order(
                                        signal=sample_signal,
                                        order_type="Market",
                                        quantity=Decimal("0.01"),
                                        price=None,
                                        trace_id="test-trace",
                                        force_reduce_only=False,
                                    )
                                    
                                    # Should succeed (balance check error is logged but doesn't block)
                                    assert result is not None
                                    
                                    # Verify balance check was attempted
                                    mock_risk_manager.check_balance.assert_called_once()


@pytest.mark.asyncio
async def test_110017_position_closed_force_reduce_only_returns_none(sample_signal):
    """Test that error 110017 with force_reduce_only returns None when position is already closed."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110017
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "current position is zero, cannot fix reduce-only order qty",
        "result": {},
    }
    
    # Mock position manager client - position is closed (None)
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=None)  # Position closed
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response_110017)
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock position manager for _prepare_bybit_order_params
            with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                mock_get_pos.return_value = None
                
                # Mock settings
                with patch('src.services.order_executor.settings') as mock_settings:
                    mock_settings.order_manager_enable_dry_run = False
                    mock_settings.order_manager_enable_balance_check = True
                    mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                    mock_settings.bybit_market_category = "linear"
                    mock_settings.order_manager_tp_sl_enabled = False
                    mock_settings.order_manager_enable_min_notional_fee_check = False
                    
                    # Mock fee_rate_manager
                    with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                        mock_fee.return_value = None
                        
                        # Mock order validator
                        with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                            mock_validator = MagicMock()
                            mock_validator.validate_order_against_instruments_info = AsyncMock()
                            mock_validator_class.return_value = mock_validator
                            
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
async def test_110017_position_closed_zero_size_returns_none(sample_signal):
    """Test that error 110017 returns None when position size is zero."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110017
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "current position is zero, cannot fix reduce-only order qty",
        "result": {},
    }
    
    # Mock position with zero size
    mock_position_zero = Position(
        id=uuid4(),
        asset="BTCUSDT",
        size=Decimal("0.0"),  # Zero size
        average_entry_price=Decimal("50000.0"),
        unrealized_pnl=Decimal("0.0"),
        realized_pnl=Decimal("0.0"),
        mode="one-way",
        last_updated=datetime.now(timezone.utc),
    )
    
    # Mock position manager client
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(return_value=mock_position_zero)
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response_110017)
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock position manager for _prepare_bybit_order_params
            with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                mock_get_pos.return_value = mock_position_zero
                
                # Mock settings
                with patch('src.services.order_executor.settings') as mock_settings:
                    mock_settings.order_manager_enable_dry_run = False
                    mock_settings.order_manager_enable_balance_check = True
                    mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                    mock_settings.bybit_market_category = "linear"
                    mock_settings.order_manager_tp_sl_enabled = False
                    mock_settings.order_manager_enable_min_notional_fee_check = False
                    
                    # Mock fee_rate_manager
                    with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                        mock_fee.return_value = None
                        
                        # Mock order validator
                        with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                            mock_validator = MagicMock()
                            mock_validator.validate_order_against_instruments_info = AsyncMock()
                            mock_validator_class.return_value = mock_validator
                            
                            # Call create_order with force_reduce_only=True
                            result = await executor.create_order(
                                signal=sample_signal,
                                order_type="Market",
                                quantity=Decimal("0.01"),
                                price=None,
                                trace_id="test-trace",
                                force_reduce_only=True,
                            )
                            
                            # Should return None (position closed - success case)
                            assert result is None
                            
                            # Verify position was checked
                            mock_position_client.get_position_from_bybit.assert_called_once()


@pytest.mark.asyncio
async def test_110017_position_fetch_fails_raises_error(sample_signal):
    """Test that error 110017 raises exception when position fetch fails."""
    executor = OrderExecutor()
    
    # Mock Bybit API response with error 110017
    mock_bybit_response_110017 = {
        "retCode": 110017,
        "retMsg": "current position is zero, cannot fix reduce-only order qty",
        "result": {},
    }
    
    # Mock position manager client - fetch fails
    mock_position_client = MagicMock()
    mock_position_client.get_position_from_bybit = AsyncMock(side_effect=Exception("Network error"))
    mock_position_client.trigger_bybit_sync_async = AsyncMock()
    
    # Mock Bybit API
    with patch('src.services.order_executor.get_bybit_client') as mock_bybit_client:
        mock_client = MagicMock()
        mock_bybit_client.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_bybit_response_110017)
        mock_client.get = AsyncMock(return_value={
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {"qtyStep": "0.001"},
                    "priceFilter": {"tickSize": "0.01"},
                }]
            }
        })
        
        # Mock PositionManagerClient
        with patch('src.services.position_manager_client.PositionManagerClient', return_value=mock_position_client):
            # Mock position manager for _prepare_bybit_order_params
            with patch.object(executor.position_manager_client, 'get_position', new_callable=AsyncMock) as mock_get_pos:
                mock_get_pos.return_value = None
                
                # Mock settings
                with patch('src.services.order_executor.settings') as mock_settings:
                    mock_settings.order_manager_enable_dry_run = False
                    mock_settings.order_manager_enable_balance_check = True
                    mock_settings.order_manager_auto_sync_position_after_bybit_fetch = False
                    mock_settings.bybit_market_category = "linear"
                    mock_settings.order_manager_tp_sl_enabled = False
                    mock_settings.order_manager_enable_min_notional_fee_check = False
                    
                    # Mock fee_rate_manager
                    with patch.object(executor.fee_rate_manager, 'get_fee_rate', new_callable=AsyncMock) as mock_fee:
                        mock_fee.return_value = None
                        
                        # Mock order validator
                        with patch('src.services.order_validator.OrderValidator') as mock_validator_class:
                            mock_validator = MagicMock()
                            mock_validator.validate_order_against_instruments_info = AsyncMock()
                            mock_validator_class.return_value = mock_validator
                            
                            # Call create_order - should raise OrderExecutionError
                            with pytest.raises(OrderExecutionError) as exc_info:
                                await executor.create_order(
                                    signal=sample_signal,
                                    order_type="Market",
                                    quantity=Decimal("0.01"),
                                    price=None,
                                    trace_id="test-trace",
                                    force_reduce_only=True,
                                )
                            
                            # Should raise OrderExecutionError about failed position verification
                            error_msg = str(exc_info.value).lower()
                            assert ("cannot retry order without reduceonly" in error_msg or
                                    "failed to verify position existence" in error_msg or
                                    "position fetch failed" in error_msg)
                            
                            # Verify position fetch was attempted
                            mock_position_client.get_position_from_bybit.assert_called_once()

