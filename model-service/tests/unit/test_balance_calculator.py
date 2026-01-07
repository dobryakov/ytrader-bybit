"""
Unit tests for BalanceCalculator.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.services.balance_calculator import BalanceCalculator
from src.services.position_manager_client import position_manager_client


@pytest.fixture
def balance_calculator():
    """Create BalanceCalculator instance for testing."""
    return BalanceCalculator(safety_margin=0.95)


@pytest.fixture
def mock_balance_data():
    """Mock balance data from database."""
    return {
        "coin": "USDT",
        "available_balance": Decimal("1000.0"),
        "wallet_balance": Decimal("1200.0"),
        "frozen": Decimal("200.0"),
        "received_at": datetime.now(timezone.utc),
        "event_timestamp": datetime.now(timezone.utc),
    }


@pytest.fixture
def mock_position_data():
    """Mock position data from Position Manager."""
    return {
        "asset": "ETHUSDT",
        "size": "1.5",  # 1.5 ETH
        "unrealized_pnl_pct": 2.5,
        "unrealized_pnl": 50.0,
        "closed_at": None,
    }


@pytest.mark.asyncio
async def test_get_available_balance_for_buy_success(balance_calculator, mock_balance_data):
    """Test successful retrieval of available balance for BUY order."""
    # Mock account-level balance to return None (fallback to coin-level)
    with patch.object(balance_calculator, "_get_account_level_balance", new_callable=AsyncMock) as mock_account_balance:
        mock_account_balance.return_value = None
        
        with patch.object(balance_calculator.balance_repo, "get_latest_balance", new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = mock_balance_data
            
            available = await balance_calculator.get_available_balance_for_buy(
                trading_pair="ETHUSDT",
                current_price=3000.0,
            )
            
            assert available is not None
            assert available == 950.0  # 1000 * 0.95 (safety_margin)
            mock_get_balance.assert_called_once_with("USDT")


@pytest.mark.asyncio
async def test_get_available_balance_for_buy_no_balance(balance_calculator):
    """Test when balance data is unavailable."""
    # Mock account-level balance to return None (fallback to coin-level)
    with patch.object(balance_calculator, "_get_account_level_balance", new_callable=AsyncMock) as mock_account_balance:
        mock_account_balance.return_value = None
        
        with patch.object(balance_calculator.balance_repo, "get_latest_balance", new_callable=AsyncMock) as mock_get_balance:
            mock_get_balance.return_value = None
            with patch.object(balance_calculator, "_trigger_balance_sync", new_callable=AsyncMock) as mock_sync:
                mock_sync.return_value = {"success": False, "updated_coins": [], "updated_count": 0}
                
                available = await balance_calculator.get_available_balance_for_buy(
                    trading_pair="ETHUSDT",
                    current_price=3000.0,
                )
                
                assert available is None




@pytest.mark.asyncio
async def test_get_available_position_for_sell_success(balance_calculator, mock_position_data):
    """Test successful retrieval of available position for SELL order."""
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = mock_position_data
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=3000.0,
        )
        
        assert available is not None
        # 1.5 ETH * 0.95 (safety_margin) * 3000 (price) = 4275 USDT
        assert available == pytest.approx(4275.0, rel=0.01)
        mock_get_position.assert_called_once_with("ETHUSDT")


@pytest.mark.asyncio
async def test_get_available_position_for_sell_no_position(balance_calculator):
    """Test when position is not found."""
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = None
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=3000.0,
        )
        
        assert available is None


@pytest.mark.asyncio
async def test_get_available_position_for_sell_closed_position(balance_calculator):
    """Test when position is closed."""
    closed_position = {
        "asset": "ETHUSDT",
        "size": "1.5",
        "closed_at": datetime.now(timezone.utc),
    }
    
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = closed_position
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=3000.0,
        )
        
        assert available is None


@pytest.mark.asyncio
async def test_get_available_position_for_sell_short_position(balance_calculator):
    """Test when position is short (negative size)."""
    short_position = {
        "asset": "ETHUSDT",
        "size": "-1.5",  # Short position
        "closed_at": None,
    }
    
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = short_position
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=3000.0,
        )
        
        assert available is None


@pytest.mark.asyncio
async def test_get_available_position_for_sell_no_price(balance_calculator, mock_position_data):
    """Test when current price is not provided."""
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = mock_position_data
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=None,
        )
        
        assert available is None


@pytest.mark.asyncio
async def test_get_available_position_for_sell_zero_price(balance_calculator, mock_position_data):
    """Test when current price is zero or negative."""
    with patch.object(position_manager_client, "get_position", new_callable=AsyncMock) as mock_get_position:
        mock_get_position.return_value = mock_position_data
        
        available = await balance_calculator.get_available_position_for_sell(
            trading_pair="ETHUSDT",
            current_price=0.0,
        )
        
        assert available is None


@pytest.mark.asyncio
async def test_calculate_amount_with_available_resource():
    """Test that _calculate_amount respects available_resource limit."""
    from src.services.intelligent_signal_generator import IntelligentSignalGenerator
    
    generator = IntelligentSignalGenerator(
        min_confidence_threshold=0.6,
        min_amount=100.0,
        max_amount=1000.0,
    )
    
    # Test with available_resource that limits amount (but still >= min_amount)
    # min_amount parameter is optional, will use self.min_amount if not provided
    amount = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=200.0,  # Less than calculated amount, but >= min_amount
        min_amount=100.0,  # Explicitly pass min_amount
    )
    
    # Should be limited to 200.0 (available_resource)
    assert amount is not None
    assert amount == 200.0
    
    # Test with available_resource larger than calculated amount
    amount2 = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=2000.0,  # More than calculated amount
        min_amount=100.0,  # Explicitly pass min_amount
    )
    
    # Should be the calculated amount (not limited)
    assert amount2 is not None
    assert amount2 > 200.0
    assert amount2 <= 1000.0  # max_amount
    
    # Test with available_resource less than min_amount
    amount3 = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=50.0,  # Less than min_amount (100.0)
        min_amount=100.0,  # Explicitly pass min_amount
    )
    
    # Should return None because available_resource < min_amount
    assert amount3 is None


@pytest.mark.asyncio
async def test_calculate_amount_without_available_resource():
    """Test that _calculate_amount works without available_resource."""
    from src.services.intelligent_signal_generator import IntelligentSignalGenerator
    
    generator = IntelligentSignalGenerator(
        min_confidence_threshold=0.6,
        min_amount=100.0,
        max_amount=1000.0,
    )
    
    amount = generator._calculate_amount(
        current_price=3000.0,
        confidence=0.8,
        prediction_result=None,
        available_resource=None,
    )
    
    # Should calculate normally without resource limit
    assert amount is not None
    assert amount >= 100.0
    assert amount <= 1000.0

