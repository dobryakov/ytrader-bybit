"""Integration tests for TP/SL order creation."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.services.order_executor import OrderExecutor
from src.config.settings import settings
from src.config.database import DatabaseConnection


@pytest.fixture
async def db_pool():
    """Create database connection pool for testing."""
    pool = await DatabaseConnection.get_pool()
    yield pool
    await pool.close()


@pytest.fixture
def order_executor():
    """Create OrderExecutor instance for testing."""
    executor = OrderExecutor()
    executor.instrument_info_manager = MagicMock()
    executor.instrument_info_manager.get_instrument = AsyncMock()
    executor.position_manager_client = MagicMock()
    executor.position_manager_client.get_position = AsyncMock(return_value=None)
    return executor


@pytest.fixture
def sample_signal_with_metadata():
    """Create sample signal with TP/SL metadata."""
    return TradingSignal(
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
            "take_profit_price": 52000.0,
            "stop_loss_price": 48000.0,
        },
        trace_id=None,
    )


@pytest.fixture
def sample_signal_without_metadata():
    """Create sample signal without TP/SL metadata."""
    return TradingSignal(
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
        metadata=None,
        trace_id=None,
    )


@pytest.fixture
def mock_instrument_info():
    """Create mock instrument info."""
    mock_info = MagicMock()
    mock_info.price_tick_size = Decimal("0.01")
    return mock_info


@pytest.mark.asyncio
async def test_prepare_bybit_params_with_tp_sl_from_metadata(
    order_executor, sample_signal_with_metadata, mock_instrument_info
):
    """Test that TP/SL from metadata are correctly added to Bybit params."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", True):
            with patch.object(settings, "order_manager_sl_enabled", True):
                with patch.object(settings, "order_manager_tp_sl_priority", "metadata"):
                    with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                        # Mock market price to match metadata TP/SL expectations
                        # Use snapshot price to ensure TP (52000) > entry_price
                        with patch(
                            "src.services.order_type_selector.OrderTypeSelector._get_current_market_price",
                            new=AsyncMock(return_value=sample_signal_with_metadata.market_data_snapshot.price),
                        ):
                            params = await order_executor._prepare_bybit_order_params(
                                signal=sample_signal_with_metadata,
                                order_type="Market",
                                quantity=Decimal("0.02"),
                                price=None,
                            )
                            
                            # Verify TP/SL are in params
                            assert "takeProfit" in params
                            assert "stopLoss" in params
                            assert params["tpTriggerBy"] == "LastPrice"
                            assert params["slTriggerBy"] == "LastPrice"
                            
                            # Verify values from metadata
                            assert Decimal(params["takeProfit"]) == Decimal("52000.0")
                            assert Decimal(params["stopLoss"]) == Decimal("48000.0")


@pytest.mark.asyncio
async def test_prepare_bybit_params_with_tp_sl_from_settings(
    order_executor, sample_signal_without_metadata, mock_instrument_info
):
    """Test that TP/SL from settings are correctly added to Bybit params."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info

    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", True):
            with patch.object(settings, "order_manager_sl_enabled", True):
                with patch.object(settings, "order_manager_tp_threshold_pct", 3.0):
                    with patch.object(settings, "order_manager_sl_threshold_pct", -2.0):
                        with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                            with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                                # Ensure current market price used for TP/SL equals snapshot price for deterministic tests
                                with patch(
                                    "src.services.order_type_selector.OrderTypeSelector._get_current_market_price",
                                    new=AsyncMock(
                                        return_value=sample_signal_without_metadata.market_data_snapshot.price
                                    ),
                                ):
                                    params = await order_executor._prepare_bybit_order_params(
                                        signal=sample_signal_without_metadata,
                                        order_type="Market",
                                        quantity=Decimal("0.02"),
                                        price=None,
                                    )

                                # Verify TP/SL are in params
                                assert "takeProfit" in params
                                assert "stopLoss" in params
                                assert params["tpTriggerBy"] == "LastPrice"
                                assert params["slTriggerBy"] == "LastPrice"

                                # Verify calculated values (entry_price = 50000.0)
                                # TP = 50000 * (1 + 0.03) = 51500.0
                                # SL = 50000 * (1 - 0.02) = 49000.0
                                assert Decimal(params["takeProfit"]) == Decimal("51500.00")
                                assert Decimal(params["stopLoss"]) == Decimal("49000.00")


@pytest.mark.asyncio
async def test_prepare_bybit_params_tp_sl_disabled(
    order_executor, sample_signal_with_metadata, mock_instrument_info
):
    """Test that TP/SL are not added when disabled."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    with patch.object(settings, "order_manager_tp_sl_enabled", False):
        params = await order_executor._prepare_bybit_order_params(
            signal=sample_signal_with_metadata,
            order_type="Market",
            quantity=Decimal("0.02"),
            price=None,
        )
        
        # Verify TP/SL are NOT in params
        assert "takeProfit" not in params
        assert "stopLoss" not in params
        assert "tpTriggerBy" not in params
        assert "slTriggerBy" not in params


@pytest.mark.asyncio
async def test_prepare_bybit_params_tp_disabled_sl_enabled(
    order_executor, sample_signal_without_metadata, mock_instrument_info
):
    """Test that only SL is added when TP is disabled."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", False):
            with patch.object(settings, "order_manager_sl_enabled", True):
                with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                    with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                        params = await order_executor._prepare_bybit_order_params(
                            signal=sample_signal_without_metadata,
                            order_type="Market",
                            quantity=Decimal("0.02"),
                            price=None,
                        )
                        
                        # Verify only SL is in params
                        assert "takeProfit" not in params
                        assert "stopLoss" in params
                        assert params["slTriggerBy"] == "LastPrice"


@pytest.mark.asyncio
async def test_prepare_bybit_params_sl_disabled_tp_enabled(
    order_executor, sample_signal_without_metadata, mock_instrument_info
):
    """Test that only TP is added when SL is disabled."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", True):
            with patch.object(settings, "order_manager_sl_enabled", False):
                with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                    with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                        params = await order_executor._prepare_bybit_order_params(
                            signal=sample_signal_without_metadata,
                            order_type="Market",
                            quantity=Decimal("0.02"),
                            price=None,
                        )
                        
                        # Verify only TP is in params
                        assert "takeProfit" in params
                        assert "stopLoss" not in params
                        assert params["tpTriggerBy"] == "LastPrice"


@pytest.mark.asyncio
async def test_prepare_bybit_params_with_limit_order_and_tp_sl(
    order_executor, sample_signal_without_metadata, mock_instrument_info
):
    """Test TP/SL with limit order (entry price from limit price)."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    limit_price = Decimal("49900.0")
    
    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", True):
            with patch.object(settings, "order_manager_sl_enabled", True):
                with patch.object(settings, "order_manager_tp_threshold_pct", 3.0):
                    with patch.object(settings, "order_manager_sl_threshold_pct", -2.0):
                        with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                            with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                                params = await order_executor._prepare_bybit_order_params(
                                    signal=sample_signal_without_metadata,
                                    order_type="Limit",
                                    quantity=Decimal("0.02"),
                                    price=limit_price,
                                )
                                
                                # Verify TP/SL are calculated from limit price
                                assert "takeProfit" in params
                                assert "stopLoss" in params
                                
                                # TP = 49900 * (1 + 0.03) = 51397.0
                                # SL = 49900 * (1 - 0.02) = 48902.0
                                assert Decimal(params["takeProfit"]) == Decimal("51397.00")
                                assert Decimal(params["stopLoss"]) == Decimal("48902.00")


@pytest.mark.asyncio
async def test_prepare_bybit_params_sell_order_tp_sl(
    order_executor, mock_instrument_info
):
    """Test TP/SL calculation for sell orders (inverse logic)."""
    order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
    
    sell_signal = TradingSignal(
        signal_id=uuid4(),
        signal_type="sell",
        asset="ETHUSDT",
        amount=Decimal("500.0"),
        confidence=Decimal("0.80"),
        timestamp=datetime.now(timezone.utc),
        strategy_id="test-strategy",
        model_version="v1",
        is_warmup=False,
        market_data_snapshot=MarketDataSnapshot(
            price=Decimal("3000.0"),
            spread=Decimal("0.01"),
            volume_24h=Decimal("500000.0"),
            volatility=Decimal("0.015"),
        ),
        metadata=None,
        trace_id=None,
    )

    with patch.object(settings, "order_manager_tp_sl_enabled", True):
        with patch.object(settings, "order_manager_tp_enabled", True):
            with patch.object(settings, "order_manager_sl_enabled", True):
                with patch.object(settings, "order_manager_tp_threshold_pct", 3.0):
                    with patch.object(settings, "order_manager_sl_threshold_pct", -2.0):
                        with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                            with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                                with patch(
                                    "src.services.order_type_selector.OrderTypeSelector._get_current_market_price",
                                    new=AsyncMock(return_value=sell_signal.market_data_snapshot.price),
                                ):
                                    params = await order_executor._prepare_bybit_order_params(
                                        signal=sell_signal,
                                        order_type="Market",
                                        quantity=Decimal("0.1"),
                                        price=None,
                                    )

                                # Verify TP/SL are in params
                                assert "takeProfit" in params
                                assert "stopLoss" in params

                                # For sell orders:
                                # TP = 3000 * (1 - 0.03) = 2910.0 (lower price = profit)
                                # SL = 3000 * (1 + 0.02) = 3060.0 (higher price = loss)
                                assert Decimal(params["takeProfit"]) == Decimal("2910.00")
                                assert Decimal(params["stopLoss"]) == Decimal("3060.00")

