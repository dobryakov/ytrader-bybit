"""Unit tests for TP/SL calculation in OrderExecutor."""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from src.models.trading_signal import TradingSignal, MarketDataSnapshot
from src.services.order_executor import OrderExecutor
from src.config.settings import settings


@pytest.fixture
def order_executor():
    """Create OrderExecutor instance for testing."""
    executor = OrderExecutor()
    executor.instrument_info_manager = MagicMock()
    executor.instrument_info_manager.get_instrument = AsyncMock()
    return executor


@pytest.fixture
def sample_signal_buy():
    """Create sample buy signal for testing."""
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
def sample_signal_sell():
    """Create sample sell signal for testing."""
    return TradingSignal(
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


@pytest.fixture
def mock_instrument_info():
    """Create mock instrument info with tick size."""
    mock_info = MagicMock()
    mock_info.price_tick_size = Decimal("0.01")
    return mock_info


class TestCalculateTpSlFromSettings:
    """Test TP/SL calculation from settings."""

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_buy_order_with_defaults(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation for buy order with default settings."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        entry_price = Decimal("50000.0")
        tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
            signal=sample_signal_buy,
            entry_price=entry_price,
        )
        
        # TP = 50000 * (1 + 0.03) = 51500.0
        assert tp_price == Decimal("51500.00")
        # SL = 50000 * (1 - 0.02) = 49000.0
        assert sl_price == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_sell_order_with_defaults(
        self, order_executor, sample_signal_sell, mock_instrument_info
    ):
        """Test TP/SL calculation for sell order with default settings."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        entry_price = Decimal("3000.0")
        tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
            signal=sample_signal_sell,
            entry_price=entry_price,
        )
        
        # TP = 3000 * (1 - 0.03) = 2910.0
        assert tp_price == Decimal("2910.00")
        # SL = 3000 * (1 + 0.02) = 3060.0
        assert sl_price == Decimal("3060.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_with_custom_thresholds(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with custom thresholds."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        with patch.object(settings, "order_manager_tp_threshold_pct", 5.0):
            with patch.object(settings, "order_manager_sl_threshold_pct", -1.5):
                entry_price = Decimal("50000.0")
                tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
                    signal=sample_signal_buy,
                    entry_price=entry_price,
                )
                
                # TP = 50000 * (1 + 0.05) = 52500.0
                assert tp_price == Decimal("52500.00")
                # SL = 50000 * (1 - 0.015) = 49250.0
                assert sl_price == Decimal("49250.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_tp_disabled(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation when TP is disabled."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        with patch.object(settings, "order_manager_tp_enabled", False):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            assert tp_price is None
            assert sl_price == Decimal("49000.00")  # SL still calculated

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_sl_disabled(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation when SL is disabled."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        with patch.object(settings, "order_manager_sl_enabled", False):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            assert tp_price == Decimal("51500.00")  # TP still calculated
            assert sl_price is None

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_price_rounding(
        self, order_executor, sample_signal_buy
    ):
        """Test TP/SL price rounding to tick size."""
        mock_info = MagicMock()
        mock_info.price_tick_size = Decimal("0.1")  # Different tick size
        order_executor.instrument_info_manager.get_instrument.return_value = mock_info
        
        entry_price = Decimal("50000.0")
        tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
            signal=sample_signal_buy,
            entry_price=entry_price,
        )
        
        # TP = 51500.0, rounded to 0.1 tick size = 51500.0
        assert tp_price == Decimal("51500.0")
        # SL = 49000.0, rounded to 0.1 tick size = 49000.0
        assert sl_price == Decimal("49000.0")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_instrument_info_not_found(
        self, order_executor, sample_signal_buy
    ):
        """Test TP/SL calculation when instrument info is not found."""
        order_executor.instrument_info_manager.get_instrument.return_value = None
        
        entry_price = Decimal("50000.0")
        tp_price, sl_price = await order_executor._calculate_tp_sl_from_settings(
            signal=sample_signal_buy,
            entry_price=entry_price,
        )
        
        # Should use default tick size (0.01)
        assert tp_price == Decimal("51500.00")
        assert sl_price == Decimal("49000.00")


class TestCalculateTpSlHybrid:
    """Test hybrid TP/SL calculation (metadata priority)."""

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_from_metadata_priority_metadata(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with metadata priority = 'metadata'."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Add metadata to signal
        sample_signal_buy.metadata = {
            "take_profit_price": 52000.0,
            "stop_loss_price": 48000.0,
        }
        
        with patch.object(settings, "order_manager_tp_sl_priority", "metadata"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should use metadata values
            assert tp_price == Decimal("52000.0")
            assert sl_price == Decimal("48000.0")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_from_metadata_priority_both(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with metadata priority = 'both'."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Add metadata to signal
        sample_signal_buy.metadata = {
            "take_profit_price": 52000.0,
            "stop_loss_price": 48000.0,
        }
        
        with patch.object(settings, "order_manager_tp_sl_priority", "both"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should use metadata values (priority)
            assert tp_price == Decimal("52000.0")
            assert sl_price == Decimal("48000.0")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_fallback_to_settings_when_no_metadata(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation falls back to settings when no metadata."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # No metadata
        sample_signal_buy.metadata = None
        
        with patch.object(settings, "order_manager_tp_sl_priority", "both"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should use settings values
            assert tp_price == Decimal("51500.00")
            assert sl_price == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_partial_metadata(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with partial metadata (only TP)."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Only TP in metadata
        sample_signal_buy.metadata = {
            "take_profit_price": 52000.0,
        }
        
        with patch.object(settings, "order_manager_tp_sl_priority", "both"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # TP from metadata, SL from settings
            assert tp_price == Decimal("52000.0")
            assert sl_price == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_priority_settings(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with priority = 'settings' (ignore metadata)."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Add metadata to signal
        sample_signal_buy.metadata = {
            "take_profit_price": 52000.0,
            "stop_loss_price": 48000.0,
        }
        
        with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should use settings values (ignore metadata)
            assert tp_price == Decimal("51500.00")
            assert sl_price == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_priority_metadata_no_metadata_fallback(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with priority = 'metadata' but no TP/SL in metadata (fallback to settings)."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Metadata exists but doesn't contain TP/SL
        sample_signal_buy.metadata = {
            "reasoning": "Model prediction",
            "prediction_result": {"prediction": "buy"},
        }
        
        with patch.object(settings, "order_manager_tp_sl_priority", "metadata"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should fallback to settings values
            assert tp_price == Decimal("51500.00")
            assert sl_price == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_calculate_tp_sl_priority_metadata_none_metadata_fallback(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test TP/SL calculation with priority = 'metadata' but metadata is None (fallback to settings)."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        
        # Metadata is None
        sample_signal_buy.metadata = None
        
        with patch.object(settings, "order_manager_tp_sl_priority", "metadata"):
            entry_price = Decimal("50000.0")
            tp_price, sl_price = await order_executor._calculate_tp_sl(
                signal=sample_signal_buy,
                entry_price=entry_price,
            )
            
            # Should fallback to settings values
            assert tp_price == Decimal("51500.00")
            assert sl_price == Decimal("49000.00")


class TestPrepareBybitOrderParamsWithTpSl:
    """Test _prepare_bybit_order_params with TP/SL."""

    @pytest.mark.asyncio
    async def test_prepare_bybit_params_with_tp_sl_enabled(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test Bybit params preparation with TP/SL enabled."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        order_executor.position_manager = MagicMock()
        order_executor.position_manager.get_position = AsyncMock(return_value=None)
        
        with patch.object(settings, "order_manager_tp_sl_enabled", True):
            with patch.object(settings, "order_manager_tp_enabled", True):
                with patch.object(settings, "order_manager_sl_enabled", True):
                    with patch.object(settings, "order_manager_tp_sl_priority", "settings"):
                        with patch.object(settings, "order_manager_tp_sl_trigger_by", "LastPrice"):
                            params = await order_executor._prepare_bybit_order_params(
                                signal=sample_signal_buy,
                                order_type="Market",
                                quantity=Decimal("0.02"),
                                price=None,
                            )
                            
                            assert "takeProfit" in params
                            assert "stopLoss" in params
                            assert params["tpTriggerBy"] == "LastPrice"
                            assert params["slTriggerBy"] == "LastPrice"
                            assert Decimal(params["takeProfit"]) == Decimal("51500.00")
                            assert Decimal(params["stopLoss"]) == Decimal("49000.00")

    @pytest.mark.asyncio
    async def test_prepare_bybit_params_with_tp_sl_disabled(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test Bybit params preparation with TP/SL disabled."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        order_executor.position_manager = MagicMock()
        order_executor.position_manager.get_position = AsyncMock(return_value=None)
        
        with patch.object(settings, "order_manager_tp_sl_enabled", False):
            params = await order_executor._prepare_bybit_order_params(
                signal=sample_signal_buy,
                order_type="Market",
                quantity=Decimal("0.02"),
                price=None,
            )
            
            assert "takeProfit" not in params
            assert "stopLoss" not in params

    @pytest.mark.asyncio
    async def test_prepare_bybit_params_with_tp_sl_from_metadata(
        self, order_executor, sample_signal_buy, mock_instrument_info
    ):
        """Test Bybit params preparation with TP/SL from metadata."""
        order_executor.instrument_info_manager.get_instrument.return_value = mock_instrument_info
        order_executor.position_manager = MagicMock()
        order_executor.position_manager.get_position = AsyncMock(return_value=None)
        
        # Add metadata to signal
        sample_signal_buy.metadata = {
            "take_profit_price": 52000.0,
            "stop_loss_price": 48000.0,
        }
        
        with patch.object(settings, "order_manager_tp_sl_enabled", True):
            with patch.object(settings, "order_manager_tp_enabled", True):
                with patch.object(settings, "order_manager_sl_enabled", True):
                    with patch.object(settings, "order_manager_tp_sl_priority", "metadata"):
                        params = await order_executor._prepare_bybit_order_params(
                            signal=sample_signal_buy,
                            order_type="Market",
                            quantity=Decimal("0.02"),
                            price=None,
                        )
                        
                        assert Decimal(params["takeProfit"]) == Decimal("52000.0")
                        assert Decimal(params["stopLoss"]) == Decimal("48000.0")

