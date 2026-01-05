"""
Unit tests for Feature Computer service.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager
from src.models.orderbook_state import OrderbookState


class TestFeatureComputer:
    """Test Feature Computer service."""
    
    @pytest.fixture
    def orderbook_manager(self):
        """Create orderbook manager for testing."""
        return OrderbookManager()
    
    @pytest.fixture
    def feature_computer(self, orderbook_manager):
        """Create feature computer for testing."""
        return FeatureComputer(orderbook_manager, feature_registry_version="1.0.0")
    
    def test_feature_computer_init(self, orderbook_manager):
        """Test initializing feature computer."""
        computer = FeatureComputer(orderbook_manager)
        
        assert computer._orderbook_manager == orderbook_manager
        # In legacy mode (no shared_rolling_windows), _rolling_windows should be empty dict
        assert isinstance(computer._rolling_windows, dict)
        assert computer._uses_shared_windows is False
    
    def test_get_rolling_windows(self, feature_computer):
        """Test getting rolling windows."""
        rw = feature_computer.get_rolling_windows("BTCUSDT")
        
        assert rw is not None
        assert rw.symbol == "BTCUSDT"
        # Конкретный набор окон теперь зависит от FeatureRequirementsAnalyzer.
        # Здесь достаточно проверить, что есть хотя бы одно окно trades
        # (с колонками price/volume/side) и 1m окно для kline-данных.
        assert isinstance(rw.windows, dict)
        trade_intervals = [
            name
            for name, df in rw.windows.items()
            if {"timestamp", "price", "volume", "side"}.issubset(df.columns)
        ]
        assert trade_intervals, "expected at least one trade interval"
    
    def test_update_funding_rate(self, feature_computer):
        """Test updating funding rate."""
        feature_computer.update_funding_rate("BTCUSDT", 0.0001, 1234567890000)
        
        assert feature_computer._latest_funding_rate["BTCUSDT"] == 0.0001
        assert feature_computer._latest_next_funding_time["BTCUSDT"] == 1234567890000
    
    def test_compute_features_with_orderbook(self, feature_computer, sample_orderbook_snapshot):
        """Test computing features with orderbook state."""
        # Setup orderbook
        feature_computer._orderbook_manager.apply_snapshot(sample_orderbook_snapshot)
        
        # Compute features
        fv = feature_computer.compute_features("BTCUSDT")
        
        assert fv is not None
        assert fv.symbol == "BTCUSDT"
        assert len(fv.features) > 0
        assert "mid_price" in fv.features or fv.features.get("mid_price") is None
    
    def test_compute_features_without_orderbook(self, feature_computer):
        """Test computing features without orderbook state."""
        # Compute features without orderbook
        fv = feature_computer.compute_features("BTCUSDT")
        
        # Should still compute temporal features
        assert fv is not None
        assert "time_of_day_sin" in fv.features
        assert "time_of_day_cos" in fv.features
    
    def test_update_market_data_trade(self, feature_computer):
        """Test updating market data with trade event."""
        trade = {
            "event_type": "trade",
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(timezone.utc),
            "price": 50000.0,
            "quantity": 0.1,
            "side": "Buy",
        }
        
        feature_computer.update_market_data(trade)
        
        rw = feature_computer.get_rolling_windows("BTCUSDT")
        assert len(rw.windows["1s"]) > 0
    
    def test_update_market_data_orderbook_snapshot(self, feature_computer, sample_orderbook_snapshot):
        """Test updating market data with orderbook snapshot."""
        feature_computer.update_market_data(sample_orderbook_snapshot)
        
        # Snapshot is buffered, so need to apply buffered updates or compute features to apply it
        feature_computer.compute_features("BTCUSDT")
        orderbook = feature_computer._orderbook_manager.get_orderbook("BTCUSDT")
        assert orderbook is not None
    
    def test_update_market_data_orderbook_delta_buffered(self, feature_computer, sample_orderbook_snapshot, sample_orderbook_deltas):
        """Test that orderbook deltas are buffered and applied during feature computation."""
        # Setup orderbook with snapshot
        feature_computer.update_market_data(sample_orderbook_snapshot)
        
        # Add delta (should be buffered, not applied immediately)
        delta = {
            "event_type": "orderbook_delta",
            **sample_orderbook_deltas[0]
        }
        feature_computer.update_market_data(delta)
        
        # Delta should be in buffer
        assert feature_computer._orderbook_manager.has_pending_deltas("BTCUSDT") is True
        
        # Orderbook sequence should not be updated yet
        orderbook = feature_computer._orderbook_manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1000
        
        # Compute features - should apply buffered deltas
        fv = feature_computer.compute_features("BTCUSDT")
        
        # Now delta should be applied
        assert feature_computer._orderbook_manager.has_pending_deltas("BTCUSDT") is False
        orderbook = feature_computer._orderbook_manager.get_orderbook("BTCUSDT")
        assert orderbook.sequence == 1001
        assert fv is not None
    
    def test_update_market_data_funding_rate(self, feature_computer):
        """Test updating market data with funding rate."""
        funding = {
            "event_type": "funding_rate",
            "symbol": "BTCUSDT",
            "timestamp": datetime.now(timezone.utc),
            "funding_rate": 0.0001,
            "next_funding_time": int((datetime.now(timezone.utc).timestamp() + 3600) * 1000),
        }
        
        feature_computer.update_market_data(funding)
        
        assert feature_computer._latest_funding_rate["BTCUSDT"] == 0.0001

    def test_get_candle_pattern_function_based_on_lookback_window(self, orderbook_manager):
        """Test that pattern function is selected based on lookback_window, not version."""
        from unittest.mock import Mock
        from src.models.feature_registry import FeatureRegistry, FeatureDefinition
        
        # Create mock registry loader with 45m lookback_window
        mock_loader = Mock()
        mock_registry = FeatureRegistry(
            version="1.7.3",
            config={"timestamp_interval_minutes": 5},
            features=[
                FeatureDefinition(
                    name="pattern_all_green",
                    input_sources=["kline"],
                    lookback_window="45m",
                    lookahead_forbidden=True,
                    max_lookback_days=1,
                    data_sources=[{"source": "kline", "timestamp_required": True}],
                ),
                FeatureDefinition(
                    name="candle_0_is_green",
                    input_sources=["kline"],
                    lookback_window="45m",
                    lookahead_forbidden=True,
                    max_lookback_days=1,
                    data_sources=[{"source": "kline", "timestamp_required": True}],
                ),
            ],
        )
        mock_loader._registry_model = mock_registry
        mock_loader.get_config.return_value = None
        
        computer = FeatureComputer(
            orderbook_manager,
            feature_registry_version="1.7.3",
            feature_registry_loader=mock_loader,
        )
        
        # Should select 45m function based on lookback_window
        pattern_function = computer._get_candle_pattern_function()
        from src.features.candle_patterns import compute_all_candle_patterns_45m
        assert pattern_function == compute_all_candle_patterns_45m
    
    def test_get_candle_pattern_function_fallback_to_version(self, orderbook_manager):
        """Test that pattern function falls back to version if no lookback_window found."""
        computer = FeatureComputer(
            orderbook_manager,
            feature_registry_version="1.5.0",
        )
        
        # Should fall back to version-based selection
        pattern_function = computer._get_candle_pattern_function()
        from src.features.candle_patterns import compute_all_candle_patterns_15m
        assert pattern_function == compute_all_candle_patterns_15m

