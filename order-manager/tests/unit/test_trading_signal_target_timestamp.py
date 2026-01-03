"""Unit tests for TradingSignal.get_target_timestamp() method."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from src.models.trading_signal import TradingSignal, MarketDataSnapshot


@pytest.fixture
def base_signal():
    """Create base signal for testing."""
    return TradingSignal(
        signal_id=uuid4(),
        signal_type="buy",
        asset="BTCUSDT",
        amount=Decimal("1000.0"),
        confidence=Decimal("0.75"),
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
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


class TestGetTargetTimestamp:
    """Test get_target_timestamp() method."""

    def test_get_target_timestamp_with_direct_timestamp_string(self, base_signal):
        """Test extracting target_timestamp directly from metadata as ISO string."""
        target_ts = datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        base_signal.metadata = {
            "target_timestamp": target_ts.isoformat(),
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        assert result == target_ts

    def test_get_target_timestamp_with_direct_timestamp_datetime(self, base_signal):
        """Test extracting target_timestamp directly from metadata as datetime object."""
        target_ts = datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        base_signal.metadata = {
            "target_timestamp": target_ts,
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        assert result == target_ts

    def test_get_target_timestamp_with_horizon_seconds(self, base_signal):
        """Test computing target_timestamp from prediction_horizon_seconds."""
        base_signal.metadata = {
            "prediction_horizon_seconds": 180,  # 3 minutes
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        expected = base_signal.timestamp + timedelta(seconds=180)
        assert result == expected

    def test_get_target_timestamp_with_horizon_seconds_string(self, base_signal):
        """Test computing target_timestamp from prediction_horizon_seconds as string."""
        base_signal.metadata = {
            "prediction_horizon_seconds": "180",  # String format
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        expected = base_signal.timestamp + timedelta(seconds=180)
        assert result == expected

    def test_get_target_timestamp_with_no_metadata(self, base_signal):
        """Test that get_target_timestamp returns None when metadata is None."""
        base_signal.metadata = None
        
        result = base_signal.get_target_timestamp()
        
        assert result is None

    def test_get_target_timestamp_with_empty_metadata(self, base_signal):
        """Test that get_target_timestamp returns None when metadata is empty."""
        base_signal.metadata = {}
        
        result = base_signal.get_target_timestamp()
        
        assert result is None

    def test_get_target_timestamp_prefers_direct_over_horizon(self, base_signal):
        """Test that direct target_timestamp is preferred over prediction_horizon_seconds."""
        target_ts = datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
        base_signal.metadata = {
            "target_timestamp": target_ts.isoformat(),
            "prediction_horizon_seconds": 180,
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        assert result == target_ts
        # Verify it's not computed from horizon
        expected_from_horizon = base_signal.timestamp + timedelta(seconds=180)
        assert result != expected_from_horizon

    def test_get_target_timestamp_with_invalid_timestamp_string(self, base_signal):
        """Test that invalid timestamp string falls back to horizon calculation."""
        base_signal.metadata = {
            "target_timestamp": "invalid-date-string",
            "prediction_horizon_seconds": 180,
        }
        
        result = base_signal.get_target_timestamp()
        
        # Should fall back to horizon calculation
        assert result is not None
        expected = base_signal.timestamp + timedelta(seconds=180)
        assert result == expected

    def test_get_target_timestamp_with_invalid_horizon(self, base_signal):
        """Test that invalid horizon returns None."""
        base_signal.metadata = {
            "prediction_horizon_seconds": "invalid",
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is None

    def test_get_target_timestamp_with_zero_horizon(self, base_signal):
        """Test that zero horizon returns None."""
        base_signal.metadata = {
            "prediction_horizon_seconds": 0,
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is None

    def test_get_target_timestamp_with_negative_horizon(self, base_signal):
        """Test that negative horizon returns None."""
        base_signal.metadata = {
            "prediction_horizon_seconds": -10,
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is None

    def test_get_target_timestamp_with_horizon_int_type(self, base_signal):
        """Test computing target_timestamp from prediction_horizon_seconds as int."""
        base_signal.metadata = {
            "prediction_horizon_seconds": 900,  # Int type (not string)
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        expected = base_signal.timestamp + timedelta(seconds=900)
        assert result == expected

    def test_get_target_timestamp_with_naive_timestamp(self, base_signal):
        """Test computing target_timestamp when signal timestamp is timezone-naive."""
        # Create signal with naive timestamp
        base_signal.timestamp = datetime(2025, 1, 1, 12, 0, 0)  # No timezone
        base_signal.metadata = {
            "prediction_horizon_seconds": 180,
        }
        
        result = base_signal.get_target_timestamp()
        
        # Should still work, assuming UTC for naive timestamps
        assert result is not None
        # Result should be timezone-aware UTC
        assert result.tzinfo is not None

    def test_get_target_timestamp_with_target_timestamp_z_suffix(self, base_signal):
        """Test parsing target_timestamp with Z suffix (as it comes from model-service)."""
        target_ts_str = "2025-12-27T17:44:34.977982Z"
        base_signal.metadata = {
            "target_timestamp": target_ts_str,
        }
        
        result = base_signal.get_target_timestamp()
        
        assert result is not None
        # Should parse correctly despite Z suffix
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 27
        assert result.hour == 17
        assert result.minute == 44

