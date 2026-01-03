"""Unit tests for Order model with target_timestamp field."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from uuid import uuid4

from src.models.order import Order


@pytest.fixture
def order_data():
    """Create order data dictionary for testing."""
    return {
        "id": uuid4(),
        "order_id": "test-order-123",
        "signal_id": uuid4(),
        "asset": "BTCUSDT",
        "side": "Buy",
        "order_type": "Market",
        "quantity": Decimal("0.01"),
        "price": None,
        "status": "pending",
        "filled_quantity": Decimal("0"),
        "average_price": None,
        "fees": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "executed_at": None,
        "trace_id": "test-trace",
        "is_dry_run": False,
        "rejection_reason": None,
        "target_timestamp": None,
    }


class TestOrderTargetTimestamp:
    """Test Order model with target_timestamp field."""

    def test_order_without_target_timestamp(self, order_data):
        """Test creating order without target_timestamp."""
        order_data.pop("target_timestamp", None)
        order = Order.from_dict(order_data)
        
        assert order.target_timestamp is None

    def test_order_with_target_timestamp(self, order_data):
        """Test creating order with target_timestamp."""
        target_ts = datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        order_data["target_timestamp"] = target_ts
        
        order = Order.from_dict(order_data)
        
        assert order.target_timestamp == target_ts

    def test_order_with_target_timestamp_string(self, order_data):
        """Test creating order with target_timestamp as ISO string."""
        target_ts = datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        order_data["target_timestamp"] = target_ts.isoformat()
        
        order = Order.from_dict(order_data)
        
        assert order.target_timestamp is not None
        # Compare as timestamps (may have minor timezone differences)
        assert order.target_timestamp.timestamp() == pytest.approx(target_ts.timestamp(), abs=1)

    def test_order_to_dict_with_target_timestamp(self, order_data):
        """Test converting order with target_timestamp to dictionary."""
        target_ts = datetime(2025, 1, 1, 12, 3, 0, tzinfo=timezone.utc)
        order_data["target_timestamp"] = target_ts
        order = Order.from_dict(order_data)
        
        order_dict = order.to_dict()
        
        assert "target_timestamp" in order_dict
        assert order_dict["target_timestamp"] == target_ts

    def test_order_to_dict_without_target_timestamp(self, order_data):
        """Test converting order without target_timestamp to dictionary."""
        order_data["target_timestamp"] = None
        order = Order.from_dict(order_data)
        
        order_dict = order.to_dict()
        
        assert "target_timestamp" in order_dict
        assert order_dict["target_timestamp"] is None

