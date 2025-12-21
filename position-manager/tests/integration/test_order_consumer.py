from decimal import Decimal

import pytest

from src.consumers.position_order_linker_consumer import OrderExecutionEvent


def test_order_execution_event_from_message_buy() -> None:
    payload = {
        "event_type": "filled",
        "order": {
            "order_id": "bybit-order-123",
            "signal_id": "signal-456",
            "asset": "BTCUSDT",
            "side": "Buy",
            "filled_quantity": "2.0",
            "average_price": "30000.0",
            "fees": "1.5",
        },
        "mode": "one-way",
        "trace_id": "test-trace",
    }

    event = OrderExecutionEvent.from_message(payload)

    assert event.order_id == "bybit-order-123"
    assert event.signal_id == "signal-456"
    assert event.asset == "BTCUSDT"
    assert event.mode == "one-way"
    assert event.side == "buy"
    assert event.filled_quantity == Decimal("2.0")
    assert event.execution_price == Decimal("30000.0")
    assert event.execution_fees == Decimal("1.5")
    assert event.trace_id == "test-trace"


def test_order_execution_event_from_message_sell() -> None:
    payload = {
        "event_type": "filled",
        "order": {
            "order_id": "bybit-order-456",
            "asset": "BTCUSDT",
            "side": "Sell",
            "filled_quantity": "2.0",
            "average_price": "30000.0",
        },
        "mode": "one-way",
    }

    event = OrderExecutionEvent.from_message(payload)

    assert event.side == "sell"
    assert event.filled_quantity == Decimal("2.0")


@pytest.mark.parametrize(
    "payload",
    [
        {"event_type": "unknown", "asset": "BTCUSDT"},  # unsupported type
        {"event_type": "filled"},  # missing order_id
        {"event_type": "filled", "order": {"order_id": "123"}},  # missing asset
        {"event_type": "filled", "order": {"order_id": "123", "asset": "BTCUSDT"}},  # missing qty/price
    ],
)
def test_order_execution_event_from_message_invalid(payload) -> None:
    with pytest.raises(ValueError):
        OrderExecutionEvent.from_message(payload)


