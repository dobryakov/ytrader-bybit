from decimal import Decimal

import pytest

from src.consumers.order_position_consumer import OrderExecutionEvent


def test_order_execution_event_from_message_buy() -> None:
    payload = {
        "event_type": "order_executed",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "side": "buy",
        "filled_quantity": "2.0",
        "execution_price": "30000.0",
        "execution_fees": "1.5",
        "trace_id": "test-trace",
    }

    event = OrderExecutionEvent.from_message(payload)

    assert event.asset == "BTCUSDT"
    assert event.mode == "one-way"
    assert event.size_delta == Decimal("2.0")
    assert event.execution_price == Decimal("30000.0")
    assert event.execution_fees == Decimal("1.5")
    assert event.trace_id == "test-trace"


def test_order_execution_event_from_message_sell() -> None:
    payload = {
        "event_type": "order_executed",
        "asset": "BTCUSDT",
        "mode": "one-way",
        "side": "sell",
        "filled_quantity": "2.0",
        "execution_price": "30000.0",
    }

    event = OrderExecutionEvent.from_message(payload)

    # Sell should produce negative size_delta
    assert event.size_delta == Decimal("-2.0")


@pytest.mark.parametrize(
    "payload",
    [
        {"event_type": "unknown", "asset": "BTCUSDT"},  # unsupported type
        {"event_type": "order_executed"},  # missing asset
        {"event_type": "order_executed", "asset": "BTCUSDT"},  # missing qty/price
    ],
)
def test_order_execution_event_from_message_invalid(payload) -> None:
    with pytest.raises(ValueError):
        OrderExecutionEvent.from_message(payload)


