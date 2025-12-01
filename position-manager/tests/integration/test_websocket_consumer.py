from decimal import Decimal

import pytest

from src.consumers.websocket_position_consumer import WebSocketPositionEvent


def test_websocket_position_event_from_message_success() -> None:
    payload = {
        "event_type": "position",
        "trace_id": "test-trace",
        "data": {
            "symbol": "BTCUSDT",
            "mode": "one-way",
            "size": "1.5",
            "avgPrice": "50000.0",
            "unrealisedPnl": "100.0",
            "realisedPnl": "10.0",
            "markPrice": "50500.0",
        },
    }

    event = WebSocketPositionEvent.from_message(payload)

    assert event.asset == "BTCUSDT"
    assert event.mode == "one-way"
    assert event.size == Decimal("1.5")
    assert event.avg_price == Decimal("50000.0")
    assert event.unrealized_pnl == Decimal("100.0")
    assert event.realized_pnl == Decimal("10.0")
    assert event.mark_price == Decimal("50500.0")
    assert event.trace_id == "test-trace"


@pytest.mark.parametrize(
    "payload",
    [
        {"event_type": "position", "data": {}},  # missing symbol
        {"event_type": "order", "data": {"symbol": "BTCUSDT"}},  # wrong event_type
    ],
)
def test_websocket_position_event_from_message_invalid(payload) -> None:
    with pytest.raises(ValueError):
        WebSocketPositionEvent.from_message(payload)


