"""Unit tests for Subscription/Event models and subscription service helpers."""

import datetime as dt

import pytest

from src.models.event import Event
from src.models.subscription import Subscription
from src.services.subscription.subscription_service import SubscriptionService
from src.services.websocket.event_parser import parse_events_from_message


def test_subscription_create_builds_expected_fields():
    sub = Subscription.create(
        channel_type="trades",
        topic="trade.BTCUSDT",
        requesting_service="test-service",
        symbol="BTCUSDT",
    )

    assert sub.id is not None
    assert sub.channel_type == "trades"
    assert sub.topic == "trade.BTCUSDT"
    assert sub.requesting_service == "test-service"
    assert sub.is_active is True
    assert sub.symbol == "BTCUSDT"
    assert sub.created_at is not None
    assert sub.updated_at is not None


def test_event_create_sets_timestamps_and_ids():
    ts = dt.datetime.utcnow()
    event = Event.create(
        event_type="trade",
        topic="trade.BTCUSDT",
        timestamp=ts,
        payload={"symbol": "BTCUSDT"},
        trace_id="trace-123",
    )

    assert event.event_id is not None
    assert event.event_type == "trade"
    assert event.topic == "trade.BTCUSDT"
    assert event.timestamp == ts
    assert event.received_at is not None
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.trace_id == "trace-123"


@pytest.mark.parametrize(
    "channel_type, symbol, expected_topic",
    [
        ("trades", "BTCUSDT", "trade.BTCUSDT"),
        ("ticker", "BTCUSDT", "tickers.BTCUSDT"),
        ("orderbook", "BTCUSDT", "orderbook.1.BTCUSDT"),
        ("balance", None, "wallet"),
        ("order", None, "order"),
    ],
)
def test_subscription_service_build_topic(channel_type, symbol, expected_topic):
    topic = SubscriptionService._build_topic(channel_type, symbol)  # type: ignore[attr-defined]
    assert topic == expected_topic


def test_event_parser_builds_events_from_message():
    sub = Subscription.create(
        channel_type="trades",
        topic="trade.BTCUSDT",
        requesting_service="test-service",
        symbol="BTCUSDT",
    )
    message = {
        "topic": "trade.BTCUSDT",
        "ts": int(dt.datetime.utcnow().timestamp() * 1000),
        "data": [
            {
                "symbol": "BTCUSDT",
                "price": "50000.00",
                "quantity": "0.1",
                "side": "Buy",
            }
        ],
    }

    events = parse_events_from_message(
        message=message,
        subscription_lookup={sub.topic: sub},
        trace_id="trace-123",
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "trades"
    assert event.topic == "trade.BTCUSDT"
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.trace_id == "trace-123"


