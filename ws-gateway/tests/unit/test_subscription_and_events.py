"""Unit tests for Subscription/Event models and subscription service helpers."""

import datetime as dt

import pytest

from src.models.event import Event
from src.models.subscription import Subscription
from src.services.subscription.subscription_service import SubscriptionService
from src.services.websocket.event_parser import parse_events_from_message
from src.services.positions.position_event_normalizer import PositionEventNormalizer


def test_subscription_create_builds_expected_fields():
    sub = Subscription.create(
        channel_type="trades",
        topic="publicTrade.BTCUSDT",  # Bybit v5 uses publicTrade format
        requesting_service="test-service",
        symbol="BTCUSDT",
    )

    assert sub.id is not None
    assert sub.channel_type == "trades"
    assert sub.topic == "publicTrade.BTCUSDT"
    assert sub.requesting_service == "test-service"
    assert sub.is_active is True
    assert sub.symbol == "BTCUSDT"
    assert sub.created_at is not None
    assert sub.updated_at is not None


def test_event_create_sets_timestamps_and_ids():
    ts = dt.datetime.utcnow()
    event = Event.create(
        event_type="trade",
        topic="publicTrade.BTCUSDT",  # Bybit v5 uses publicTrade format
        timestamp=ts,
        payload={"symbol": "BTCUSDT"},
        trace_id="trace-123",
    )

    assert event.event_id is not None
    assert event.event_type == "trade"
    assert event.topic == "publicTrade.BTCUSDT"
    assert event.timestamp == ts
    assert event.received_at is not None
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.trace_id == "trace-123"


@pytest.mark.parametrize(
    "channel_type, symbol, expected_topic",
    [
        ("trades", "BTCUSDT", "publicTrade.BTCUSDT"),  # Bybit v5 uses publicTrade format for spot
        ("ticker", "BTCUSDT", "tickers.BTCUSDT"),
        ("orderbook", "BTCUSDT", "orderbook.1.BTCUSDT"),
        ("balance", None, "wallet"),
        ("order", None, "order"),
        ("funding", "BTCUSDT", "fundingRate.BTCUSDT"),
    ],
)
def test_subscription_service_build_topic(channel_type, symbol, expected_topic):
    topic = SubscriptionService._build_topic(channel_type, symbol)  # type: ignore[attr-defined]
    assert topic == expected_topic


def test_event_parser_builds_events_from_message():
    sub = Subscription.create(
        channel_type="trades",
        topic="publicTrade.BTCUSDT",  # Bybit v5 uses publicTrade format
        requesting_service="test-service",
        symbol="BTCUSDT",
    )
    message = {
        "topic": "publicTrade.BTCUSDT",  # Bybit v5 uses publicTrade format
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
    assert event.event_type == "trade"  # Normalized from "trades" to match EventType
    assert event.topic == "publicTrade.BTCUSDT"
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.trace_id == "trace-123"


def test_event_parser_builds_position_event_with_full_payload():
    sub = Subscription.create(
        channel_type="position",
        topic="position",
        requesting_service="test-service",
        symbol=None,
    )
    ts_ms = int(dt.datetime.utcnow().timestamp() * 1000)
    message = {
        "topic": "position",
        "ts": ts_ms,
        "data": [
            {
                "symbol": "BTCUSDT",
                "size": "0.01",
                "side": "Buy",
                "avgPrice": "50000.0",
                "unrealisedPnl": "10.0",
                "cumRealisedPnl": "-2.0",
                "positionIdx": 1,
            }
        ],
    }

    events = parse_events_from_message(
        message=message,
        subscription_lookup={sub.topic: sub},
        trace_id="trace-pos-1",
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "position"
    assert event.topic == "position"
    # Timestamp must be parsed from ts and propagated into the Event
    expected_ts = dt.datetime.utcfromtimestamp(ts_ms / 1000.0)
    # allow for small rounding differences
    assert abs((event.timestamp - expected_ts).total_seconds()) < 1
    # Payload should preserve full data array for the normalizer
    assert "data" in event.payload
    assert isinstance(event.payload["data"], list)
    assert event.payload["data"][0]["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_position_event_normalizer_normalizes_and_publishes(monkeypatch):
    # Build a synthetic position event
    ts = dt.datetime.utcnow()
    event = Event.create(
        event_type="position",
        topic="position",
        timestamp=ts,
        payload={
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "size": "0.01",
                    "side": "Buy",
                    "avgPrice": "50000.0",
                    "unrealisedPnl": "10.0",
                    "cumRealisedPnl": "-2.0",
                    "positionIdx": 1,
                }
            ]
        },
        trace_id="trace-pos-2",
    )

    published_events = []

    class DummyPublisher:
        async def publish_event(self, event_arg, queue_name):
            published_events.append((event_arg, queue_name))
            return True

    async def dummy_get_publisher():
        return DummyPublisher()

    monkeypatch.setattr(
        "src.services.positions.position_event_normalizer.get_publisher",
        dummy_get_publisher,
    )

    success = await PositionEventNormalizer.normalize_and_publish(event)

    assert success is True
    assert len(published_events) == 1
    normalized_event, queue_name = published_events[0]
    assert queue_name == "ws-gateway.position"
    assert normalized_event.event_type == "position"
    assert normalized_event.payload["symbol"] == "BTCUSDT"
    assert normalized_event.payload["size"] == "0.01"
    assert normalized_event.payload["mode"] == 1
    # Timestamp must be available for conflict resolution
    assert normalized_event.payload["timestamp"] == event.timestamp.isoformat()


def test_event_parser_builds_funding_event_from_message():
    """Test that funding rate events are parsed correctly from Bybit messages."""
    sub = Subscription.create(
        channel_type="funding",
        topic="fundingRate.BTCUSDT",
        requesting_service="test-service",
        symbol="BTCUSDT",
    )
    ts_ms = int(dt.datetime.utcnow().timestamp() * 1000)
    message = {
        "topic": "fundingRate.BTCUSDT",
        "ts": ts_ms,
        "data": [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingRateTimestamp": str(ts_ms),
                "nextFundingTime": str(ts_ms + 3600000),  # 1 hour later
            }
        ],
    }

    events = parse_events_from_message(
        message=message,
        subscription_lookup={sub.topic: sub},
        trace_id="trace-funding-1",
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "funding"
    assert event.topic == "fundingRate.BTCUSDT"
    # Payload should contain funding rate data
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.payload["fundingRate"] == "0.0001"
    assert event.payload["nextFundingTime"] == str(ts_ms + 3600000)
    assert event.trace_id == "trace-funding-1"


@pytest.mark.asyncio
async def test_funding_event_processing(monkeypatch):
    """Test that funding rate events are processed and published correctly."""
    ts = dt.datetime.utcnow()
    event = Event.create(
        event_type="funding",
        topic="fundingRate.BTCUSDT",
        timestamp=ts,
        payload={
            "symbol": "BTCUSDT",
            "fundingRate": "0.0001",
            "fundingRateTimestamp": str(int(ts.timestamp() * 1000)),
            "nextFundingTime": str(int(ts.timestamp() * 1000) + 3600000),
        },
        trace_id="trace-funding-2",
    )

    published_events = []

    class DummyPublisher:
        async def publish_event(self, event_arg, queue_name):
            published_events.append((event_arg, queue_name))
            return True

    async def dummy_get_publisher():
        return DummyPublisher()

    monkeypatch.setattr(
        "src.services.websocket.event_processor.get_publisher",
        dummy_get_publisher,
    )

    from src.services.websocket.event_processor import process_event

    await process_event(event)

    assert len(published_events) == 1
    published_event, queue_name = published_events[0]
    assert queue_name == "ws-gateway.funding"
    assert published_event.event_type == "funding"
    assert published_event.payload["symbol"] == "BTCUSDT"
    assert published_event.payload["fundingRate"] == "0.0001"
    assert "nextFundingTime" in published_event.payload


