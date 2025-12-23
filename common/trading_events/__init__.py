"""
Common utilities for publishing trading/business events to RabbitMQ exchange
`trading_events`, which are then forwarded to Graylog via trading-events-forwarder.
"""

from .publisher import TradingEventsPublisher, trading_events_publisher  # noqa: F401


