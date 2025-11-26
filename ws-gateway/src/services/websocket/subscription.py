"""Bybit WebSocket subscription message formatting."""

from __future__ import annotations

from typing import List

from ...config.logging import get_logger
from ...models.subscription import Subscription

logger = get_logger(__name__)


def build_subscribe_message(subscriptions: List[Subscription]) -> dict:
    """Build a Bybit subscription message for given subscriptions.

    Bybit WebSocket API expects messages in the form:
    {
        "op": "subscribe",
        "args": ["topic1", "topic2", ...]
    }
    """
    topics = sorted({s.topic for s in subscriptions if s.is_active})
    msg = {"op": "subscribe", "args": topics}
    logger.debug("build_subscribe_message", topics=topics)
    return msg


def build_unsubscribe_message(subscriptions: List[Subscription]) -> dict:
    """Build an unsubscribe message for given subscriptions."""
    topics = sorted({s.topic for s in subscriptions})
    msg = {"op": "unsubscribe", "args": topics}
    logger.debug("build_unsubscribe_message", topics=topics)
    return msg


