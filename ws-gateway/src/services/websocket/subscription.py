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
    
    Note: Bybit limits args to max 10 topics per message.
    Use build_subscribe_messages() for subscriptions with more than 10 topics.
    """
    topics = sorted({s.topic for s in subscriptions if s.is_active})
    msg = {"op": "subscribe", "args": topics}
    logger.debug("build_subscribe_message", topics=topics)
    return msg


def build_subscribe_messages(subscriptions: List[Subscription], max_topics_per_message: int = 10) -> List[dict]:
    """Build multiple Bybit subscription messages, splitting into batches.
    
    Bybit limits subscription messages to max 10 topics per message.
    This function splits subscriptions into batches and returns multiple messages.
    
    Args:
        subscriptions: List of subscriptions to subscribe to
        max_topics_per_message: Maximum topics per message (default: 10, Bybit limit)
        
    Returns:
        List of subscription messages, each with at most max_topics_per_message topics
    """
    topics = sorted({s.topic for s in subscriptions if s.is_active})
    
    messages = []
    for i in range(0, len(topics), max_topics_per_message):
        batch_topics = topics[i:i + max_topics_per_message]
        msg = {"op": "subscribe", "args": batch_topics}
        messages.append(msg)
        logger.debug("build_subscribe_messages_batch", batch_index=i // max_topics_per_message, topics_count=len(batch_topics))
    
    return messages


def build_unsubscribe_message(subscriptions: List[Subscription]) -> dict:
    """Build an unsubscribe message for given subscriptions."""
    topics = sorted({s.topic for s in subscriptions})
    msg = {"op": "unsubscribe", "args": topics}
    logger.debug("build_unsubscribe_message", topics=topics)
    return msg


