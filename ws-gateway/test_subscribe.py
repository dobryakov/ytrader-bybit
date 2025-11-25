"""Test script to subscribe to Bybit WebSocket channels and see what data is available."""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.websocket.connection import get_connection
from src.config.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def test_subscriptions():
    """Test subscribing to various Bybit channels."""
    conn = get_connection()
    
    # Wait for connection
    if not conn.is_connected:
        logger.info("Waiting for connection...")
        await conn.wait_connected(timeout=10.0)
    
    if not conn.is_connected:
        logger.error("Failed to connect to WebSocket")
        return
    
    logger.info("Connected! Testing subscriptions...")
    
    # Test subscriptions for public data (available on testnet)
    test_subscriptions = [
        # Public ticker data
        {"op": "subscribe", "args": ["tickers.BTCUSDT"]},
        # Public trades
        {"op": "subscribe", "args": ["trade.BTCUSDT"]},
        # Public orderbook
        {"op": "subscribe", "args": ["orderbook.1.BTCUSDT"]},
        # Private: wallet balance (requires auth, which we have)
        {"op": "subscribe", "args": ["wallet"]},
        # Private: position updates
        {"op": "subscribe", "args": ["position"]},
        # Private: order updates
        {"op": "subscribe", "args": ["order"]},
    ]
    
    for sub_msg in test_subscriptions:
        try:
            logger.info(f"Sending subscription: {sub_msg['args']}")
            await conn.send(sub_msg)
            await asyncio.sleep(0.5)  # Small delay between subscriptions
        except Exception as e:
            logger.error(f"Failed to subscribe to {sub_msg['args']}: {e}")
    
    logger.info("Subscriptions sent! Waiting for data (10 seconds)...")
    logger.info("Check the logs to see incoming messages")
    
    # Wait a bit to receive some data
    await asyncio.sleep(10)
    
    logger.info("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_subscriptions())

