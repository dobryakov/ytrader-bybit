"""Temporary test endpoint for subscribing to Bybit channels."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from ..services.websocket.connection import get_connection
from ..config.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class SubscribeRequest(BaseModel):
    """Subscription request model."""
    topics: List[str]


@router.post("/test/subscribe")
async def test_subscribe(request: SubscribeRequest):
    """
    Test endpoint to subscribe to Bybit WebSocket channels.
    
    Example topics:
    - Public: ["tickers.BTCUSDT", "trade.BTCUSDT", "orderbook.1.BTCUSDT"]
    - Private: ["wallet", "position", "order"]
    """
    conn = get_connection()
    
    if not conn.is_connected:
        return {"error": "WebSocket not connected", "status": conn.state.status.value}
    
    results = []
    for topic in request.topics:
        try:
            subscribe_msg = {
                "op": "subscribe",
                "args": [topic]
            }
            await conn.send(subscribe_msg)
            results.append({"topic": topic, "status": "subscribed"})
            logger.info(f"Subscribed to {topic}")
        except Exception as e:
            results.append({"topic": topic, "status": "error", "error": str(e)})
            logger.error(f"Failed to subscribe to {topic}: {e}")
    
    return {
        "status": "success",
        "subscriptions": results,
        "message": "Check logs for incoming data. Messages are logged with 'websocket_message_received'"
    }

