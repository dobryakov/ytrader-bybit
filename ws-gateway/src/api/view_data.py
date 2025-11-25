"""Endpoint to view received WebSocket data in real-time."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import List
from collections import deque

from ..services.websocket.connection import get_connection, get_recent_messages
from ..config.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/test/data")
async def get_recent_data():
    """
    Get recent WebSocket messages received.
    Useful for checking what data is coming from Bybit.
    """
    messages = get_recent_messages()
    return {
        "count": len(messages),
        "messages": messages
    }


@router.get("/test/data/stream")
async def stream_data():
    """
    Stream WebSocket messages in real-time (SSE).
    """
    async def event_generator():
        last_count = 0
        while True:
            messages = get_recent_messages()
            current_count = len(messages)
            if current_count > last_count:
                # New messages arrived
                new_messages = messages[last_count:]
                for msg in new_messages:
                    yield f"data: {json.dumps(msg)}\n\n"
                last_count = current_count
            await asyncio.sleep(0.5)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

