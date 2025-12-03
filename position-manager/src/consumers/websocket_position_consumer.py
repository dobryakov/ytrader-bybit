"""WebSocket position event consumer.

Consumes position events from `ws-gateway.position` queue and delegates
processing to PositionManager.update_position_from_websocket().
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Optional

from aio_pika import IncomingMessage

from ..config.logging import get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..services.position_manager import PositionManager
from ..utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


@dataclass
class WebSocketPositionEvent:
    """Typed view over WS position event payload."""

    asset: str
    mode: str
    size: Optional[Decimal]
    avg_price: Optional[Decimal]
    unrealized_pnl: Optional[Decimal]
    realized_pnl: Optional[Decimal]
    mark_price: Optional[Decimal]
    trace_id: Optional[str]
    event_timestamp: Optional[datetime]

    @classmethod
    def from_message(cls, payload: Dict[str, Any]) -> "WebSocketPositionEvent":
        event_type = payload.get("event_type")
        if event_type != "position":
            raise ValueError(f"Unsupported event_type: {event_type}")

        trace_id = payload.get("trace_id") or get_or_create_trace_id()
        # ws-gateway sends events with "payload" field, but we also support "data" for backward compatibility
        data = payload.get("payload") or payload.get("data") or {}

        # Log for debugging
        logger.debug(
            "ws_position_event_parsing",
            event_type=event_type,
            has_payload_field="payload" in payload,
            has_data_field="data" in payload,
            data_keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
            trace_id=trace_id,
        )

        symbol = data.get("symbol")
        if not symbol:
            logger.error(
                "ws_position_event_missing_symbol",
                payload_keys=list(payload.keys()),
                data_keys=list(data.keys()) if isinstance(data, dict) else type(data).__name__,
                data_preview=str(data)[:500] if data else None,
                trace_id=trace_id,
            )
            raise ValueError("Missing 'symbol' in WebSocket position event")

        mode = data.get("mode") or "one-way"

        def to_decimal(val: Any) -> Optional[Decimal]:
            if val is None:
                return None
            try:
                return Decimal(str(val))
            except Exception:
                return None

        # Timestamp from normalized payload (Phase 9: used for conflict resolution)
        ts_raw = data.get("timestamp")
        event_ts: Optional[datetime] = None
        if isinstance(ts_raw, str):
            try:
                # Support both naive and 'Z'-suffixed ISO8601 strings
                if ts_raw.endswith("Z"):
                    event_ts = datetime.fromisoformat(ts_raw[:-1])
                else:
                    event_ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                event_ts = None

        return cls(
            asset=symbol,
            mode=mode,
            size=to_decimal(data.get("size")),
            avg_price=to_decimal(data.get("avgPrice")),
            unrealized_pnl=to_decimal(data.get("unrealisedPnl")),
            realized_pnl=to_decimal(data.get("realisedPnl")),
            mark_price=to_decimal(data.get("markPrice")),
            trace_id=trace_id,
            event_timestamp=event_ts,
        )


class WebSocketPositionConsumer:
    """Async consumer for `ws-gateway.position` queue."""

    def __init__(self, position_manager: Optional[PositionManager] = None) -> None:
        self._position_manager = position_manager or PositionManager()
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def _handle_message(self, message: IncomingMessage) -> None:
        trace_id = None
        try:
            payload = json.loads(message.body.decode("utf-8"))
            event = WebSocketPositionEvent.from_message(payload)
            trace_id = event.trace_id

            logger.info(
                "ws_position_event_received",
                asset=event.asset,
                mode=event.mode,
                trace_id=trace_id,
            )

            await self._position_manager.update_position_from_websocket(
                asset=event.asset,
                mode=event.mode,
                mark_price=event.mark_price,
                avg_price=event.avg_price,
                size_from_ws=event.size,
                unrealized_pnl=event.unrealized_pnl,
                realized_pnl=event.realized_pnl,
                trace_id=trace_id,
                event_timestamp=event.event_timestamp,
            )

            await message.ack()
            logger.debug("ws_position_event_acked", trace_id=trace_id)
        except (ValueError, KeyError) as e:
            # Permanent error: reject without requeue
            logger.error(
                "ws_position_event_validation_failed",
                error=str(e),
                trace_id=trace_id,
            )
            await message.reject(requeue=False)
        except Exception as e:
            # Transient error: nack with requeue
            logger.error(
                "ws_position_event_processing_failed",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            await message.nack(requeue=True)

    async def start(self) -> None:
        """Start consuming from ws-gateway.position."""
        channel = await RabbitMQConnection.get_channel()
        queue = await channel.declare_queue("ws-gateway.position", durable=True)

        logger.info("ws_position_consumer_started", queue=queue.name)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if self._stopped.is_set():
                    break
                await self._handle_message(message)

        logger.info("ws_position_consumer_stopped")

    async def run_forever(self) -> None:
        """Run the consumer until stop() is called."""
        try:
            await self.start()
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            logger.info("ws_position_consumer_cancelled")
        except Exception as e:  # pragma: no cover
            logger.error("ws_position_consumer_crashed", error=str(e), exc_info=True)

    def spawn(self) -> None:
        """Spawn the consumer in a background task."""
        if self._task is None or self._task.done():
            self._stopped.clear()
            self._task = asyncio.create_task(self.run_forever())

    async def stop(self) -> None:
        """Signal the consumer loop to stop."""
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # type: ignore[name-defined]
                await self._task


