"""Position event normalization and publishing to RabbitMQ.

This module implements Phase 7.5 (Position Channel Support) tasks:

- Parse raw Bybit position events from Event.payload
- Validate key fields (symbol, size, mode) and drop invalid events
- Normalize payload for Position Manager consumption
- Publish normalized events to the ``ws-gateway.position`` queue
- Handle publish failures gracefully (per FR-017)
- Add structured logging with trace_id, asset, mode, and source_channel
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List

from ...config.logging import get_logger
from ...exceptions import QueueError
from ...models.event import Event
from ..queue.publisher import get_publisher
from ..queue.router import get_queue_name_for_event_type

logger = get_logger(__name__)


class PositionEventNormalizer:
    """Normalize Bybit position events and publish them to RabbitMQ.

    Notes:
        - This service does NOT write directly to the positions table.
          Position persistence is owned by the position-manager service.
        - Normalized events are published to the ``ws-gateway.position`` queue
          for consumption by position-manager.
    """

    @staticmethod
    def _parse_positions_from_event(event: Event) -> List[Dict[str, Any]]:
        """Extract raw position entries from an Event payload.

        Bybit v5 private position stream typically uses:
            {
              "topic": "position",
              "data": [
                {
                  "symbol": "BTCUSDT",
                  "size": "0.01",
                  "side": "Buy",
                  "avgPrice": "50000.0",
                  "unrealisedPnl": "10.0",
                  "cumRealisedPnl": "-2.0",
                  "positionIdx": 1,
                  ...
                }
              ]
            }

        The parser is tolerant and will treat any dict inside ``payload["data"]``
        as a position record.
        """
        payload = event.payload or {}
        data = payload.get("data")

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        if isinstance(data, dict):
            # Some variants may send a single object
            return [data]

        logger.warning(
            "position_event_no_data",
            event_id=str(event.event_id),
            trace_id=event.trace_id,
            payload_type=type(payload).__name__,
        )
        return []

    @staticmethod
    def _normalize_single_position(
        raw: Dict[str, Any],
        event: Event,
    ) -> Dict[str, Any] | None:
        """Normalize a single raw position dict.

        Required normalized fields:
            - symbol: trading pair symbol
            - size: Decimal, absolute non-negative position size
            - side: "Buy" / "Sell" or equivalent
            - avg_price: average entry price
            - unrealised_pnl: unrealised PnL
            - realised_pnl: realised PnL (if available)
            - mode: position mode / index for conflict resolution
            - timestamp: ISO8601 from Event.timestamp
            - source_channel: original Bybit topic

        Invalid positions (missing symbol, invalid size/mode) are logged
        and dropped (log-and-drop).
        """
        symbol = raw.get("symbol") or raw.get("s")
        if not symbol or not isinstance(symbol, str):
            logger.warning(
                "position_normalize_missing_symbol",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                raw_keys=list(raw.keys()),
            )
            return None

        # Size parsing and validation (non-negative)
        size_raw = raw.get("size") or raw.get("qty") or raw.get("positionQty")
        try:
            size_dec = Decimal(str(size_raw)) if size_raw is not None else Decimal(0)
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(
                "position_normalize_invalid_size",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                symbol=symbol,
                size=size_raw,
            )
            return None

        if size_dec < 0:
            # Normalize to absolute size; direction is encoded in side
            size_dec = abs(size_dec)

        # Side and mode
        side = raw.get("side") or raw.get("positionSide") or raw.get("sSide")
        mode_raw = raw.get("positionIdx") or raw.get("positionMode") or raw.get("mode")

        # Basic mode validation: allow known numeric indices or non-empty strings
        valid_mode = False
        if isinstance(mode_raw, int):
            valid_mode = mode_raw in (0, 1, 2, 3)
        elif isinstance(mode_raw, str):
            valid_mode = mode_raw.strip() != ""

        if not valid_mode:
            logger.warning(
                "position_normalize_invalid_mode",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                symbol=symbol,
                mode=mode_raw,
            )
            return None

        # PnL fields (optional)
        unrealised_raw = raw.get("unrealisedPnl") or raw.get("unrealisedPnlUsd")
        realised_raw = raw.get("cumRealisedPnl") or raw.get("realisedPnl")

        def _to_decimal(value: Any) -> Decimal | None:
            if value is None or value == "":
                return None
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError, TypeError):
                return None

        unrealised_pnl = _to_decimal(unrealised_raw)
        realised_pnl = _to_decimal(realised_raw)

        normalized: Dict[str, Any] = {
            "symbol": symbol,
            "size": str(size_dec),
            "side": side,
            "avg_price": raw.get("avgPrice") or raw.get("avgEntryPrice"),
            "unrealised_pnl": str(unrealised_pnl) if unrealised_pnl is not None else None,
            "realised_pnl": str(realised_pnl) if realised_pnl is not None else None,
            "mode": mode_raw,
            "timestamp": event.timestamp.isoformat(),
            "source_channel": event.topic,
            "trace_id": event.trace_id,
        }

        # Include full raw record for advanced consumers that need extra fields
        normalized["raw"] = raw

        return normalized

    @staticmethod
    async def normalize_and_publish(event: Event) -> bool:
        """Normalize and publish a position Event to RabbitMQ.

        Returns:
            True if at least one normalized position event was published, False otherwise.

        Behavior:
            - If event.event_type != "position", it is ignored and returns False.
            - Any parsing/validation errors are logged and do not raise.
            - Queue publish failures are logged and do not raise (per FR-017).
        """
        if event.event_type != "position":
            logger.debug(
                "position_normalizer_skipped_non_position_event",
                event_id=str(event.event_id),
                event_type=event.event_type,
                trace_id=event.trace_id,
            )
            return False

        raw_positions = PositionEventNormalizer._parse_positions_from_event(event)
        if not raw_positions:
            logger.warning(
                "position_normalizer_no_positions",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
            )
            return False

        normalized_positions: List[Dict[str, Any]] = []
        for raw in raw_positions:
            normalized = PositionEventNormalizer._normalize_single_position(raw, event)
            if normalized is not None:
                normalized_positions.append(normalized)

        if not normalized_positions:
            logger.warning(
                "position_normalizer_all_positions_invalid",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
            )
            return False

        # Publish each normalized position as a separate message to ws-gateway.position
        queue_name = get_queue_name_for_event_type("position")
        publisher = await get_publisher()

        success_count = 0

        for pos in normalized_positions:
            try:
                # Build a derived Event with the normalized payload while preserving
                # event_id, timestamp and trace_id for conflict resolution.
                normalized_event = Event(
                    event_id=event.event_id,
                    event_type="position",
                    topic=queue_name.replace("ws-gateway.", "position."),
                    timestamp=event.timestamp,
                    received_at=event.received_at,
                    payload=pos,
                    trace_id=event.trace_id,
                )

                success = await publisher.publish_event(normalized_event, queue_name)
                if success:
                    success_count += 1
                    logger.info(
                        "position_event_published",
                        event_id=str(event.event_id),
                        symbol=pos.get("symbol"),
                        mode=pos.get("mode"),
                        queue_name=queue_name,
                        source_channel=event.topic,
                        trace_id=event.trace_id,
                    )
                else:
                    logger.warning(
                        "position_event_publish_failed",
                        event_id=str(event.event_id),
                        symbol=pos.get("symbol"),
                        mode=pos.get("mode"),
                        queue_name=queue_name,
                        source_channel=event.topic,
                        trace_id=event.trace_id,
                    )
            except QueueError as qe:
                logger.error(
                    "position_event_queue_error",
                    event_id=str(event.event_id),
                    queue_name=queue_name,
                    error=str(qe),
                    error_type=type(qe).__name__,
                    trace_id=event.trace_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "position_event_publish_exception",
                    event_id=str(event.event_id),
                    queue_name=queue_name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    trace_id=event.trace_id,
                )
                # Continue with next position record

        return success_count > 0


