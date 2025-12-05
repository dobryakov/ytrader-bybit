"""
Market data consumer for consuming market data from RabbitMQ queues.
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
import structlog
import aio_pika
from aio_pika import IncomingMessage

from src.mq.connection import MQConnectionManager
from src.http.client import HTTPClient
from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager
from src.services.data_storage import DataStorageService
from src.models.market_data import (
    OrderbookSnapshot,
    OrderbookDelta,
    Trade,
    Kline,
    Ticker,
    FundingRate,
)

logger = structlog.get_logger(__name__)


class MarketDataConsumer:
    """Consumes market data from RabbitMQ queues and updates feature computation state."""
    
    def __init__(
        self,
        mq_manager: MQConnectionManager,
        http_client: HTTPClient,
        feature_computer: FeatureComputer,
        orderbook_manager: OrderbookManager,
        data_storage: Optional[DataStorageService] = None,
        service_name: str = "feature-service",
        symbols: Optional[List[str]] = None,
    ):
        """Initialize market data consumer."""
        self._mq_manager = mq_manager
        self._http_client = http_client
        self._feature_computer = feature_computer
        self._orderbook_manager = orderbook_manager
        self._data_storage = data_storage
        self._service_name = service_name
        self._symbols = symbols or []
        self._subscriptions: Set[str] = set()
        self._consumers: List[asyncio.Task] = []
        self._running = False
    
    async def start(self) -> None:
        """Start consuming market data."""
        logger.info("Starting market data consumer", service_name=self._service_name)
        
        # Create subscriptions via ws-gateway REST API
        await self._create_subscriptions()
        
        # Start consuming from queues
        self._running = True
        await self._start_consumers()
        
        logger.info("Market data consumer started")
    
    async def stop(self) -> None:
        """Stop consuming market data."""
        logger.info("Stopping market data consumer")
        
        self._running = False
        
        # Cancel all consumer tasks
        for task in self._consumers:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._consumers, return_exceptions=True)
        
        logger.info("Market data consumer stopped")
    
    async def _create_subscriptions(self) -> None:
        """Create subscriptions via ws-gateway REST API."""
        channels = [
            {"channel_type": "orderbook", "symbol": symbol}
            for symbol in self._symbols
        ] + [
            {"channel_type": "trades", "symbol": symbol}
            for symbol in self._symbols
        ] + [
            {"channel_type": "ticker", "symbol": symbol}
            for symbol in self._symbols
        ] + [
            {"channel_type": "kline", "symbol": symbol}
            for symbol in self._symbols
        ] + [
            {"channel_type": "funding", "symbol": symbol}
            for symbol in self._symbols
        ]
        
        for channel_config in channels:
            try:
                response = await self._http_client.post(
                    "/api/v1/subscriptions",
                    json={
                        **channel_config,
                        "requesting_service": self._service_name,
                    },
                )
                
                if response.status_code == 200 or response.status_code == 201:
                    data = response.json()
                    # API returns "id" not "subscription_id"
                    subscription_id = data.get("id") or data.get("subscription_id")
                    if subscription_id:
                        self._subscriptions.add(subscription_id)
                        logger.info(
                            "subscription_created",
                            subscription_id=subscription_id,
                            channel_type=channel_config["channel_type"],
                            symbol=channel_config.get("symbol"),
                        )
                else:
                    # Log response body for debugging 422 errors
                    try:
                        error_body = response.json()
                    except Exception:
                        error_body = response.text
                    
                    logger.warning(
                        "subscription_failed",
                        channel_type=channel_config["channel_type"],
                        symbol=channel_config.get("symbol"),
                        status_code=response.status_code,
                        error_body=error_body,
                    )
            except Exception as e:
                # Handle ws-gateway unavailability (T078): continue with last available data, log issues
                logger.error(
                    "subscription_error_ws_gateway_unavailable",
                    channel_type=channel_config["channel_type"],
                    symbol=channel_config.get("symbol"),
                    error=str(e),
                    message="ws-gateway unavailable, will retry subscription later",
                    exc_info=True,
                )
                # Continue processing - service will work with last available data
    
    async def _start_consumers(self) -> None:
        """Start consuming from all relevant queues."""
        queues = [
            "ws-gateway.orderbook",
            "ws-gateway.trades",
            "ws-gateway.ticker",
            "ws-gateway.kline",
            "ws-gateway.funding",
        ]
        
        for queue_name in queues:
            task = asyncio.create_task(self._consume_queue(queue_name))
            self._consumers.append(task)
    
    async def _consume_queue(self, queue_name: str) -> None:
        """Consume messages from a specific queue."""
        retry_delay = 5  # Initial retry delay in seconds
        max_retry_delay = 60  # Maximum retry delay
        
        while self._running:
            try:
                channel = await self._mq_manager.get_channel()
                # Use passive=True to use existing queue with its parameters (TTL, etc.)
                # ws-gateway queues have x-message-ttl: 86400000 (24 hours)
                # We must use passive=True to avoid PRECONDITION_FAILED errors
                # NOTE: When using passive=True, do NOT specify durable or any other arguments
                # - queue already exists with specific parameters set by ws-gateway
                queue = await channel.declare_queue(queue_name, passive=True)
                
                async def process_message(message: IncomingMessage):
                    async with message.process():
                        try:
                            body = json.loads(message.body.decode())
                            logger.debug(
                                "message_received",
                                queue=queue_name,
                                event_type=body.get("event_type"),
                                topic=body.get("topic"),
                            )
                            await self._process_market_data_event(body, queue_name)
                        except Exception as e:
                            logger.error(
                                "message_processing_error",
                                queue=queue_name,
                                error=str(e),
                                exc_info=True,
                            )
                
                await queue.consume(process_message)
                logger.info("consuming_queue", queue=queue_name)
                retry_delay = 5  # Reset retry delay on success
                
                # Keep consuming while running
                while self._running:
                    await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                logger.info("queue_consumer_cancelled", queue=queue_name)
                break
            except Exception as e:
                # Handle ws-gateway unavailability (T078): retry with exponential backoff
                logger.warning(
                    "queue_consumer_error_retrying",
                    queue=queue_name,
                    error=str(e),
                    retry_delay=retry_delay,
                    message="ws-gateway queue unavailable, retrying...",
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)  # Exponential backoff
    
    def _extract_symbol_from_topic(self, topic: str) -> Optional[str]:
        """
        Extract symbol from topic string.
        
        Examples:
        - "tickers.BTCUSDT" -> "BTCUSDT"
        - "orderbook.1.ETHUSDT" -> "ETHUSDT"
        - "trade.BTCUSDT" -> "BTCUSDT"
        
        Args:
            topic: Topic string from ws-gateway
            
        Returns:
            Symbol string or None if not found
        """
        if not topic:
            return None
        
        # Topic format: "channel.symbol" or "channel.param.symbol"
        parts = topic.split(".")
        if len(parts) >= 2:
            # Last part is usually the symbol
            return parts[-1]
        
        return None
    
    async def _process_market_data_event(self, event: Dict, queue_name: str) -> None:
        """Process a market data event."""
        # Add internal timestamp and exchange timestamp to all received messages (T074)
        now = datetime.now(timezone.utc)
        event["internal_timestamp"] = now.isoformat()
        
        # Preserve exchange timestamp if present
        if "timestamp" in event:
            if isinstance(event["timestamp"], str):
                event["exchange_timestamp"] = event["timestamp"]
            elif isinstance(event["timestamp"], (int, float)):
                # Convert Unix timestamp to ISO string
                event["exchange_timestamp"] = datetime.fromtimestamp(
                    event["timestamp"] / 1000 if event["timestamp"] > 1e10 else event["timestamp"],
                    tz=timezone.utc
                ).isoformat()
        else:
            # If no timestamp in event, use internal timestamp
            event["exchange_timestamp"] = now.isoformat()
        
        event_type = event.get("event_type")
        
        # Extract symbol from event structure
        # ws-gateway publishes events with payload containing symbol
        # Symbol can be in: event.symbol, event.payload.symbol, event.payload.s, or extracted from topic
        payload = event.get("payload", {})
        topic = event.get("topic", "")
        
        symbol = (
            event.get("symbol") or
            (payload.get("symbol") if isinstance(payload, dict) else None) or
            (payload.get("s") if isinstance(payload, dict) else None) or
            self._extract_symbol_from_topic(topic)
        )
        
        if not symbol:
            logger.warning(
                "event_missing_symbol",
                event_type=event_type,
                queue=queue_name,
                topic=topic,
                has_payload=bool(payload),
                payload_keys=list(payload.keys()) if isinstance(payload, dict) else None,
            )
            return
        
        # Add symbol to event if it was extracted from payload or topic
        if "symbol" not in event:
            event["symbol"] = symbol
        
        try:
            # Update feature computer state
            self._feature_computer.update_market_data(event)
            
            # Store raw data (T135: Integrate raw data storage into market data consumer)
            if self._data_storage:
                # Store asynchronously (non-blocking) to avoid impacting feature computation latency
                asyncio.create_task(self._data_storage.store_market_data_event(event))
            
            # Handle orderbook snapshot requests
            if event_type == "orderbook_snapshot" and self._orderbook_manager.is_desynchronized(symbol):
                # Snapshot received, orderbook will be updated
                pass
            
            logger.debug(
                "market_data_event_processed",
                event_type=event_type,
                symbol=symbol,
                queue=queue_name,
            )
        
        except Exception as e:
            logger.error(
                "event_processing_error",
                event_type=event_type,
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )

