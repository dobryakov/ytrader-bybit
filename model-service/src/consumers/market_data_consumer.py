"""
Market data consumer.

Consumes events from RabbitMQ queues (ws-gateway.ticker, ws-gateway.orderbook, ws-gateway.kline),
parses and caches latest values in memory for fast access.
"""

import json
import asyncio
import threading
from typing import Dict, Optional, Any
from datetime import datetime
from collections import defaultdict

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from ..config.rabbitmq import rabbitmq_manager
from ..config.logging import get_logger
from ..config.exceptions import MessageQueueError

logger = get_logger(__name__)


class MarketDataCache:
    """In-memory cache for latest market data values."""

    def __init__(self):
        """Initialize market data cache."""
        # Cache structure: symbol -> {price, spread, volume_24h, volatility, last_updated}
        self._cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()

    def update_ticker(self, symbol: str, price: float, volume_24h: float) -> None:
        """
        Update cache with ticker data.

        Args:
            symbol: Trading pair symbol
            price: Current market price
            volume_24h: 24-hour trading volume
        """
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            self._cache[symbol]["price"] = price
            self._cache[symbol]["volume_24h"] = volume_24h
            self._cache[symbol]["last_updated"] = datetime.utcnow()

    def update_orderbook(self, symbol: str, spread: float, bid_depth: Optional[float] = None, ask_depth: Optional[float] = None) -> None:
        """
        Update cache with orderbook data.

        Args:
            symbol: Trading pair symbol
            spread: Bid-ask spread
            bid_depth: Bid depth (optional)
            ask_depth: Ask depth (optional)
        """
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            self._cache[symbol]["spread"] = spread
            if bid_depth is not None:
                self._cache[symbol]["bid_depth"] = bid_depth
            if ask_depth is not None:
                self._cache[symbol]["ask_depth"] = ask_depth
            self._cache[symbol]["last_updated"] = datetime.utcnow()

    def update_kline(self, symbol: str, volatility: float) -> None:
        """
        Update cache with kline data (for volatility calculation).

        Args:
            symbol: Trading pair symbol
            volatility: Calculated volatility measure
        """
        with self._lock:
            if symbol not in self._cache:
                self._cache[symbol] = {}
            self._cache[symbol]["volatility"] = volatility
            self._cache[symbol]["last_updated"] = datetime.utcnow()

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest market data for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with price, spread, volume_24h, volatility, or None if data unavailable
        """
        with self._lock:
            data = self._cache.get(symbol, {})
            if not data:
                return None

            # Check if we have all required fields
            required_fields = ["price", "spread", "volume_24h", "volatility"]
            if not all(field in data for field in required_fields):
                return None

            return {
                "price": data.get("price"),
                "spread": data.get("spread"),
                "volume_24h": data.get("volume_24h"),
                "volatility": data.get("volatility"),
                "orderbook_depth": {
                    "bid_depth": data.get("bid_depth"),
                    "ask_depth": data.get("ask_depth"),
                } if "bid_depth" in data or "ask_depth" in data else None,
                "last_updated": data.get("last_updated"),
            }

    def clear(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache for a symbol or all symbols.

        Args:
            symbol: Symbol to clear (None for all)
        """
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()


class MarketDataConsumer:
    """Consumes market data events from RabbitMQ queues."""

    def __init__(self, cache: MarketDataCache):
        """
        Initialize market data consumer.

        Args:
            cache: Market data cache instance
        """
        self.cache = cache
        self._consumers: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Start consuming market data from RabbitMQ queues."""
        if self._running:
            logger.warning("Market data consumer already running")
            return

        self._running = True
        logger.info("Starting market data consumer")

        # Start consumers for each queue
        queues = ["ws-gateway.ticker", "ws-gateway.orderbook", "ws-gateway.kline"]
        for queue_name in queues:
            try:
                task = asyncio.create_task(self._consume_queue(queue_name))
                self._consumers[queue_name] = task
                logger.info("Started consumer for queue", queue=queue_name)
            except Exception as e:
                logger.error("Failed to start consumer", queue=queue_name, error=str(e), exc_info=True)

    async def stop(self) -> None:
        """Stop consuming market data."""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping market data consumer")

        # Cancel all consumer tasks
        for queue_name, task in self._consumers.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped consumer for queue", queue=queue_name)

        self._consumers.clear()

    async def _consume_queue(self, queue_name: str) -> None:
        """
        Consume messages from a RabbitMQ queue.

        Args:
            queue_name: Name of the queue to consume from
        """
        try:
            channel = await rabbitmq_manager.get_channel()
            # Use existing queue without redeclaring (queues are created by ws-gateway with TTL)
            # Just bind to the existing queue
            queue = await channel.declare_queue(queue_name, durable=True, passive=True)

            async with queue.iterator() as queue_iter:
                async for message in queue_iter:
                    try:
                        async with message.process():
                            await self._process_message(queue_name, message)
                    except Exception as e:
                        logger.error(
                            "Error processing message",
                            queue=queue_name,
                            error=str(e),
                            exc_info=True,
                        )
                        # Continue processing other messages
        except asyncio.CancelledError:
            logger.info("Consumer cancelled", queue=queue_name)
            raise
        except Exception as e:
            logger.error("Consumer error", queue=queue_name, error=str(e), exc_info=True)
            raise

    async def _process_message(self, queue_name: str, message: AbstractIncomingMessage) -> None:
        """
        Process a single message from a queue.

        Args:
            queue_name: Name of the queue
            message: Incoming message
        """
        try:
            body = message.body.decode("utf-8")
            data = json.loads(body)

            if queue_name == "ws-gateway.ticker":
                self._process_ticker(data)
            elif queue_name == "ws-gateway.orderbook":
                self._process_orderbook(data)
            elif queue_name == "ws-gateway.kline":
                self._process_kline(data)
            else:
                logger.warning("Unknown queue type", queue=queue_name)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse message JSON", queue=queue_name, error=str(e))
        except Exception as e:
            logger.error("Error processing message", queue=queue_name, error=str(e), exc_info=True)

    def _process_ticker(self, data: Dict[str, Any]) -> None:
        """
        Process ticker event.

        Args:
            data: Ticker event data (may be wrapped in 'payload' or be direct)
        """
        try:
            # Extract payload if event is wrapped
            payload = data.get("payload", data)
            
            # Extract symbol from payload or topic
            symbol = payload.get("symbol") or payload.get("s")
            if not symbol:
                # Try to extract from topic if available
                topic = data.get("topic", "")
                if topic.startswith("tickers."):
                    symbol = topic.replace("tickers.", "")
                else:
                    logger.warning("Ticker event missing symbol", data=data)
                    return

            # Try different possible field names for price
            price = payload.get("lastPrice") or payload.get("price") or payload.get("c")
            if price is None:
                logger.warning("Ticker event missing price", symbol=symbol, data=data)
                return

            # Try different possible field names for volume
            volume_24h = payload.get("volume24h") or payload.get("volume_24h") or payload.get("v")
            if volume_24h is None:
                logger.warning("Ticker event missing volume_24h", symbol=symbol, data=data)
                return

            self.cache.update_ticker(symbol, float(price), float(volume_24h))
            logger.debug("Updated ticker cache", symbol=symbol, price=price, volume_24h=volume_24h)

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Error processing ticker event", error=str(e), data=data, exc_info=True)

    def _process_orderbook(self, data: Dict[str, Any]) -> None:
        """
        Process orderbook event.

        Args:
            data: Orderbook event data (may be wrapped in 'payload' or be direct)
        """
        try:
            # Extract payload if event is wrapped
            payload = data.get("payload", data)
            
            # Extract symbol from payload or topic
            symbol = payload.get("symbol") or payload.get("s")
            if not symbol:
                # Try to extract from topic if available
                topic = data.get("topic", "")
                if topic.startswith("orderbook."):
                    # Extract from topic like "orderbook.1.BTCUSDT"
                    parts = topic.split(".")
                    if len(parts) >= 3:
                        symbol = parts[-1]
                    else:
                        logger.warning("Orderbook event missing symbol", data=data)
                        return
                else:
                    logger.warning("Orderbook event missing symbol", data=data)
                    return

            # Extract best bid and ask prices
            bids = payload.get("bids") or payload.get("b") or []
            asks = payload.get("asks") or payload.get("a") or []

            if not bids or not asks:
                logger.warning("Orderbook event missing bid/ask data", symbol=symbol, data=data)
                return

            # Best bid is first element (highest price)
            best_bid = float(bids[0][0]) if isinstance(bids[0], list) else float(bids[0].get("price", bids[0]))
            # Best ask is first element (lowest price)
            best_ask = float(asks[0][0]) if isinstance(asks[0], list) else float(asks[0].get("price", asks[0]))

            spread = best_ask - best_bid

            # Calculate depth if available
            bid_depth = None
            ask_depth = None
            if len(bids) > 0 and isinstance(bids[0], list) and len(bids[0]) > 1:
                bid_depth = sum(float(bid[1]) for bid in bids[:10])  # Sum of top 10 bid levels
            if len(asks) > 0 and isinstance(asks[0], list) and len(asks[0]) > 1:
                ask_depth = sum(float(ask[1]) for ask in asks[:10])  # Sum of top 10 ask levels

            self.cache.update_orderbook(symbol, spread, bid_depth, ask_depth)
            logger.debug("Updated orderbook cache", symbol=symbol, spread=spread)

        except (ValueError, KeyError, TypeError, IndexError) as e:
            logger.error("Error processing orderbook event", error=str(e), data=data, exc_info=True)

    def _process_kline(self, data: Dict[str, Any]) -> None:
        """
        Process kline (candlestick) event.

        Args:
            data: Kline event data (may be wrapped in 'payload' or be direct)
        """
        try:
            # Extract payload if event is wrapped
            payload = data.get("payload", data)
            
            # Extract symbol from payload or topic
            symbol = payload.get("symbol") or payload.get("s")
            if not symbol:
                # Try to extract from topic if available
                topic = data.get("topic", "")
                if topic.startswith("kline."):
                    # Extract from topic like "kline.1.BTCUSDT"
                    parts = topic.split(".")
                    if len(parts) >= 3:
                        symbol = parts[-1]
                    else:
                        logger.warning("Kline event missing symbol", data=data)
                        return
                else:
                    logger.warning("Kline event missing symbol", data=data)
                    return

            # Calculate volatility from kline data
            # Volatility = (high - low) / close
            high = payload.get("high") or payload.get("h")
            low = payload.get("low") or payload.get("l")
            close = payload.get("close") or payload.get("c")

            if high is None or low is None or close is None:
                logger.warning("Kline event missing price data", symbol=symbol, data=data)
                return

            high = float(high)
            low = float(low)
            close = float(close)

            if close == 0:
                logger.warning("Kline event has zero close price", symbol=symbol)
                return

            volatility = (high - low) / close

            self.cache.update_kline(symbol, volatility)
            logger.debug("Updated kline cache", symbol=symbol, volatility=volatility)

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Error processing kline event", error=str(e), data=data, exc_info=True)


# Global market data cache instance
market_data_cache = MarketDataCache()

# Global market data consumer instance
market_data_consumer = MarketDataConsumer(market_data_cache)

