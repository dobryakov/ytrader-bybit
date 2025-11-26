"""
Market data subscription service.

Manages subscriptions to ws-gateway channels via REST API for ticker, orderbook, and kline data.
"""

import httpx
from typing import List, Dict, Optional
from uuid import UUID

from ..config.settings import settings
from ..config.logging import get_logger
from ..config.exceptions import ModelServiceError

logger = get_logger(__name__)


class MarketDataSubscriptionError(ModelServiceError):
    """Error during market data subscription operations."""

    pass


class MarketDataSubscriber:
    """Manages subscriptions to ws-gateway market data channels."""

    def __init__(self, ws_gateway_url: str, api_key: str):
        """
        Initialize market data subscriber.

        Args:
            ws_gateway_url: Base URL for ws-gateway service (e.g., 'http://ws-gateway:4400')
            api_key: API key for ws-gateway authentication
        """
        self.ws_gateway_url = ws_gateway_url.rstrip("/")
        self.api_key = api_key
        self.subscriptions: Dict[str, str] = {}  # channel_type -> subscription_id
        self.client = httpx.AsyncClient(timeout=10.0)

    async def subscribe(
        self, channel_type: str, symbol: str, requesting_service: str = "model-service"
    ) -> str:
        """
        Subscribe to a market data channel.

        Args:
            channel_type: Type of channel ('ticker', 'orderbook', 'kline')
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            requesting_service: Service identifier requesting the subscription

        Returns:
            Subscription ID (UUID string)

        Raises:
            MarketDataSubscriptionError: If subscription fails
        """
        if channel_type not in ("ticker", "orderbook", "kline"):
            raise ValueError(f"Invalid channel_type: {channel_type}. Must be 'ticker', 'orderbook', or 'kline'")

        subscription_key = f"{channel_type}:{symbol}"
        if subscription_key in self.subscriptions:
            logger.info(
                "Subscription already exists",
                channel_type=channel_type,
                symbol=symbol,
                subscription_id=self.subscriptions[subscription_key],
            )
            return self.subscriptions[subscription_key]

        try:
            logger.info("Subscribing to market data channel", channel_type=channel_type, symbol=symbol)

            response = await self.client.post(
                f"{self.ws_gateway_url}/api/v1/subscriptions",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
                json={
                    "channel_type": channel_type,
                    "symbol": symbol,
                    "requesting_service": requesting_service,
                },
            )

            if response.status_code == 201:
                data = response.json()
                subscription_id = data.get("id")
                if not subscription_id:
                    raise MarketDataSubscriptionError("Subscription response missing 'id' field")

                self.subscriptions[subscription_key] = subscription_id
                logger.info(
                    "Successfully subscribed to market data channel",
                    channel_type=channel_type,
                    symbol=symbol,
                    subscription_id=subscription_id,
                )
                return subscription_id
            elif response.status_code == 409:
                # Subscription already exists for this service
                data = response.json()
                subscription_id = data.get("id")
                if subscription_id:
                    self.subscriptions[subscription_key] = subscription_id
                    logger.info(
                        "Subscription already exists for service",
                        channel_type=channel_type,
                        symbol=symbol,
                        subscription_id=subscription_id,
                    )
                    return subscription_id
                raise MarketDataSubscriptionError(f"Subscription conflict but no ID returned: {response.text}")
            else:
                error_text = response.text
                logger.error(
                    "Failed to subscribe to market data channel",
                    channel_type=channel_type,
                    symbol=symbol,
                    status_code=response.status_code,
                    error=error_text,
                )
                raise MarketDataSubscriptionError(
                    f"Failed to subscribe to {channel_type} for {symbol}: {response.status_code} - {error_text}"
                )
        except httpx.RequestError as e:
            logger.error("Network error during subscription", channel_type=channel_type, symbol=symbol, error=str(e))
            raise MarketDataSubscriptionError(f"Network error during subscription: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during subscription", channel_type=channel_type, symbol=symbol, error=str(e))
            raise MarketDataSubscriptionError(f"Unexpected error during subscription: {e}") from e

    async def subscribe_all_channels(self, symbols: List[str], requesting_service: str = "model-service") -> Dict[str, Dict[str, str]]:
        """
        Subscribe to all required channels (ticker, orderbook, kline) for given symbols.

        Args:
            symbols: List of trading pair symbols
            requesting_service: Service identifier requesting the subscriptions

        Returns:
            Dictionary mapping symbol to channel_type to subscription_id
        """
        subscriptions = {}
        for symbol in symbols:
            subscriptions[symbol] = {}
            for channel_type in ("ticker", "orderbook", "kline"):
                try:
                    subscription_id = await self.subscribe(channel_type, symbol, requesting_service)
                    subscriptions[symbol][channel_type] = subscription_id
                except MarketDataSubscriptionError as e:
                    logger.warning(
                        "Failed to subscribe to channel",
                        channel_type=channel_type,
                        symbol=symbol,
                        error=str(e),
                    )
                    # Continue with other channels
        return subscriptions

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Cancel a subscription.

        Args:
            subscription_id: Subscription ID to cancel

        Raises:
            MarketDataSubscriptionError: If unsubscription fails
        """
        try:
            logger.info("Cancelling subscription", subscription_id=subscription_id)

            response = await self.client.delete(
                f"{self.ws_gateway_url}/api/v1/subscriptions/{subscription_id}",
                headers={"X-API-Key": self.api_key},
            )

            if response.status_code == 200:
                # Remove from local tracking
                for key, sub_id in list(self.subscriptions.items()):
                    if sub_id == subscription_id:
                        del self.subscriptions[key]
                        break

                logger.info("Successfully cancelled subscription", subscription_id=subscription_id)
            elif response.status_code == 404:
                logger.warning("Subscription not found (may already be cancelled)", subscription_id=subscription_id)
            else:
                error_text = response.text
                logger.error(
                    "Failed to cancel subscription",
                    subscription_id=subscription_id,
                    status_code=response.status_code,
                    error=error_text,
                )
                raise MarketDataSubscriptionError(
                    f"Failed to cancel subscription {subscription_id}: {response.status_code} - {error_text}"
                )
        except httpx.RequestError as e:
            logger.error("Network error during unsubscription", subscription_id=subscription_id, error=str(e))
            raise MarketDataSubscriptionError(f"Network error during unsubscription: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during unsubscription", subscription_id=subscription_id, error=str(e))
            raise MarketDataSubscriptionError(f"Unexpected error during unsubscription: {e}") from e

    async def unsubscribe_all(self) -> None:
        """Cancel all active subscriptions."""
        subscription_ids = list(self.subscriptions.values())
        for subscription_id in subscription_ids:
            try:
                await self.unsubscribe(subscription_id)
            except MarketDataSubscriptionError as e:
                logger.warning("Failed to unsubscribe", subscription_id=subscription_id, error=str(e))
                # Continue with other subscriptions

    async def close(self) -> None:
        """Close HTTP client and cleanup."""
        await self.client.aclose()

