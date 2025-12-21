"""WebSocket Gateway API client for position manager."""

from __future__ import annotations

import httpx
from typing import Optional

from ..config.logging import get_logger
from ..config.settings import settings
from ..exceptions import QueueError

logger = get_logger(__name__)


class WSGatewayClient:
    """Client for interacting with ws-gateway API."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize WS Gateway client."""
        self.base_url = base_url or settings.ws_gateway_url
        self.api_key = api_key or settings.ws_gateway_api_key

    async def subscribe_to_position(self) -> None:
        """
        Subscribe to position updates from ws-gateway.

        Raises:
            QueueError: If subscription fails
        """
        url = f"{self.base_url}/api/v1/subscriptions"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "channel_type": "position",
            "requesting_service": settings.position_manager_service_name,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                logger.info(
                    "ws_gateway_position_subscription_created",
                    subscription_id=result.get("id"),
                    topic=result.get("topic"),
                )
        except httpx.HTTPStatusError as e:
            # If subscription already exists (409), that's okay
            if e.response.status_code == 409:
                logger.info(
                    "ws_gateway_position_subscription_already_exists",
                    error=e.response.text,
                )
                return
            error_msg = f"Failed to create subscription: {e.response.status_code} - {e.response.text}"
            logger.error(
                "ws_gateway_position_subscription_failed",
                status_code=e.response.status_code,
                error=e.response.text,
                exc_info=True,
            )
            raise QueueError(error_msg) from e
        except httpx.RequestError as e:
            logger.error(
                "ws_gateway_position_subscription_request_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise QueueError(f"HTTP error creating subscription: {e}") from e
        except Exception as e:
            logger.error(
                "ws_gateway_position_subscription_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise QueueError(f"Failed to subscribe to position events: {e}") from e

