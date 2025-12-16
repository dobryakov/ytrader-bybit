"""
Position Manager REST API client.

Provides access to position data from Position Manager service for risk management.
"""

from typing import Optional, Dict, Any
import httpx
from decimal import Decimal

from ..config.settings import settings
from ..config.logging import get_logger
from .position_cache import position_cache

logger = get_logger(__name__)


class PositionManagerClient:
    """Client for Position Manager REST API."""

    def __init__(self):
        """Initialize Position Manager client."""
        self.base_url = settings.position_manager_url
        self.api_key = settings.position_manager_api_key
        self.timeout = 5.0  # 5 second timeout for position queries

    async def get_position(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Get position data for an asset from Position Manager.

        Checks cache first, then falls back to REST API if cache miss or expired.
        Updates cache after successful API response.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Dictionary with position data or None if not found/error:
            {
                'asset': str,
                'unrealized_pnl_pct': float,  # Unrealized P&L percentage
                'position_size_norm': float,   # Position size normalized (0.0-1.0)
                'size': float,                 # Position size
                'unrealized_pnl': float,       # Unrealized P&L absolute value
                ...
            }
        """
        # Check cache first
        cached_data = await position_cache.get(asset)
        if cached_data is not None:
            logger.debug("Cache hit for position", asset=asset)
            return cached_data

        # Cache miss or expired - fetch from REST API
        logger.debug("Cache miss for position, fetching from API", asset=asset)
        url = f"{self.base_url}/api/v1/positions/{asset}"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # Update cache after successful API response
                await position_cache.set(asset, data)
                logger.debug("Retrieved position from Position Manager and cached", asset=asset, data=data)
                return data

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Position not found - this is normal if no position exists
                logger.debug("Position not found for asset", asset=asset)
                return None
            logger.error(
                "Position Manager API error",
                asset=asset,
                status_code=e.response.status_code,
                error=str(e),
            )
            return None
        except httpx.TimeoutException:
            logger.warning("Position Manager API timeout", asset=asset, timeout=self.timeout)
            return None
        except Exception as e:
            logger.error("Failed to query Position Manager", asset=asset, error=str(e), exc_info=True)
            return None

    async def get_unrealized_pnl_pct(self, asset: str) -> Optional[float]:
        """
        Get unrealized P&L percentage for an asset.

        Args:
            asset: Trading pair symbol

        Returns:
            Unrealized P&L percentage or None if position not found/error/closed
        """
        position = await self.get_position(asset)
        if not position:
            return None

        unrealized_pnl_pct = position.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is None:
            # Check if position is closed (size = 0) - this is normal, not an error
            size = position.get("size")
            if size is not None:
                try:
                    size_float = float(size)
                    if size_float == 0:
                        # Position is closed - no need to log warning
                        logger.debug("Position is closed (size=0), unrealized_pnl_pct unavailable", asset=asset)
                        return None
                except (ValueError, TypeError):
                    pass
            
            # Position exists but unrealized_pnl_pct is None for other reasons (e.g., missing average_entry_price)
            logger.warning("Position data missing unrealized_pnl_pct", asset=asset, position=position)
            return None

        return float(unrealized_pnl_pct)

    async def get_position_size_norm(self, asset: str) -> Optional[float]:
        """
        Get normalized position size for an asset.

        Args:
            asset: Trading pair symbol

        Returns:
            Normalized position size (0.0-1.0) or None if position not found/error
        """
        position = await self.get_position(asset)
        if not position:
            return None

        position_size_norm = position.get("position_size_norm")
        if position_size_norm is None:
            logger.warning("Position data missing position_size_norm", asset=asset, position=position)
            return None

        return float(position_size_norm)

    async def get_position_size(self, asset: str) -> Optional[float]:
        """
        Get absolute position size for an asset.

        Args:
            asset: Trading pair symbol

        Returns:
            Position size (absolute value) or None if position not found/error
        """
        position = await self.get_position(asset)
        if not position:
            return None

        size = position.get("size")
        if size is None:
            logger.warning("Position data missing size", asset=asset, position=position)
            return None

        return abs(float(size))  # Return absolute value


# Global Position Manager client instance
position_manager_client = PositionManagerClient()

