"""
Position Manager REST API client.

Provides access to position data from Position Manager service for risk management.
"""

from typing import Optional, Dict, Any, List
import httpx
from decimal import Decimal
from datetime import datetime, timedelta

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
        # Circuit breaker state for critical failures
        self._consecutive_critical_errors = 0
        self._max_consecutive_errors = 5  # Stop after 5 consecutive critical errors
        self._circuit_breaker_until: Optional[datetime] = None
        self._circuit_breaker_duration = timedelta(minutes=5)  # Stop attempts for 5 minutes

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

        # Check circuit breaker first
        if self._check_circuit_breaker():
            logger.warning(
                "Position Manager API circuit breaker active - skipping request",
                asset=asset,
                blocked_until=self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None,
            )
            return None
        
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

                # Record success - reset circuit breaker
                self._record_success()

                # Update cache after successful API response
                await position_cache.set(asset, data)
                logger.debug("Retrieved position from Position Manager and cached", asset=asset, data=data)
                return data

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if status_code == 404:
                # Position not found - this is normal if no position exists
                logger.debug("Position not found for asset", asset=asset)
                # Reset consecutive errors on 404 (expected case)
                self._consecutive_critical_errors = 0
                return None
            
            # 5xx errors are critical
            if status_code >= 500:
                self._record_critical_error(
                    "http_5xx",
                    {
                        "asset": asset,
                        "status_code": status_code,
                        "error": str(e),
                        "url": url,
                    },
                )
            else:
                logger.error(
                    "Position Manager API client error",
                    asset=asset,
                    status_code=status_code,
                    error=str(e),
                )
                # Reset consecutive errors on 4xx (client error, not server failure)
                self._consecutive_critical_errors = 0
            return None
        except httpx.TimeoutException:
            # Timeout is a critical error
            self._record_critical_error(
                "timeout",
                {
                    "asset": asset,
                    "timeout_seconds": self.timeout,
                    "url": url,
                },
            )
            return None
        except (httpx.ConnectError, httpx.NetworkError) as e:
            # Connection errors are critical
            self._record_critical_error(
                "connection_error",
                {
                    "asset": asset,
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return None
        except Exception as e:
            # Other unexpected errors - treat as critical
            self._record_critical_error(
                "unexpected_error",
                {
                    "asset": asset,
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.exception("Failed to query Position Manager", extra={"asset": asset})
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

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is active (too many consecutive critical errors).
        
        Returns:
            True if circuit breaker is active and requests should be blocked, False otherwise
        """
        if self._circuit_breaker_until is not None:
            if datetime.utcnow() < self._circuit_breaker_until:
                return True
            else:
                # Circuit breaker period expired, reset
                logger.info(
                    "Position Manager API circuit breaker expired, resuming attempts",
                    was_blocked_until=self._circuit_breaker_until.isoformat(),
                )
                self._circuit_breaker_until = None
                self._consecutive_critical_errors = 0
                return False
        return False

    def _record_critical_error(self, error_type: str, error_details: Dict[str, Any]) -> None:
        """
        Record a critical error and activate circuit breaker if threshold reached.
        
        Args:
            error_type: Type of error (e.g., 'timeout', 'http_5xx', 'connection_error')
            error_details: Dictionary with error details for logging
        """
        self._consecutive_critical_errors += 1
        
        logger.error(
            "Position Manager API critical error",
            error_type=error_type,
            consecutive_errors=self._consecutive_critical_errors,
            max_allowed=self._max_consecutive_errors,
            **error_details,
        )
        
        if self._consecutive_critical_errors >= self._max_consecutive_errors:
            self._circuit_breaker_until = datetime.utcnow() + self._circuit_breaker_duration
            logger.critical(
                "Position Manager API circuit breaker activated - stopping all attempts",
                consecutive_errors=self._consecutive_critical_errors,
                blocked_until=self._circuit_breaker_until.isoformat(),
                duration_minutes=self._circuit_breaker_duration.total_seconds() / 60,
                message="Position Manager API is unavailable. All position queries will be blocked until circuit breaker expires.",
            )

    def _record_success(self) -> None:
        """Reset circuit breaker state after successful request."""
        if self._consecutive_critical_errors > 0:
            logger.info(
                "Position Manager API recovered from errors",
                previous_consecutive_errors=self._consecutive_critical_errors,
            )
            self._consecutive_critical_errors = 0
            self._circuit_breaker_until = None

    async def get_all_positions(
        self,
        asset: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all active positions from Position Manager API.

        Args:
            asset: Optional asset filter (e.g., 'BTCUSDT')
            mode: Optional mode filter ('one-way' or 'hedge')

        Returns:
            List of position dictionaries or empty list on error/circuit breaker
        """
        # Check circuit breaker first
        if self._check_circuit_breaker():
            logger.warning(
                "Position Manager API circuit breaker active - skipping request",
                asset=asset,
                mode=mode,
                blocked_until=self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None,
            )
            return []
        
        url = f"{self.base_url}/api/v1/positions"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {}
        if asset:
            params["asset"] = asset
        if mode:
            params["mode"] = mode

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                positions = data.get("positions", [])
                
                # Record success - reset circuit breaker
                self._record_success()
                
                logger.debug(
                    "Retrieved positions from Position Manager",
                    count=len(positions),
                    asset=asset,
                    mode=mode,
                )
                return positions

        except httpx.TimeoutException as e:
            # Timeout is a critical error
            self._record_critical_error(
                "timeout",
                {
                    "asset": asset,
                    "mode": mode,
                    "timeout_seconds": self.timeout,
                    "url": url,
                },
            )
            return []
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            # 5xx errors are critical, 4xx are client errors (less critical but still logged)
            if status_code >= 500:
                # Server errors are critical
                self._record_critical_error(
                    "http_5xx",
                    {
                        "asset": asset,
                        "mode": mode,
                        "status_code": status_code,
                        "url": url,
                        "error": str(e),
                    },
                )
            else:
                # 4xx errors are client errors - log but don't count as critical
                logger.error(
                    "Position Manager API client error",
                    status_code=status_code,
                    error=str(e),
                    asset=asset,
                    mode=mode,
                    url=url,
                )
                # Reset consecutive errors on 4xx (client error, not server failure)
                self._consecutive_critical_errors = 0
            return []
        except (httpx.ConnectError, httpx.NetworkError) as e:
            # Connection errors are critical
            self._record_critical_error(
                "connection_error",
                {
                    "asset": asset,
                    "mode": mode,
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return []
        except Exception as e:
            # Other unexpected errors - treat as critical
            self._record_critical_error(
                "unexpected_error",
                {
                    "asset": asset,
                    "mode": mode,
                    "url": url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.exception(
                "Unexpected error querying Position Manager for positions list",
                extra={
                    "asset": asset,
                    "mode": mode,
                },
            )
            return []


# Global Position Manager client instance
position_manager_client = PositionManagerClient()

