"""HTTP client for Position Manager REST API.

This client allows Order Manager to query positions and portfolio metrics
from Position Manager service as the single source of truth for position data.
"""

from decimal import Decimal
from typing import Optional, List
from datetime import datetime

import httpx
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.position import Position
from ..exceptions import OrderExecutionError

logger = get_logger(__name__)


class PortfolioExposure(BaseModel):
    """Portfolio exposure response from Position Manager."""

    total_exposure_usdt: Decimal = Field(..., description="Total portfolio exposure in USDT")
    calculated_at: datetime = Field(..., description="Timestamp when exposure was calculated")


class PositionManagerClient:
    """HTTP client for Position Manager REST API."""

    def __init__(self):
        """Initialize Position Manager client."""
        self.base_url = f"http://{settings.position_manager_host}:{settings.position_manager_port}"
        self.api_key = settings.position_manager_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=10.0,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_position(
        self, asset: str, mode: str = "one-way", trace_id: Optional[str] = None
    ) -> Optional[Position]:
        """Get position for a specific asset from Position Manager.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')
            mode: Trading mode ('one-way' or 'hedge')
            trace_id: Optional trace ID for logging

        Returns:
            Position object if found, None otherwise

        Raises:
            OrderExecutionError: If API call fails
        """
        try:
            client = await self._get_client()
            url = f"/api/v1/positions/{asset}"
            params = {"mode": mode}

            logger.debug(
                "position_manager_get_position_request",
                asset=asset,
                mode=mode,
                trace_id=trace_id,
            )

            response = await client.get(url, params=params)

            if response.status_code == 404:
                logger.debug(
                    "position_manager_position_not_found",
                    asset=asset,
                    mode=mode,
                    trace_id=trace_id,
                )
                return None

            if response.status_code != 200:
                error_msg = f"Failed to get position: {response.status_code} - {response.text}"
                logger.error(
                    "position_manager_get_position_failed",
                    asset=asset,
                    mode=mode,
                    status_code=response.status_code,
                    error=error_msg,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)

            data = response.json()

            # Convert Position Manager API response to Order Manager Position model
            position = Position.from_dict(data)

            logger.debug(
                "position_manager_get_position_success",
                asset=asset,
                mode=mode,
                size=float(position.size),
                trace_id=trace_id,
            )

            return position

        except httpx.HTTPError as e:
            error_msg = f"HTTP error getting position: {e}"
            logger.error(
                "position_manager_get_position_http_error",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to get position: {e}"
            logger.error(
                "position_manager_get_position_error",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise OrderExecutionError(error_msg) from e

    async def get_all_positions(
        self, asset: Optional[str] = None, mode: Optional[str] = None, trace_id: Optional[str] = None
    ) -> List[Position]:
        """Get all positions from Position Manager.

        Args:
            asset: Optional asset filter
            mode: Optional mode filter
            trace_id: Optional trace ID for logging

        Returns:
            List of Position objects

        Raises:
            OrderExecutionError: If API call fails
        """
        try:
            client = await self._get_client()
            url = "/api/v1/positions"
            params = {}
            if asset:
                params["asset"] = asset
            if mode:
                params["mode"] = mode

            logger.debug(
                "position_manager_get_all_positions_request",
                asset=asset,
                mode=mode,
                trace_id=trace_id,
            )

            response = await client.get(url, params=params)

            if response.status_code != 200:
                error_msg = f"Failed to get positions: {response.status_code} - {response.text}"
                logger.error(
                    "position_manager_get_all_positions_failed",
                    asset=asset,
                    mode=mode,
                    status_code=response.status_code,
                    error=error_msg,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)

            data = response.json()
            positions_data = data.get("positions", [])

            # Convert Position Manager API responses to Order Manager Position models
            positions = [Position.from_dict(pos_data) for pos_data in positions_data]

            logger.debug(
                "position_manager_get_all_positions_success",
                count=len(positions),
                asset=asset,
                mode=mode,
                trace_id=trace_id,
            )

            return positions

        except httpx.HTTPError as e:
            error_msg = f"HTTP error getting positions: {e}"
            logger.error(
                "position_manager_get_all_positions_http_error",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to get positions: {e}"
            logger.error(
                "position_manager_get_all_positions_error",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise OrderExecutionError(error_msg) from e

    async def get_portfolio_exposure(self, trace_id: Optional[str] = None) -> PortfolioExposure:
        """Get portfolio exposure from Position Manager.

        Args:
            trace_id: Optional trace ID for logging

        Returns:
            PortfolioExposure with total_exposure_usdt and calculated_at

        Raises:
            OrderExecutionError: If API call fails
        """
        try:
            client = await self._get_client()
            url = "/api/v1/portfolio/exposure"

            logger.debug(
                "position_manager_get_portfolio_exposure_request",
                trace_id=trace_id,
            )

            response = await client.get(url)

            if response.status_code == 503:
                # Position Manager explicitly signals temporary unavailability
                error_msg = "Portfolio exposure is temporarily unavailable from Position Manager"
                logger.warning(
                    "position_manager_portfolio_exposure_unavailable",
                    status_code=response.status_code,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)

            if response.status_code != 200:
                error_msg = f"Failed to get portfolio exposure: {response.status_code} - {response.text}"
                logger.error(
                    "position_manager_get_portfolio_exposure_failed",
                    status_code=response.status_code,
                    error=error_msg,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(error_msg)

            data = response.json()

            # Convert response to PortfolioExposure model
            exposure = PortfolioExposure(
                total_exposure_usdt=Decimal(str(data["total_exposure_usdt"])),
                calculated_at=datetime.fromisoformat(data["calculated_at"].replace("Z", "+00:00")),
            )

            logger.debug(
                "position_manager_get_portfolio_exposure_success",
                total_exposure_usdt=float(exposure.total_exposure_usdt),
                calculated_at=exposure.calculated_at.isoformat(),
                trace_id=trace_id,
            )

            return exposure

        except httpx.HTTPError as e:
            error_msg = f"HTTP error getting portfolio exposure: {e}"
            logger.error(
                "position_manager_get_portfolio_exposure_http_error",
                error=str(e),
                trace_id=trace_id,
            )
            raise OrderExecutionError(error_msg) from e
        except Exception as e:
            if isinstance(e, OrderExecutionError):
                raise
            error_msg = f"Failed to get portfolio exposure: {e}"
            logger.error(
                "position_manager_get_portfolio_exposure_error",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise OrderExecutionError(error_msg) from e

