"""Portfolio-related API routes for Position Manager."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...publishers import PositionEventPublisher
from ...services.portfolio_manager import PortfolioManager, default_portfolio_manager
from ...services.position_manager import PositionManager
from ...utils.tracing import get_or_create_trace_id
from ..middleware.auth import api_key_auth


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["portfolio"])


def get_portfolio_manager() -> PortfolioManager:
    # Use shared default instance so that cache invalidation triggered by
    # position updates affects subsequent API calls.
    return default_portfolio_manager


@router.get(
    "/portfolio",
    dependencies=[Depends(api_key_auth)],
)
async def get_portfolio(
    include_positions: bool = Query(
        False,
        description="Include individual positions in the response",
    ),
    asset: Optional[str] = Query(
        None,
        description="Optional asset filter for portfolio metrics",
    ),
    portfolio_manager: PortfolioManager = Depends(get_portfolio_manager),
):
    """Return full portfolio metrics, optionally including positions."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "portfolio_request",
        include_positions=include_positions,
        asset=asset,
        trace_id=trace_id,
    )

    try:
        metrics = await portfolio_manager.get_portfolio_metrics(
            include_positions=include_positions,
            asset_filter=asset,
        )

        response = metrics.model_dump()

        # Attach positions if present on metrics (set in calculate_metrics_from_positions)
        positions = getattr(metrics, "_positions", None)
        if positions is not None:
            response["positions"] = [
                {
                    "asset": p.asset,
                    "mode": p.mode,
                    "size": str(p.size),
                    "current_price": str(p.current_price) if p.current_price is not None else None,
                }
                for p in positions
            ]

        # Best-effort publish portfolio_updated event
        try:
            await PositionEventPublisher.publish_portfolio_updated(metrics, trace_id=trace_id)
        except Exception:  # pragma: no cover
            logger.warning("portfolio_event_publish_failed", trace_id=trace_id)

        logger.info("portfolio_completed", trace_id=trace_id)
        return JSONResponse(status_code=200, content=jsonable_encoder(response))
    except Exception as e:
        logger.error("portfolio_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to calculate portfolio metrics") from e


@router.get(
    "/portfolio/exposure",
    dependencies=[Depends(api_key_auth)],
)
async def get_portfolio_exposure(
    request: Request,
    portfolio_manager: PortfolioManager = Depends(get_portfolio_manager),
):
    """Return portfolio exposure-only metrics."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "portfolio_exposure_request",
        trace_id=trace_id,
        client=str(request.client.host) if request.client else None,
    )

    try:
        exposure = await portfolio_manager.get_total_exposure()
        logger.info(
            "portfolio_exposure_completed",
            trace_id=trace_id,
            calculated_at=str(exposure.calculated_at),
        )
        return JSONResponse(status_code=200, content=jsonable_encoder(exposure.model_dump()))
    except Exception as e:
        logger.error(
            "portfolio_exposure_failed",
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        # For risk management components, explicitly signal temporary
        # unavailability with 503 so they can fall back to last-known
        # metrics instead of treating it as a hard failure (T069).
        raise HTTPException(
            status_code=503,
            detail="Portfolio exposure is temporarily unavailable",
        ) from e


@router.get(
    "/portfolio/pnl",
    dependencies=[Depends(api_key_auth)],
)
async def get_portfolio_pnl(
    portfolio_manager: PortfolioManager = Depends(get_portfolio_manager),
):
    """Return portfolio PnL-only metrics."""
    trace_id = get_or_create_trace_id()
    logger.info("portfolio_pnl_request", trace_id=trace_id)

    try:
        pnl = await portfolio_manager.get_portfolio_pnl()
        logger.info("portfolio_pnl_completed", trace_id=trace_id)
        return JSONResponse(status_code=200, content=jsonable_encoder(pnl.model_dump()))
    except Exception as e:
        logger.error("portfolio_pnl_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get portfolio PnL") from e



