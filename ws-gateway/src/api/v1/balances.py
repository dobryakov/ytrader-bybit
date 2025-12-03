"""Balance and margin REST API (v1)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...config.logging import get_logger
from ...services.database.account_margin_balance_repository import (
    AccountMarginBalanceRepository,
)
from ...services.database.balance_repository import BalanceRepository
from .schemas import (
    BalanceHistoryResponse,
    LatestBalanceView,
    LatestBalancesResponse,
    MarginBalanceView,
)

router = APIRouter(prefix="/api/v1/balances", tags=["balances"])
logger = get_logger(__name__)


def _to_latest_balance_view(balance) -> LatestBalanceView:
    return LatestBalanceView(
        coin=balance.coin,
        wallet_balance=balance.wallet_balance,
        available_balance=balance.available_balance,
        frozen=balance.frozen,
        equity=balance.equity,
        usd_value=balance.usd_value,
        margin_collateral=balance.margin_collateral,
        total_order_im=balance.total_order_im,
        total_position_im=balance.total_position_im,
        event_timestamp=balance.event_timestamp,
        received_at=balance.received_at,
    )


def _to_margin_balance_view(margin_balance) -> MarginBalanceView:
    return MarginBalanceView(
        account_type=margin_balance.account_type,
        total_equity=margin_balance.total_equity,
        total_wallet_balance=margin_balance.total_wallet_balance,
        total_margin_balance=margin_balance.total_margin_balance,
        total_available_balance=margin_balance.total_available_balance,
        total_initial_margin=margin_balance.total_initial_margin,
        total_maintenance_margin=margin_balance.total_maintenance_margin,
        total_order_im=margin_balance.total_order_im,
        base_currency=margin_balance.base_currency,
        event_timestamp=margin_balance.event_timestamp,
        received_at=margin_balance.received_at,
    )


@router.get(
    "",
    response_model=LatestBalancesResponse,
    summary="Get latest balances and margin summary",
)
async def get_latest_balances(
    coin: Optional[str] = Query(
        default=None,
        description="Optional coin filter (e.g., 'USDT'). If omitted, returns latest balance per coin.",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of coins to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Offset for pagination over distinct coins.",
    ),
) -> LatestBalancesResponse:
    """Return latest balance per coin plus latest account-level margin balance."""
    try:
        balances = await BalanceRepository.list_latest_balances(
            coin=coin,
            limit=limit,
            offset=offset,
        )
        margin_balance = await AccountMarginBalanceRepository.get_latest_margin_balance()
    except Exception as exc:
        logger.error(
            "balances_latest_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch latest balances")

    balance_views = [_to_latest_balance_view(b) for b in balances]
    margin_view = _to_margin_balance_view(margin_balance) if margin_balance else None

    return LatestBalancesResponse(
        balances=balance_views,
        margin_balance=margin_view,
        total=len(balance_views),
    )


@router.get(
    "/history",
    response_model=BalanceHistoryResponse,
    summary="Get historical balance records",
)
async def get_balance_history(
    coin: Optional[str] = Query(
        default=None,
        description="Optional coin filter (e.g., 'USDT'). If omitted, returns history for all coins.",
    ),
    from_time: Optional[datetime] = Query(
        default=None,
        alias="from",
        description="Start of time range filter (inclusive, event_timestamp).",
    ),
    to_time: Optional[datetime] = Query(
        default=None,
        alias="to",
        description="End of time range filter (inclusive, event_timestamp).",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of records to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Offset for pagination.",
    ),
) -> BalanceHistoryResponse:
    """Return historical balance records with optional coin and time range filters."""
    if from_time and to_time and from_time > to_time:
        raise HTTPException(
            status_code=400,
            detail={"error": "'from' must be earlier than or equal to 'to'"},
        )

    try:
        balances = await BalanceRepository.list_balances(
            coin=coin,
            limit=limit,
            offset=offset,
            start_time=from_time,
            end_time=to_time,
        )
    except Exception as exc:
        logger.error(
            "balances_history_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch balance history")

    # Total is best-effort; for now we report the number of records returned
    from .schemas import BalanceRecord

    records: list[BalanceRecord] = [
        BalanceRecord(
            id=b.id,
            coin=b.coin,
            wallet_balance=b.wallet_balance,
            available_balance=b.available_balance,
            frozen=b.frozen,
            event_timestamp=b.event_timestamp,
            received_at=b.received_at,
            trace_id=b.trace_id,
            equity=b.equity,
            usd_value=b.usd_value,
            margin_collateral=b.margin_collateral,
            total_order_im=b.total_order_im,
            total_position_im=b.total_position_im,
        )
        for b in balances
    ]

    return BalanceHistoryResponse(balances=records, total=len(records))


@router.post(
    "/sync",
    summary="Trigger immediate balance sync from Bybit REST API",
)
async def sync_balances():
    """Placeholder endpoint for triggering balance sync from Bybit REST API.

    NOTE: Full Bybit REST sync implementation will be added in a dedicated service.
    For now this endpoint exists so that clients can integrate and will return
    501 to indicate that server-side sync is not yet implemented.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "error": "Balance sync via REST API not yet implemented",
            "code": "NOT_IMPLEMENTED",
        },
    )


