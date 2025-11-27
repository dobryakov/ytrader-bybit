"""Manual synchronization endpoints for order state synchronization."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...config.logging import get_logger
from ...services.order_state_sync import OrderStateSync
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)

router = APIRouter()


class SyncRequest(BaseModel):
    """Request model for manual synchronization."""

    asset: Optional[str] = None
    scope: str = "active"  # 'all' or 'active'


@router.post("/sync")
async def trigger_sync(request: SyncRequest = Body(default=SyncRequest())):
    """Trigger manual order state synchronization with Bybit exchange.

    Args:
        request: Sync request with optional asset filter and scope

    Returns:
        Synchronization results
    """
    trace_id = get_or_create_trace_id()
    logger.info(
        "sync_request",
        asset=request.asset,
        scope=request.scope,
        trace_id=trace_id,
    )

    try:
        # Validate scope
        if request.scope not in {"all", "active"}:
            raise HTTPException(status_code=400, detail="Invalid scope. Must be 'all' or 'active'")

        # Currently, we only support 'active' scope (sync_active_orders)
        # 'all' scope would require additional implementation
        if request.scope != "active":
            # For now, return an error if 'all' is requested
            raise HTTPException(
                status_code=501,
                detail="'all' scope synchronization is not yet implemented. Use 'active' scope.",
            )

        # Perform synchronization
        sync_service = OrderStateSync()
        result = await sync_service.sync_active_orders(trace_id=trace_id)

        # Determine status
        errors = result.get("errors", [])
        discrepancies = result.get("discrepancies", [])
        synced_count = result.get("synced_count", 0)
        updated_orders = len(discrepancies)

        if errors and not updated_orders:
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "completed"

        logger.info(
            "sync_completed",
            status=status,
            synced_count=synced_count,
            updated_orders=updated_orders,
            errors_count=len(errors),
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "synced_orders": synced_count,
                "updated_orders": updated_orders,
                "errors": errors,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("sync_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synchronization failed: {e}") from e

