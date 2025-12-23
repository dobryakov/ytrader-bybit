"""Position query endpoints."""

from typing import Optional
from decimal import Decimal

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/positions")
async def list_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    mode: Optional[str] = Query(None, description="Filter by trading mode (one-way, hedge)"),
    size_min: Optional[float] = Query(None, description="Minimum position size"),
    size_max: Optional[float] = Query(None, description="Maximum position size"),
    position_id: Optional[str] = Query(None, description="Filter by position ID (UUID)"),
):
    """List positions with optional filtering."""
    trace_id = get_or_create_trace_id()
    logger.info("position_list_request", asset=asset, mode=mode, trace_id=trace_id)

    try:
        # Build query
        query = """
            SELECT 
                id, asset, size, average_entry_price, current_price,
                unrealized_pnl, realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, last_updated, created_at, closed_at
            FROM positions
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        if mode:
            if mode.lower() not in {"one-way", "hedge"}:
                raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")
            query += f" AND mode = ${param_idx}"
            params.append(mode)
            param_idx += 1

        if size_min is not None:
            query += f" AND ABS(size) >= ${param_idx}"
            params.append(size_min)
            param_idx += 1

        if size_max is not None:
            query += f" AND ABS(size) <= ${param_idx}"
            params.append(size_max)
            param_idx += 1

        if position_id:
            query += f" AND id = ${param_idx}::uuid"
            params.append(position_id)
            param_idx += 1

        query += " ORDER BY last_updated DESC"

        rows = await DatabaseConnection.fetch(query, *params)

        positions_data = []
        for row in rows:
            position_dict = {
                "id": str(row["id"]),
                "asset": row["asset"],
                "size": str(row["size"]),
                "average_entry_price": str(row["average_entry_price"]) if row["average_entry_price"] else None,
                "current_price": str(row["current_price"]) if row["current_price"] else None,
                "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else None,
                "mode": row["mode"],
                "long_size": str(row["long_size"]) if row["long_size"] else None,
                "short_size": str(row["short_size"]) if row["short_size"] else None,
                "long_avg_price": str(row["long_avg_price"]) if row["long_avg_price"] else None,
                "short_avg_price": str(row["short_avg_price"]) if row["short_avg_price"] else None,
                "last_updated": row["last_updated"].isoformat() + "Z" if row["last_updated"] else None,
                "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
            }
            positions_data.append(position_dict)

        logger.info("position_list_completed", count=len(positions_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "positions": positions_data,
                "count": len(positions_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve positions: {str(e)}")


@router.get("/positions/{asset}")
async def get_position_by_asset(asset: str):
    """Get position details for specific asset."""
    trace_id = get_or_create_trace_id()
    logger.info("position_get_request", asset=asset, trace_id=trace_id)

    try:
        query = """
            SELECT 
                id, asset, size, average_entry_price, current_price,
                unrealized_pnl, realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, last_updated, created_at, closed_at
            FROM positions
            WHERE asset = $1
            ORDER BY last_updated DESC
            LIMIT 1
        """

        row = await DatabaseConnection.fetchrow(query, asset)

        if not row:
            raise HTTPException(status_code=404, detail=f"Position not found for asset: {asset}")

        position_dict = {
            "id": str(row["id"]),
            "asset": row["asset"],
            "size": str(row["size"]),
            "average_entry_price": str(row["average_entry_price"]) if row["average_entry_price"] else None,
            "current_price": str(row["current_price"]) if row["current_price"] else None,
            "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
            "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else None,
            "mode": row["mode"],
            "long_size": str(row["long_size"]) if row["long_size"] else None,
            "short_size": str(row["short_size"]) if row["short_size"] else None,
            "long_avg_price": str(row["long_avg_price"]) if row["long_avg_price"] else None,
            "short_avg_price": str(row["short_avg_price"]) if row["short_avg_price"] else None,
            "last_updated": row["last_updated"].isoformat() + "Z" if row["last_updated"] else None,
            "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
            "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
        }

        logger.info("position_get_completed", asset=asset, trace_id=trace_id)

        return JSONResponse(status_code=200, content=position_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_get_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position: {str(e)}")

