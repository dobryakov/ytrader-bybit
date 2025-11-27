"""Position query endpoints for retrieving position information."""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...models.position import Position
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)

router = APIRouter()


@router.get("/positions")
async def list_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    mode: Optional[str] = Query(None, description="Filter by trading mode (one-way, hedge)"),
):
    """List positions with optional filtering.

    Args:
        asset: Trading pair filter (e.g., BTCUSDT)
        mode: Trading mode filter (one-way, hedge)

    Returns:
        List of positions
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_list_request", asset=asset, mode=mode, trace_id=trace_id)

    try:
        # Validate mode if provided
        if mode is not None:
            mode_lower = mode.lower()
            if mode_lower not in {"one-way", "hedge"}:
                raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

        pool = await DatabaseConnection.get_pool()

        # Build query
        where_conditions = []
        params = []
        param_idx = 1

        if asset:
            where_conditions.append(f"asset = ${param_idx}")
            params.append(asset.upper())
            param_idx += 1

        if mode:
            where_conditions.append(f"mode = ${param_idx}")
            params.append(mode.lower())
            param_idx += 1

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        query = f"""
            SELECT id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                   mode, long_size, short_size, long_avg_price, short_avg_price,
                   last_updated, last_snapshot_at
            FROM positions
            WHERE {where_clause}
            ORDER BY asset, mode
        """

        rows = await pool.fetch(query, *params)

        # Convert to Position objects
        positions = [Position.from_dict(dict(row)) for row in rows]

        # Serialize positions to JSON
        positions_data = []
        for position in positions:
            position_dict = {
                "id": str(position.id),
                "asset": position.asset,
                "size": str(position.size),
                "average_entry_price": str(position.average_entry_price) if position.average_entry_price is not None else None,
                "unrealized_pnl": str(position.unrealized_pnl) if position.unrealized_pnl is not None else None,
                "realized_pnl": str(position.realized_pnl) if position.realized_pnl is not None else None,
                "mode": position.mode,
                "long_size": str(position.long_size) if position.long_size is not None else None,
                "short_size": str(position.short_size) if position.short_size is not None else None,
                "long_avg_price": str(position.long_avg_price) if position.long_avg_price is not None else None,
                "short_avg_price": str(position.short_avg_price) if position.short_avg_price is not None else None,
                "last_updated": position.last_updated.isoformat() + "Z",
                "last_snapshot_at": position.last_snapshot_at.isoformat() + "Z" if position.last_snapshot_at else None,
            }
            positions_data.append(position_dict)

        logger.info("position_list_completed", count=len(positions), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "positions": positions_data,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve positions: {e}") from e


@router.get("/positions/{asset}")
async def get_position_by_asset(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
):
    """Get position for a specific asset.

    Args:
        asset: Trading pair symbol (e.g., BTCUSDT)
        mode: Trading mode (one-way, hedge)

    Returns:
        Position details
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_get_request", asset=asset, mode=mode, trace_id=trace_id)

    try:
        # Validate mode
        mode_lower = mode.lower()
        if mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

        pool = await DatabaseConnection.get_pool()
        query = """
            SELECT id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                   mode, long_size, short_size, long_avg_price, short_avg_price,
                   last_updated, last_snapshot_at
            FROM positions
            WHERE asset = $1 AND mode = $2
        """
        row = await pool.fetchrow(query, asset.upper(), mode_lower)

        if row is None:
            logger.warning("position_not_found", asset=asset, mode=mode, trace_id=trace_id)
            raise HTTPException(status_code=404, detail=f"Position not found for asset: {asset}, mode: {mode}")

        position = Position.from_dict(dict(row))

        # Serialize position to JSON
        position_dict = {
            "id": str(position.id),
            "asset": position.asset,
            "size": str(position.size),
            "average_entry_price": str(position.average_entry_price) if position.average_entry_price is not None else None,
            "unrealized_pnl": str(position.unrealized_pnl) if position.unrealized_pnl is not None else None,
            "realized_pnl": str(position.realized_pnl) if position.realized_pnl is not None else None,
            "mode": position.mode,
            "long_size": str(position.long_size) if position.long_size is not None else None,
            "short_size": str(position.short_size) if position.short_size is not None else None,
            "long_avg_price": str(position.long_avg_price) if position.long_avg_price is not None else None,
            "short_avg_price": str(position.short_avg_price) if position.short_avg_price is not None else None,
            "last_updated": position.last_updated.isoformat() + "Z",
            "last_snapshot_at": position.last_snapshot_at.isoformat() + "Z" if position.last_snapshot_at else None,
        }

        logger.info("position_get_completed", asset=asset, mode=mode, trace_id=trace_id)

        return JSONResponse(status_code=200, content=position_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_get_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position: {e}") from e

