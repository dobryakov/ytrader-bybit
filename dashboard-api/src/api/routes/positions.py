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
                long_avg_price, short_avg_price, last_updated, created_at, closed_at, opened_at
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
                "opened_at": row["opened_at"].isoformat() + "Z" if row["opened_at"] else None,
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


@router.get("/positions/closed")
async def list_closed_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
):
    """List closed positions history."""
    trace_id = get_or_create_trace_id()
    logger.info("closed_positions_list_request", asset=asset, limit=limit, offset=offset, trace_id=trace_id)

    try:
        # Build query
        query = """
            SELECT 
                id, original_position_id, asset, mode,
                final_size, average_entry_price, exit_price, current_price,
                realized_pnl, unrealized_pnl_at_close,
                long_size, short_size, long_avg_price, short_avg_price,
                total_fees, opened_at, closed_at, version
            FROM closed_positions
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        query += " ORDER BY closed_at DESC"
        query += f" LIMIT ${param_idx}"
        params.append(limit)
        param_idx += 1

        if offset > 0:
            query += f" OFFSET ${param_idx}"
            params.append(offset)

        rows = await DatabaseConnection.fetch(query, *params)

        closed_positions_data = []
        for row in rows:
            # Calculate total_pnl for closed position
            # For a fully closed position, realized_pnl already includes all PnL
            # (unrealized_pnl_at_close became realized when position was closed)
            # Therefore, total_pnl = realized_pnl
            total_pnl = row["realized_pnl"] or Decimal("0")
            
            closed_position_dict = {
                "id": str(row["id"]),
                "original_position_id": str(row["original_position_id"]),
                "asset": row["asset"],
                "mode": row["mode"],
                "final_size": str(row["final_size"]),
                "average_entry_price": str(row["average_entry_price"]) if row["average_entry_price"] else None,
                "exit_price": str(row["exit_price"]) if row["exit_price"] else None,
                "current_price": str(row["current_price"]) if row["current_price"] else None,
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else None,
                "unrealized_pnl_at_close": str(row["unrealized_pnl_at_close"]) if row["unrealized_pnl_at_close"] else None,
                "total_pnl": str(total_pnl),
                "long_size": str(row["long_size"]) if row["long_size"] else None,
                "short_size": str(row["short_size"]) if row["short_size"] else None,
                "long_avg_price": str(row["long_avg_price"]) if row["long_avg_price"] else None,
                "short_avg_price": str(row["short_avg_price"]) if row["short_avg_price"] else None,
                "total_fees": str(row["total_fees"]) if row["total_fees"] else None,
                "opened_at": row["opened_at"].isoformat() + "Z" if row["opened_at"] else None,
                "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
                "version": row["version"],
            }
            closed_positions_data.append(closed_position_dict)

        logger.info("closed_positions_list_completed", count=len(closed_positions_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "closed_positions": closed_positions_data,
                "count": len(closed_positions_data),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("closed_positions_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve closed positions: {str(e)}")


@router.get("/positions/{asset}/orders")
async def get_position_orders(
    asset: str,
    mode: Optional[str] = Query("one-way", description="Trading mode (one-way, hedge)"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type (opened, increased, decreased, closed, reversed)"),
):
    """Get orders for a position with relationship information from position_orders table."""
    trace_id = get_or_create_trace_id()
    logger.info("position_orders_get_request", asset=asset, mode=mode, relationship_type=relationship_type, trace_id=trace_id)

    try:
        mode_lower = mode.lower() if mode else None
        if mode_lower and mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

        # First, get position ID
        position_query = """
            SELECT id, opened_at
            FROM positions
            WHERE asset = $1
        """
        position_params = [asset]
        
        if mode_lower:
            position_query += " AND mode = $2"
            position_params.append(mode_lower)
        
        position_query += " ORDER BY last_updated DESC LIMIT 1"

        position_row = await DatabaseConnection.fetchrow(position_query, *position_params)

        if not position_row:
            raise HTTPException(status_code=404, detail=f"Position not found for asset: {asset}")

        position_id = position_row["id"]
        opened_at = position_row.get("opened_at")

        # Build query to get orders with position_orders relationship info
        # Handle both cases: when order_id is set and when only bybit_order_id is available
        # Filter by opened_at to show only orders from current position cycle (since last opening)
        query = """
            SELECT 
                COALESCE(o.id, o2.id) as id,
                COALESCE(o.order_id, o2.order_id, po.bybit_order_id) as order_id,
                COALESCE(o.signal_id, o2.signal_id) as signal_id,
                COALESCE(o.asset, o2.asset) as asset,
                COALESCE(o.side, o2.side) as side,
                COALESCE(o.order_type, o2.order_type) as order_type,
                COALESCE(o.quantity, o2.quantity) as quantity,
                COALESCE(o.price, o2.price) as price,
                COALESCE(o.status, o2.status) as status,
                COALESCE(o.filled_quantity, o2.filled_quantity) as filled_quantity,
                COALESCE(o.average_price, o2.average_price) as average_price,
                COALESCE(o.fees, o2.fees) as fees,
                COALESCE(o.created_at, o2.created_at) as created_at,
                COALESCE(o.updated_at, o2.updated_at) as updated_at,
                COALESCE(o.executed_at, o2.executed_at) as executed_at,
                po.relationship_type, po.size_delta, po.execution_price, po.executed_at as po_executed_at,
                po.bybit_order_id
            FROM position_orders po
            LEFT JOIN orders o ON o.id = po.order_id
            LEFT JOIN orders o2 ON po.order_id IS NULL AND o2.order_id = po.bybit_order_id
            WHERE po.position_id = $1
              AND (po.order_id IS NOT NULL OR po.bybit_order_id IS NOT NULL)
        """
        params = [position_id]
        param_idx = 2

        # Filter by opened_at to show only orders from current position cycle
        # Show only orders executed at or after the last opening/reopening (opened_at)
        # This excludes historical orders from previous position cycles
        if opened_at:
            # Include orders executed at or after opened_at (with 1 minute window before for safety)
            # to catch orders that opened the position
            query += f" AND po.executed_at >= ${param_idx}::timestamptz - INTERVAL '1 minute'"
            params.append(opened_at)
            param_idx += 1
        else:
            # If opened_at is NULL, position might be old or never properly opened
            # In this case, show only orders with relationship_type 'opened' or 'reversed'
            # to avoid showing all historical orders
            query += f" AND po.relationship_type IN ('opened', 'reversed')"

        if relationship_type:
            valid_types = {"opened", "increased", "decreased", "closed", "reversed"}
            if relationship_type.lower() not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid relationship_type. Must be one of: {', '.join(valid_types)}"
                )
            query += f" AND po.relationship_type = ${param_idx}"
            params.append(relationship_type.lower())
            param_idx += 1

        query += " ORDER BY po.executed_at DESC"

        rows = await DatabaseConnection.fetch(query, *params)

        orders_data = []
        for row in rows:
            # Handle case when order is not yet created in DB (order_id is NULL)
            # In this case, use data from position_orders
            order_id = row.get("order_id") or row.get("bybit_order_id")
            if not order_id:
                continue  # Skip if no order identifier available
            
            order_dict = {
                "id": str(row["id"]) if row["id"] else None,
                "order_id": order_id,
                "bybit_order_id": row.get("bybit_order_id"),
                "signal_id": str(row["signal_id"]) if row["signal_id"] else None,
                "asset": row["asset"] or asset,  # Fallback to asset from URL if not in order
                "side": row["side"] or "Unknown",
                "order_type": row["order_type"] or "Market",
                "quantity": str(row["quantity"]) if row["quantity"] else "0",
                "price": str(row["price"]) if row["price"] else None,
                "status": row["status"] or "unknown",
                "filled_quantity": str(row["filled_quantity"]) if row["filled_quantity"] else str(row.get("size_delta", "0")),
                "average_price": str(row["average_price"]) if row["average_price"] else str(row["execution_price"]),
                "fees": str(row["fees"]) if row["fees"] else None,
                "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() + "Z" if row["updated_at"] else None,
                "executed_at": row["executed_at"].isoformat() + "Z" if row["executed_at"] else None,
                "relationship_type": row["relationship_type"],
                "size_delta": str(row["size_delta"]),
                "execution_price": str(row["execution_price"]),
                "po_executed_at": row["po_executed_at"].isoformat() + "Z" if row["po_executed_at"] else None,
            }
            orders_data.append(order_dict)

        logger.info("position_orders_get_completed", asset=asset, count=len(orders_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "orders": orders_data,
                "count": len(orders_data),
                "position_id": str(position_id),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_orders_get_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position orders: {str(e)}")


@router.get("/positions/{asset}")
async def get_position_by_asset(
    asset: str,
    mode: Optional[str] = Query("one-way", description="Trading mode (one-way, hedge)"),
):
    """Get position details for specific asset."""
    trace_id = get_or_create_trace_id()
    logger.info("position_get_request", asset=asset, mode=mode, trace_id=trace_id)

    try:
        mode_lower = mode.lower() if mode else None
        if mode_lower and mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

        query = """
            SELECT 
                id, asset, size, average_entry_price, current_price,
                unrealized_pnl, realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, last_updated, created_at, closed_at, opened_at
            FROM positions
            WHERE asset = $1
        """
        params = [asset]
        
        if mode_lower:
            query += " AND mode = $2"
            params.append(mode_lower)
        
        query += " ORDER BY last_updated DESC LIMIT 1"

        row = await DatabaseConnection.fetchrow(query, *params)

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
            "opened_at": row["opened_at"].isoformat() + "Z" if row["opened_at"] else None,
        }

        logger.info("position_get_completed", asset=asset, trace_id=trace_id)

        return JSONResponse(status_code=200, content=position_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_get_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position: {str(e)}")

