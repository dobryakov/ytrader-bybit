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
    include_closed: bool = Query(False, description="Include closed positions (default: only active)"),
    closed_only: bool = Query(False, description="Return only closed positions"),
):
    """List positions with optional filtering.
    
    By default, returns only active positions (closed_at IS NULL).
    Use include_closed=true to get all positions, or closed_only=true for only closed positions.
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_list_request", asset=asset, mode=mode, include_closed=include_closed, closed_only=closed_only, trace_id=trace_id)

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

        # Filter by closed_at status
        if closed_only:
            query += f" AND closed_at IS NOT NULL"
        elif not include_closed:
            # Default: only active positions
            query += f" AND closed_at IS NULL"

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

        # Order by: closed positions by closed_at DESC, active by last_updated DESC
        if closed_only or include_closed:
            query += " ORDER BY CASE WHEN closed_at IS NULL THEN 0 ELSE 1 END, COALESCE(closed_at, last_updated) DESC"
        else:
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


@router.get("/positions/closed")
async def list_closed_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
):
    """List closed positions history from unified positions table."""
    trace_id = get_or_create_trace_id()
    logger.info("closed_positions_list_request", asset=asset, limit=limit, offset=offset, trace_id=trace_id)

    try:
        # Build query for closed positions from unified positions table
        query = """
            SELECT 
                id, asset, mode,
                size, average_entry_price, exit_price, current_price,
                realized_pnl, unrealized_pnl,
                long_size, short_size, long_avg_price, short_avg_price,
                total_fees, opening_fees, closing_fees,
                closed_at, version,
                created_at, last_updated
            FROM positions
            WHERE closed_at IS NOT NULL
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset.upper())
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
            # (unrealized_pnl became realized when position was closed)
            # Therefore, total_pnl = realized_pnl
            total_pnl = row["realized_pnl"] or Decimal("0")
            
            closed_position_dict = {
                "id": str(row["id"]),
                "asset": row["asset"],
                "mode": row["mode"],
                "size": str(row["size"]),  # Always 0 for closed positions
                "average_entry_price": str(row["average_entry_price"]) if row["average_entry_price"] else None,
                "exit_price": str(row["exit_price"]) if row["exit_price"] else None,
                "current_price": str(row["current_price"]) if row["current_price"] else None,
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else None,
                "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
                "total_pnl": str(total_pnl),
                "long_size": str(row["long_size"]) if row["long_size"] else None,
                "short_size": str(row["short_size"]) if row["short_size"] else None,
                "long_avg_price": str(row["long_avg_price"]) if row["long_avg_price"] else None,
                "short_avg_price": str(row["short_avg_price"]) if row["short_avg_price"] else None,
                "total_fees": str(row["total_fees"]) if row["total_fees"] else None,
                "opening_fees": str(row["opening_fees"]) if row["opening_fees"] else None,
                "closing_fees": str(row["closing_fees"]) if row["closing_fees"] else None,
                "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
                "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                "last_updated": row["last_updated"].isoformat() + "Z" if row["last_updated"] else None,
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


@router.get("/positions/{position_id}")
async def get_position_by_id(
    position_id: str,
):
    """Get position details by position ID (active or closed).
    
    This endpoint must be registered BEFORE /positions/{asset} to avoid route conflicts.
    FastAPI matches routes in order, so more specific routes should come first.
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_get_by_id_request", position_id=position_id, trace_id=trace_id)

    try:
        from uuid import UUID
        try:
            pos_uuid = UUID(position_id)
        except ValueError:
            # Not a valid UUID, let it fall through to other routes
            raise HTTPException(status_code=404, detail=f"Position not found for ID: {position_id}")

        query = """
            SELECT 
                id, asset, size, average_entry_price, current_price,
                unrealized_pnl, realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, last_updated, created_at, closed_at
            FROM positions
            WHERE id = $1
        """
        params = [pos_uuid]

        row = await DatabaseConnection.fetchrow(query, *params)

        if not row:
            raise HTTPException(status_code=404, detail=f"Position not found for ID: {position_id}")

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

        logger.info("position_get_by_id_completed", position_id=position_id, trace_id=trace_id)

        return JSONResponse(status_code=200, content=position_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_get_by_id_failed", position_id=position_id, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position: {str(e)}")


@router.get("/positions/{position_id}/orders")
async def get_position_orders_by_id(
    position_id: str,
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type (opened, increased, decreased, closed, reversed)"),
):
    """Get orders for a position by position ID with relationship information.
    
    This endpoint must be registered BEFORE /positions/{asset}/orders to avoid route conflicts.
    FastAPI matches routes in order, so more specific routes should come first.
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_orders_get_by_id_request", position_id=position_id, relationship_type=relationship_type, trace_id=trace_id)

    try:
        from uuid import UUID
        try:
            pos_uuid = UUID(position_id)
        except ValueError:
            # Not a valid UUID, let it fall through to other routes
            raise HTTPException(status_code=404, detail=f"Position not found for ID: {position_id}")

        # Get position details
        position_query = """
            SELECT id, asset, mode, created_at, closed_at
            FROM positions
            WHERE id = $1
        """
        position_row = await DatabaseConnection.fetchrow(position_query, pos_uuid)

        if not position_row:
            raise HTTPException(status_code=404, detail=f"Position not found for ID: {position_id}")

        asset = position_row["asset"]
        mode = position_row["mode"]
        created_at = position_row.get("created_at")
        closed_at = position_row.get("closed_at")

        # Build query to get orders with position_orders relationship info
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
        params = [pos_uuid]
        param_idx = 2

        if created_at:
            query += f" AND po.executed_at >= ${param_idx}::timestamptz - INTERVAL '1 minute'"
            params.append(created_at)
            param_idx += 1

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

        # If no orders found via position_orders, try to find orders directly by time window
        if not rows and created_at:
            logger.info(
                "position_orders_fallback_to_direct_search",
                position_id=position_id,
                asset=asset,
                created_at=created_at.isoformat() if created_at else None,
                trace_id=trace_id,
            )
            
            # Search for orders executed around position creation time (±5 minutes before, up to closed_at or now)
            from datetime import datetime, timezone
            end_time = closed_at if closed_at else datetime.now(timezone.utc)
            fallback_query = """
                SELECT 
                    o.id,
                    o.order_id,
                    o.signal_id,
                    o.asset,
                    o.side,
                    o.order_type,
                    o.quantity,
                    o.price,
                    o.status,
                    o.filled_quantity,
                    o.average_price,
                    o.fees,
                    o.created_at,
                    o.updated_at,
                    o.executed_at,
                    CASE 
                        WHEN o.executed_at <= $2::timestamptz + INTERVAL '1 minute' THEN 'opened'
                        WHEN o.side = 'SELL' AND $3::timestamptz IS NOT NULL THEN 'closed'
                        ELSE 'increased'
                    END as relationship_type,
                    o.filled_quantity as size_delta,
                    o.average_price as execution_price,
                    o.executed_at as po_executed_at,
                    o.order_id as bybit_order_id
                FROM orders o
                WHERE o.asset = $1
                  AND o.status = 'filled'
                  AND o.executed_at >= $2::timestamptz - INTERVAL '5 minutes'
                  AND o.executed_at <= $4::timestamptz
                ORDER BY o.executed_at ASC
                LIMIT 100
            """
            fallback_params = [asset.upper(), created_at, closed_at, end_time]
            rows = await DatabaseConnection.fetch(fallback_query, *fallback_params)
            
            if rows:
                logger.info(
                    "position_orders_fallback_found_orders",
                    position_id=position_id,
                    asset=asset,
                    count=len(rows),
                    trace_id=trace_id,
                )

        orders_data = []
        for row in rows:
            order_id = row.get("order_id") or row.get("bybit_order_id")
            if not order_id:
                continue
            
            order_dict = {
                "id": str(row["id"]) if row["id"] else None,
                "order_id": order_id,
                "bybit_order_id": row.get("bybit_order_id"),
                "signal_id": str(row["signal_id"]) if row["signal_id"] else None,
                "asset": row["asset"] or asset,
                "side": row["side"] or "Unknown",
                "order_type": row["order_type"] or "Market",
                "quantity": str(row["quantity"]) if row["quantity"] else "0",
                "price": str(row["price"]) if row["price"] else None,
                "status": row["status"] or "unknown",
                "filled_quantity": str(row["filled_quantity"]) if row["filled_quantity"] else str(row.get("size_delta", "0")),
                "average_price": str(row["average_price"]) if row["average_price"] else str(row.get("execution_price", "0")),
                "fees": str(row["fees"]) if row["fees"] else None,
                "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() + "Z" if row["updated_at"] else None,
                "executed_at": row["executed_at"].isoformat() + "Z" if row["executed_at"] else None,
                "relationship_type": row["relationship_type"],
                "size_delta": str(row["size_delta"]),
                "execution_price": str(row.get("execution_price", row.get("average_price", "0"))),
                "po_executed_at": row["po_executed_at"].isoformat() + "Z" if row["po_executed_at"] else None,
            }
            orders_data.append(order_dict)

        logger.info("position_orders_get_by_id_completed", position_id=position_id, count=len(orders_data), trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "orders": orders_data,
                "count": len(orders_data),
                "position_id": position_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_orders_get_by_id_failed", position_id=position_id, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position orders: {str(e)}")


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
            SELECT id, created_at
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
        created_at = position_row.get("created_at")

        # Build query to get orders with position_orders relationship info
        # Handle both cases: when order_id is set and when only bybit_order_id is available
        # Filter by created_at to show only orders from current position cycle (since position creation)
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

        # Filter by created_at to show only orders from current position cycle
        # Show only orders executed at or after position creation (created_at)
        # This excludes historical orders from previous position cycles
        if created_at:
            # Include orders executed at or after created_at (with 1 minute window before for safety)
            # to catch orders that opened the position
            query += f" AND po.executed_at >= ${param_idx}::timestamptz - INTERVAL '1 minute'"
            params.append(created_at)
            param_idx += 1
        else:
            # If created_at is NULL, position might be old
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

        # If no orders found via position_orders, try to find orders directly by time window
        # This is a fallback for cases where position_orders relationships weren't created
        if not rows and created_at:
            logger.info(
                "position_orders_fallback_to_direct_search",
                position_id=str(position_id),
                asset=asset,
                created_at=created_at.isoformat() if created_at else None,
                trace_id=trace_id,
            )
            
            # Search for orders executed around position creation time (±5 minutes)
            fallback_query = """
                SELECT 
                    o.id,
                    o.order_id,
                    o.signal_id,
                    o.asset,
                    o.side,
                    o.order_type,
                    o.quantity,
                    o.price,
                    o.status,
                    o.filled_quantity,
                    o.average_price,
                    o.fees,
                    o.created_at,
                    o.updated_at,
                    o.executed_at,
                    'opened' as relationship_type,
                    o.filled_quantity as size_delta,
                    o.average_price as execution_price,
                    o.executed_at as po_executed_at,
                    o.order_id as bybit_order_id
                FROM orders o
                WHERE o.asset = $1
                  AND o.status = 'filled'
                  AND o.executed_at >= $2::timestamptz - INTERVAL '5 minutes'
                  AND o.executed_at <= $2::timestamptz + INTERVAL '30 minutes'
                ORDER BY o.executed_at ASC
                LIMIT 100
            """
            fallback_params = [asset.upper(), created_at]
            rows = await DatabaseConnection.fetch(fallback_query, *fallback_params)
            
            if rows:
                logger.info(
                    "position_orders_fallback_found_orders",
                    position_id=str(position_id),
                    asset=asset,
                    count=len(rows),
                    trace_id=trace_id,
                )

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


@router.get("/positions/{asset}/history")
async def get_position_history(
    asset: str,
    mode: Optional[str] = Query("one-way", description="Trading mode (one-way, hedge)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
):
    """Get history of closed positions for a specific asset/mode pair."""
    trace_id = get_or_create_trace_id()
    logger.info("position_history_request", asset=asset, mode=mode, limit=limit, offset=offset, trace_id=trace_id)

    try:
        mode_lower = mode.lower() if mode else None
        if mode_lower and mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

        query = """
            SELECT 
                id, asset, size, average_entry_price, current_price, exit_price,
                unrealized_pnl, realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, total_fees, opening_fees, closing_fees,
                last_updated, created_at, closed_at
            FROM positions
            WHERE asset = $1 AND closed_at IS NOT NULL
        """
        params = [asset.upper()]
        param_idx = 2
        
        if mode_lower:
            query += f" AND mode = ${param_idx}"
            params.append(mode_lower)
            param_idx += 1

        query += " ORDER BY closed_at DESC"
        query += f" LIMIT ${param_idx}"
        params.append(limit)
        param_idx += 1

        if offset > 0:
            query += f" OFFSET ${param_idx}"
            params.append(offset)

        rows = await DatabaseConnection.fetch(query, *params)

        positions_data = []
        for row in rows:
            position_dict = {
                "id": str(row["id"]),
                "asset": row["asset"],
                "size": str(row["size"]),
                "average_entry_price": str(row["average_entry_price"]) if row["average_entry_price"] else None,
                "exit_price": str(row["exit_price"]) if row["exit_price"] else None,
                "current_price": str(row["current_price"]) if row["current_price"] else None,
                "unrealized_pnl": str(row["unrealized_pnl"]) if row["unrealized_pnl"] else None,
                "realized_pnl": str(row["realized_pnl"]) if row["realized_pnl"] else None,
                "mode": row["mode"],
                "long_size": str(row["long_size"]) if row["long_size"] else None,
                "short_size": str(row["short_size"]) if row["short_size"] else None,
                "long_avg_price": str(row["long_avg_price"]) if row["long_avg_price"] else None,
                "short_avg_price": str(row["short_avg_price"]) if row["short_avg_price"] else None,
                "total_fees": str(row["total_fees"]) if row["total_fees"] else None,
                "opening_fees": str(row["opening_fees"]) if row["opening_fees"] else None,
                "closing_fees": str(row["closing_fees"]) if row["closing_fees"] else None,
                "last_updated": row["last_updated"].isoformat() + "Z" if row["last_updated"] else None,
                "created_at": row["created_at"].isoformat() + "Z" if row["created_at"] else None,
                "closed_at": row["closed_at"].isoformat() + "Z" if row["closed_at"] else None,
            }
            positions_data.append(position_dict)

        logger.info("position_history_completed", asset=asset, count=len(positions_data), trace_id=trace_id)

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
        logger.error("position_history_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve position history: {str(e)}")


@router.get("/positions/{asset}")
async def get_position_by_asset(
    asset: str,
    mode: Optional[str] = Query("one-way", description="Trading mode (one-way, hedge)"),
):
    """Get active position details for specific asset (returns only active position, closed_at IS NULL)."""
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
                long_avg_price, short_avg_price, last_updated, created_at, closed_at
            FROM positions
            WHERE asset = $1 AND closed_at IS NULL
        """
        params = [asset]
        
        if mode_lower:
            query += " AND mode = $2"
            params.append(mode_lower)
        
        query += " ORDER BY last_updated DESC LIMIT 1"

        row = await DatabaseConnection.fetchrow(query, *params)

        if not row:
            raise HTTPException(status_code=404, detail=f"Active position not found for asset: {asset}")

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

