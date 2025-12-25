"""Order query endpoints."""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...config.database import DatabaseConnection
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


@router.get("/orders")
async def list_orders(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    status: Optional[str] = Query(None, description="Filter by order status"),
    signal_id: Optional[str] = Query(None, description="Filter by trading signal ID"),
    order_id: Optional[str] = Query(None, description="Filter by Bybit order ID"),
    side: Optional[str] = Query(None, description="Filter by order side (Buy, Sell)"),
    date_from: Optional[str] = Query(None, description="Filter orders from date (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="Filter orders until date (ISO 8601)"),
    position_id: Optional[str] = Query(None, description="Filter by position ID (UUID)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort direction (asc, desc)"),
):
    """List orders with filtering, pagination, and sorting."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "order_list_request",
        asset=asset,
        status=status,
        page=page,
        page_size=page_size,
        trace_id=trace_id,
    )

    try:
        # Validate sort_by and sort_order
        valid_sort_fields = {"created_at", "updated_at", "executed_at"}
        if sort_by not in valid_sort_fields:
            raise HTTPException(status_code=400, detail=f"Invalid sort_by. Must be one of {valid_sort_fields}")

        if sort_order.lower() not in {"asc", "desc"}:
            raise HTTPException(status_code=400, detail="Invalid sort_order. Must be 'asc' or 'desc'")

        # Determine JOIN type: use INNER JOIN when filtering by position_id for better performance
        # Otherwise use LEFT JOIN to include orders without position relationships
        join_type = "INNER JOIN" if position_id else "LEFT JOIN"
        
        # Build query with JOIN to position_orders to get position_id
        query = f"""
            SELECT 
                o.id, o.order_id, o.signal_id, o.asset, o.side, o.order_type,
                o.quantity, o.price, o.status, o.filled_quantity, o.average_price,
                o.fees, o.created_at, o.updated_at, o.rejection_reason,
                po.position_id
            FROM orders o
            {join_type} position_orders po ON o.id = po.order_id
            WHERE 1=1
        """
        params = []
        param_idx = 1

        if asset:
            query += f" AND asset = ${param_idx}"
            params.append(asset)
            param_idx += 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        if signal_id:
            query += f" AND signal_id = ${param_idx}::uuid"
            params.append(signal_id)
            param_idx += 1

        if order_id:
            query += f" AND order_id = ${param_idx}"
            params.append(order_id)
            param_idx += 1

        if side:
            query += f" AND side = ${param_idx}"
            params.append(side)
            param_idx += 1

        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                query += f" AND created_at >= ${param_idx}::timestamptz"
                params.append(date_from_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO 8601 format.")

        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                query += f" AND created_at <= ${param_idx}::timestamptz"
                params.append(date_to_dt)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO 8601 format.")

        if position_id:
            query += f" AND po.position_id = ${param_idx}::uuid"
            params.append(position_id)
            param_idx += 1

        # Add sorting
        query += f" ORDER BY {sort_by} {sort_order.upper()}"

        # Add pagination
        offset = (page - 1) * page_size
        query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([page_size, offset])

        rows = await DatabaseConnection.fetch(query, *params)

        # Get total count - use same JOIN type as main query
        count_query = f"""
            SELECT COUNT(*) as count
            FROM orders o
            {join_type} position_orders po ON o.id = po.order_id
            WHERE 1=1
        """
        count_params = []
        count_param_idx = 1

        # Rebuild filters for count query
        if asset:
            count_query += f" AND asset = ${count_param_idx}"
            count_params.append(asset)
            count_param_idx += 1
        if status:
            count_query += f" AND status = ${count_param_idx}"
            count_params.append(status)
            count_param_idx += 1
        if signal_id:
            count_query += f" AND signal_id = ${count_param_idx}::uuid"
            count_params.append(signal_id)
            count_param_idx += 1
        if order_id:
            count_query += f" AND order_id = ${count_param_idx}"
            count_params.append(order_id)
            count_param_idx += 1
        if side:
            count_query += f" AND side = ${count_param_idx}"
            count_params.append(side)
            count_param_idx += 1
        if date_from:
            count_query += f" AND created_at >= ${count_param_idx}::timestamptz"
            count_params.append(date_from_dt)
            count_param_idx += 1
        if date_to:
            count_query += f" AND created_at <= ${count_param_idx}::timestamptz"
            count_params.append(date_to_dt)
            count_param_idx += 1
        if position_id:
            count_query += f" AND po.position_id = ${count_param_idx}::uuid"
            count_params.append(position_id)
            count_param_idx += 1

        total_count = await DatabaseConnection.fetchval(count_query, *count_params)

        orders_data = []
        for row in rows:
            order_dict = {
                "id": str(row["id"]),
                "order_id": row["order_id"],
                "signal_id": str(row["signal_id"]) if row["signal_id"] else None,
                "asset": row["asset"],
                "side": row["side"],
                "order_type": row["order_type"],
                "quantity": str(row["quantity"]),
                "price": str(row["price"]) if row["price"] else None,
                "status": row["status"],
                "filled_quantity": str(row["filled_quantity"]),
                "average_price": str(row["average_price"]) if row["average_price"] else None,
                "fees": str(row["fees"]) if row["fees"] else None,
                "created_at": row["created_at"].isoformat() + "Z",
                "updated_at": row["updated_at"].isoformat() + "Z",
                "rejection_reason": row["rejection_reason"] if row.get("rejection_reason") else None,
                "position_id": str(row["position_id"]) if row["position_id"] else None,
            }
            orders_data.append(order_dict)

        logger.info("order_list_completed", count=len(orders_data), total=total_count, trace_id=trace_id)

        return JSONResponse(
            status_code=200,
            content={
                "orders": orders_data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("order_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {str(e)}")

