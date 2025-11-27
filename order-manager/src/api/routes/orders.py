"""Order query endpoints for retrieving order information."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...models.order import Order
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)

router = APIRouter()


def _parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    """Parse decimal string to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(value)
    except (ValueError, TypeError):
        return None


@router.get("/orders")
async def list_orders(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    status: Optional[str] = Query(None, description="Filter by order status"),
    signal_id: Optional[str] = Query(None, description="Filter by trading signal ID"),
    order_id: Optional[str] = Query(None, description="Filter by Bybit order ID"),
    side: Optional[str] = Query(None, description="Filter by order side (Buy, Sell)"),
    date_from: Optional[str] = Query(None, description="Filter orders from date (ISO 8601)"),
    date_to: Optional[str] = Query(None, description="Filter orders until date (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
):
    """List orders with filtering, pagination, and sorting.

    Args:
        asset: Trading pair filter (e.g., BTCUSDT)
        status: Order status filter
        signal_id: Trading signal ID filter
        order_id: Bybit order ID filter
        side: Order side filter (Buy, Sell)
        date_from: Filter orders created from this date
        date_to: Filter orders created until this date
        page: Page number for pagination
        page_size: Number of items per page
        sort_by: Field to sort by (created_at, updated_at, executed_at)
        sort_order: Sort order (asc, desc)

    Returns:
        List of orders with pagination information
    """
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
        # Validate status if provided
        if status is not None:
            valid_statuses = {"pending", "partially_filled", "filled", "cancelled", "rejected", "dry_run"}
            if status.lower() not in valid_statuses:
                raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

        # Validate side if provided
        if side is not None:
            side_upper = side.upper()
            if side_upper not in {"BUY", "SELL"}:
                raise HTTPException(status_code=400, detail="Invalid side. Must be 'Buy' or 'Sell'")

        # Validate sort_by
        valid_sort_fields = {"created_at", "updated_at", "executed_at"}
        if sort_by not in valid_sort_fields:
            raise HTTPException(status_code=400, detail=f"Invalid sort_by. Must be one of: {', '.join(valid_sort_fields)}")

        # Validate sort_order
        sort_order_lower = sort_order.lower()
        if sort_order_lower not in {"asc", "desc"}:
            raise HTTPException(status_code=400, detail="Invalid sort_order. Must be 'asc' or 'desc'")

        # Parse dates if provided
        parsed_date_from = None
        parsed_date_to = None
        if date_from:
            try:
                parsed_date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO 8601 format.")
        if date_to:
            try:
                parsed_date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO 8601 format.")

        # Build query
        pool = await DatabaseConnection.get_pool()

        # Build WHERE clause
        where_conditions = []
        params = []
        param_idx = 1

        if asset:
            where_conditions.append(f"asset = ${param_idx}")
            params.append(asset.upper())
            param_idx += 1

        if status:
            where_conditions.append(f"status = ${param_idx}")
            params.append(status.lower())
            param_idx += 1

        if signal_id:
            try:
                UUID(signal_id)  # Validate UUID format
                where_conditions.append(f"signal_id = ${param_idx}")
                params.append(signal_id)
                param_idx += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid signal_id format. Must be a valid UUID.")

        if order_id:
            where_conditions.append(f"order_id = ${param_idx}")
            params.append(order_id)
            param_idx += 1

        if side:
            where_conditions.append(f"side = ${param_idx}")
            params.append(side.upper())
            param_idx += 1

        if parsed_date_from:
            where_conditions.append(f"created_at >= ${param_idx}")
            params.append(parsed_date_from)
            param_idx += 1

        if parsed_date_to:
            where_conditions.append(f"created_at <= ${param_idx}")
            params.append(parsed_date_to)
            param_idx += 1

        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

        # Build ORDER BY clause
        order_clause = f"ORDER BY {sort_by} {sort_order.upper()}"

        # Count total items
        count_query = f"SELECT COUNT(*) FROM orders WHERE {where_clause}"
        total_items = await pool.fetchval(count_query, *params)

        # Calculate pagination
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 0
        offset = (page - 1) * page_size

        # Fetch orders
        query = f"""
            SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                   status, filled_quantity, average_price, fees, created_at, updated_at,
                   executed_at, trace_id, is_dry_run
            FROM orders
            WHERE {where_clause}
            {order_clause}
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([page_size, offset])

        rows = await pool.fetch(query, *params)

        # Convert to Order objects
        orders = [Order.from_dict(dict(row)) for row in rows]

        # Serialize orders to JSON
        orders_data = []
        for order in orders:
            order_dict = {
                "id": str(order.id),
                "order_id": order.order_id,
                "signal_id": str(order.signal_id),
                "asset": order.asset,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": str(order.quantity),
                "price": str(order.price) if order.price is not None else None,
                "status": order.status,
                "filled_quantity": str(order.filled_quantity),
                "average_price": str(order.average_price) if order.average_price is not None else None,
                "fees": str(order.fees) if order.fees is not None else None,
                "created_at": order.created_at.isoformat() + "Z",
                "updated_at": order.updated_at.isoformat() + "Z",
                "executed_at": order.executed_at.isoformat() + "Z" if order.executed_at else None,
                "trace_id": order.trace_id,
                "is_dry_run": order.is_dry_run,
            }
            orders_data.append(order_dict)

        logger.info(
            "order_list_completed",
            total_items=total_items,
            returned_items=len(orders),
            page=page,
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "orders": orders_data,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_items": total_items,
                    "total_pages": total_pages,
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("order_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve orders: {e}") from e


@router.get("/orders/{order_id}")
async def get_order_by_id(order_id: str):
    """Get order by Bybit order ID.

    Args:
        order_id: Bybit order ID

    Returns:
        Order details
    """
    trace_id = get_or_create_trace_id()
    logger.info("order_get_request", order_id=order_id, trace_id=trace_id)

    try:
        pool = await DatabaseConnection.get_pool()
        query = """
            SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                   status, filled_quantity, average_price, fees, created_at, updated_at,
                   executed_at, trace_id, is_dry_run
            FROM orders
            WHERE order_id = $1
        """
        row = await pool.fetchrow(query, order_id)

        if row is None:
            logger.warning("order_not_found", order_id=order_id, trace_id=trace_id)
            raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")

        order = Order.from_dict(dict(row))

        # Serialize order to JSON
        order_dict = {
            "id": str(order.id),
            "order_id": order.order_id,
            "signal_id": str(order.signal_id),
            "asset": order.asset,
            "side": order.side,
            "order_type": order.order_type,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price is not None else None,
            "status": order.status,
            "filled_quantity": str(order.filled_quantity),
            "average_price": str(order.average_price) if order.average_price is not None else None,
            "fees": str(order.fees) if order.fees is not None else None,
            "created_at": order.created_at.isoformat() + "Z",
            "updated_at": order.updated_at.isoformat() + "Z",
            "executed_at": order.executed_at.isoformat() + "Z" if order.executed_at else None,
            "trace_id": order.trace_id,
            "is_dry_run": order.is_dry_run,
        }

        logger.info("order_get_completed", order_id=order_id, trace_id=trace_id)

        return JSONResponse(status_code=200, content=order_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("order_get_failed", order_id=order_id, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve order: {e}") from e

