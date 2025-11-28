"""Order state synchronization service for startup reconciliation and manual sync."""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..models.order import Order
from ..utils.bybit_client import BybitClient, get_bybit_client
from ..exceptions import DatabaseError, BybitAPIError
from ..utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)


class OrderStateSync:
    """Service for synchronizing order state with Bybit exchange."""

    def __init__(self):
        """Initialize order state sync service."""
        self.bybit_client = get_bybit_client()

    async def sync_active_orders(self, trace_id: Optional[str] = None) -> dict:
        """
        Synchronize active orders from Bybit with database.

        Queries all active orders from Bybit API and compares with database,
        updating any discrepancies. Also checks all pending/partially_filled orders
        from database that are not in active list to detect filled/cancelled orders
        that may have been missed during downtime.

        Args:
            trace_id: Optional trace ID for request tracking

        Returns:
            Dictionary with sync results (synced_count, discrepancies, errors)
        """
        trace_id = trace_id or get_or_create_trace_id()
        synced_count = 0
        discrepancies = []
        errors = []

        try:
            logger.info(
                "order_sync_started",
                trace_id=trace_id,
            )

            # Query active orders from Bybit
            bybit_orders = await self._query_active_orders_from_bybit(trace_id)

            # Query active orders from database
            db_orders = await self._query_active_orders_from_db(trace_id)

            # Create lookup maps
            bybit_orders_map = {order["orderId"]: order for order in bybit_orders}
            db_orders_map = {order.order_id: order for order in db_orders}

            # Find discrepancies for orders that exist in both
            all_order_ids = set(bybit_orders_map.keys()) | set(db_orders_map.keys())

            for order_id in all_order_ids:
                bybit_order = bybit_orders_map.get(order_id)
                db_order = db_orders_map.get(order_id)

                if bybit_order and not db_order:
                    # Order exists in Bybit but not in database - create it
                    errors.append(f"Order {order_id} exists in Bybit but not in database (orphaned)")
                    logger.warning(
                        "sync_orphaned_order",
                        order_id=order_id,
                        trace_id=trace_id,
                    )
                elif bybit_order and db_order:
                    # Compare states for active orders
                    discrepancy = await self._compare_and_sync_order(
                        bybit_order, db_order, trace_id
                    )
                    if discrepancy:
                        discrepancies.append(discrepancy)
                        synced_count += 1

            # Critical: Check all pending/partially_filled orders from DB that are NOT in active Bybit list
            # These orders may have been filled or cancelled while services were down
            db_orders_not_in_active = [
                db_order
                for db_order in db_orders
                if db_order.order_id not in bybit_orders_map
            ]

            if db_orders_not_in_active:
                logger.info(
                    "checking_missed_orders",
                    count=len(db_orders_not_in_active),
                    trace_id=trace_id,
                    message="Checking pending orders that are not in active list - they may have been filled/cancelled during downtime"
                )

                for db_order in db_orders_not_in_active:
                    try:
                        # Query specific order from Bybit to get its current status
                        # This will return the order even if it's filled/cancelled
                        synced_order = await self.sync_order_by_id(
                            db_order.order_id, trace_id=trace_id
                        )

                        if synced_order is None:
                            # Order not found in Bybit - likely cancelled or expired
                            if db_order.status in ["pending", "partially_filled"]:
                                await self._update_order_status(
                                    db_order.id,
                                    "cancelled",
                                    trace_id=trace_id,
                                )
                                discrepancies.append(
                                    {
                                        "order_id": db_order.order_id,
                                        "issue": "order_not_found_in_bybit",
                                        "action": "marked_cancelled",
                                    }
                                )
                                synced_count += 1
                                logger.info(
                                    "order_marked_cancelled_not_found",
                                    order_id=db_order.order_id,
                                    old_status=db_order.status,
                                    trace_id=trace_id,
                                )
                        elif synced_order.status != db_order.status:
                            # Order status changed (e.g., pending -> filled)
                            discrepancies.append(
                                {
                                    "order_id": db_order.order_id,
                                    "issue": "status_changed_during_downtime",
                                    "old_status": db_order.status,
                                    "new_status": synced_order.status,
                                    "action": "updated_from_bybit",
                                }
                            )
                            synced_count += 1
                            logger.info(
                                "order_status_updated_during_sync",
                                order_id=db_order.order_id,
                                old_status=db_order.status,
                                new_status=synced_order.status,
                                trace_id=trace_id,
                            )
                    except Exception as e:
                        # Log error but continue with other orders
                        errors.append(f"Failed to sync order {db_order.order_id}: {e}")
                        logger.error(
                            "order_sync_by_id_failed_in_batch",
                            order_id=db_order.order_id,
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )

            logger.info(
                "order_sync_completed",
                synced_count=synced_count,
                discrepancies_count=len(discrepancies),
                errors_count=len(errors),
                trace_id=trace_id,
            )

            return {
                "synced_count": synced_count,
                "discrepancies": discrepancies,
                "errors": errors,
                "trace_id": trace_id,
            }

        except Exception as e:
            logger.error(
                "order_sync_failed",
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Order sync failed: {e}") from e

    async def _query_active_orders_from_bybit(
        self, trace_id: Optional[str] = None
    ) -> List[dict]:
        """
        Query all active orders from Bybit REST API.

        Args:
            trace_id: Optional trace ID

        Returns:
            List of active order dictionaries from Bybit API
        """
        try:
            # Query active orders (status: New, PartiallyFilled)
            # Bybit API endpoint: GET /v5/order/realtime
            params = {
                "category": "linear",  # Linear perpetual contracts
                "settleCoin": "USDT",
            }

            response = await self.bybit_client.get(
                "/v5/order/realtime",
                params=params,
            )

            if response.get("retCode") != 0:
                error_msg = response.get("retMsg", "Unknown error")
                raise BybitAPIError(f"Bybit API error: {error_msg}")

            orders = response.get("result", {}).get("list", [])

            # Filter for active orders only
            active_statuses = ["New", "PartiallyFilled"]
            active_orders = [
                order
                for order in orders
                if order.get("orderStatus") in active_statuses
            ]

            logger.debug(
                "bybit_active_orders_queried",
                total_orders=len(orders),
                active_orders=len(active_orders),
                trace_id=trace_id,
            )

            return active_orders

        except Exception as e:
            logger.error(
                "bybit_active_orders_query_failed",
                error=str(e),
                trace_id=trace_id,
            )
            raise BybitAPIError(f"Failed to query active orders from Bybit: {e}") from e

    async def _query_active_orders_from_db(
        self, trace_id: Optional[str] = None
    ) -> List[Order]:
        """
        Query all active orders from database.

        Args:
            trace_id: Optional trace ID

        Returns:
            List of Order objects with active status
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                       status, filled_quantity, average_price, fees, created_at, updated_at,
                       executed_at, trace_id, is_dry_run
                FROM orders
                WHERE status IN ('pending', 'partially_filled')
            """
            rows = await pool.fetch(query)

            orders = [Order.from_dict(dict(row)) for row in rows]

            logger.debug(
                "db_active_orders_queried",
                active_orders_count=len(orders),
                trace_id=trace_id,
            )

            return orders

        except Exception as e:
            logger.error(
                "db_active_orders_query_failed",
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to query active orders from database: {e}") from e

    async def _compare_and_sync_order(
        self, bybit_order: dict, db_order: Order, trace_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Compare Bybit order with database order and sync if different.

        Args:
            bybit_order: Order data from Bybit API
            db_order: Order from database
            trace_id: Optional trace ID

        Returns:
            Discrepancy dictionary if sync occurred, None otherwise
        """
        bybit_status = bybit_order.get("orderStatus", "")
        bybit_filled_qty = Decimal(str(bybit_order.get("cumExecQty", "0")))
        bybit_avg_price = (
            Decimal(str(bybit_order.get("avgPrice", "0")))
            if bybit_order.get("avgPrice")
            else None
        )

        # Map Bybit status to our status
        status_map = {
            "New": "pending",
            "PartiallyFilled": "partially_filled",
            "Filled": "filled",
            "Cancelled": "cancelled",
            "Rejected": "rejected",
        }
        mapped_status = status_map.get(bybit_status, bybit_status.lower())

        # Check for discrepancies
        needs_update = False
        discrepancy_reasons = []

        if db_order.status != mapped_status:
            needs_update = True
            discrepancy_reasons.append(
                f"status: {db_order.status} -> {mapped_status}"
            )

        if db_order.filled_quantity != bybit_filled_qty:
            needs_update = True
            discrepancy_reasons.append(
                f"filled_quantity: {db_order.filled_quantity} -> {bybit_filled_qty}"
            )

        if bybit_avg_price and (
            db_order.average_price is None
            or abs(db_order.average_price - bybit_avg_price) > Decimal("0.01")
        ):
            needs_update = True
            discrepancy_reasons.append(
                f"average_price: {db_order.average_price} -> {bybit_avg_price}"
            )

        if not needs_update:
            return None

        # Update database order
        await self._update_order_from_bybit(
            db_order.id,
            mapped_status,
            bybit_filled_qty,
            bybit_avg_price,
            trace_id=trace_id,
        )

        logger.info(
            "order_synced",
            order_id=db_order.order_id,
            old_status=db_order.status,
            new_status=mapped_status,
            reasons=discrepancy_reasons,
            trace_id=trace_id,
        )

        return {
            "order_id": db_order.order_id,
            "issue": "state_mismatch",
            "reasons": discrepancy_reasons,
            "old_status": db_order.status,
            "new_status": mapped_status,
        }

    async def _update_order_from_bybit(
        self,
        order_id: UUID,
        status: str,
        filled_quantity: Decimal,
        average_price: Optional[Decimal],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Update order in database with data from Bybit.

        Args:
            order_id: Internal order ID (UUID)
            status: New order status
            filled_quantity: Filled quantity from Bybit
            average_price: Average execution price from Bybit
            trace_id: Optional trace ID
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Determine executed_at timestamp if order is filled
            executed_at = None
            if status == "filled":
                executed_at = datetime.utcnow()

            update_query = """
                UPDATE orders
                SET status = $1,
                    filled_quantity = $2,
                    average_price = $3,
                    updated_at = NOW(),
                    executed_at = $4
                WHERE id = $5
            """
            await pool.execute(
                update_query,
                status,
                str(filled_quantity),
                str(average_price) if average_price else None,
                executed_at,
                str(order_id),
            )

            logger.debug(
                "order_updated_from_bybit",
                order_id=str(order_id),
                status=status,
                filled_quantity=float(filled_quantity),
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "order_update_from_bybit_failed",
                order_id=str(order_id),
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to update order from Bybit: {e}") from e

    async def _update_order_status(
        self, order_id: UUID, status: str, trace_id: Optional[str] = None
    ) -> None:
        """
        Update order status in database.

        Args:
            order_id: Internal order ID (UUID)
            status: New status
            trace_id: Optional trace ID
        """
        try:
            pool = await DatabaseConnection.get_pool()

            update_query = """
                UPDATE orders
                SET status = $1, updated_at = NOW()
                WHERE id = $2
            """
            await pool.execute(update_query, status, str(order_id))

            logger.debug(
                "order_status_updated",
                order_id=str(order_id),
                status=status,
                trace_id=trace_id,
            )

        except Exception as e:
            logger.error(
                "order_status_update_failed",
                order_id=str(order_id),
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to update order status: {e}") from e

    async def sync_order_by_id(
        self, bybit_order_id: str, trace_id: Optional[str] = None
    ) -> Optional[Order]:
        """
        Synchronize a specific order by Bybit order ID.

        Tries to find order in realtime endpoint first, then in history endpoint
        if not found. This ensures we can sync orders that were filled/cancelled
        during downtime.

        Args:
            bybit_order_id: Bybit order ID
            trace_id: Optional trace ID

        Returns:
            Updated Order object if found, None otherwise
        """
        trace_id = trace_id or get_or_create_trace_id()

        try:
            # First, try to find order in realtime endpoint (active orders)
            params = {
                "category": "linear",
                "orderId": bybit_order_id,
            }

            response = await self.bybit_client.get(
                "/v5/order/realtime",
                params=params,
            )

            bybit_order = None
            if response.get("retCode") == 0:
                orders = response.get("result", {}).get("list", [])
                if orders:
                    bybit_order = orders[0]

            # If not found in realtime, try history endpoint (filled/cancelled orders)
            if not bybit_order:
                logger.debug(
                    "order_not_found_in_realtime_trying_history",
                    order_id=bybit_order_id,
                    trace_id=trace_id,
                )
                response = await self.bybit_client.get(
                    "/v5/order/history",
                    params=params,
                )

                if response.get("retCode") == 0:
                    orders = response.get("result", {}).get("list", [])
                    if orders:
                        bybit_order = orders[0]

            # If still not found, order doesn't exist in Bybit
            if not bybit_order:
                logger.warning(
                    "order_not_found_in_bybit",
                    order_id=bybit_order_id,
                    trace_id=trace_id,
                )
                return None

            # Find order in database
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, order_id, signal_id, asset, side, order_type, quantity, price,
                       status, filled_quantity, average_price, fees, created_at, updated_at,
                       executed_at, trace_id, is_dry_run
                FROM orders
                WHERE order_id = $1
            """
            row = await pool.fetchrow(query, bybit_order_id)

            if row is None:
                logger.warning(
                    "order_not_found_in_db",
                    order_id=bybit_order_id,
                    trace_id=trace_id,
                )
                return None

            db_order = Order.from_dict(dict(row))

            # Sync order
            discrepancy = await self._compare_and_sync_order(
                bybit_order, db_order, trace_id
            )

            if discrepancy:
                # Re-fetch updated order
                row = await pool.fetchrow(query, bybit_order_id)
                return Order.from_dict(dict(row)) if row else None

            return db_order

        except Exception as e:
            logger.error(
                "order_sync_by_id_failed",
                order_id=bybit_order_id,
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to sync order by ID: {e}") from e

