"""
Position Order database repository.

Provides CRUD operations for position_orders table with support for NULL order_id.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from decimal import Decimal
import asyncpg

from ...config.database import DatabaseConnection
from ...config.logging import get_logger
from ...exceptions import DatabaseError

logger = get_logger(__name__)


class PositionOrderRepository:
    """Repository for position_orders table operations."""

    async def upsert(
        self,
        position_id: UUID,
        bybit_order_id: str,
        order_id: Optional[UUID],
        relationship_type: str,
        size_delta: Decimal,
        execution_price: Decimal,
        executed_at: datetime,
    ) -> Dict[str, Any]:
        """
        Create or update a position-order relationship.

        Uses INSERT ... ON CONFLICT to handle partial fills and updates.

        Args:
            position_id: Position UUID
            bybit_order_id: Bybit order ID (always required for linking)
            order_id: Order UUID (can be NULL if order not yet created in DB)
            relationship_type: Type of relationship ('opened', 'increased', 'decreased', 'closed', 'reversed')
            size_delta: Change in position size
            execution_price: Execution price
            executed_at: Execution timestamp

        Returns:
            Created or updated position-order relationship record

        Raises:
            DatabaseError: If upsert fails
        """
        try:
            pool = await DatabaseConnection.get_pool()
            
            # Use different conflict resolution based on whether order_id is NULL
            if order_id is not None:
                # order_id is available: use unique constraint on (position_id, order_id)
                query = """
                    INSERT INTO position_orders (
                        position_id, order_id, bybit_order_id, relationship_type,
                        size_delta, execution_price, executed_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (position_id, order_id) 
                    WHERE order_id IS NOT NULL
                    DO UPDATE SET
                        relationship_type = EXCLUDED.relationship_type,
                        size_delta = EXCLUDED.size_delta,
                        execution_price = EXCLUDED.execution_price,
                        executed_at = EXCLUDED.executed_at,
                        bybit_order_id = EXCLUDED.bybit_order_id
                    RETURNING *
                """
                record = await pool.fetchrow(
                    query,
                    position_id,
                    order_id,
                    bybit_order_id,
                    relationship_type,
                    str(size_delta),
                    str(execution_price),
                    executed_at,
                )
            else:
                # order_id is NULL: use unique constraint on (position_id, bybit_order_id)
                query = """
                    INSERT INTO position_orders (
                        position_id, order_id, bybit_order_id, relationship_type,
                        size_delta, execution_price, executed_at
                    )
                    VALUES ($1, NULL, $2, $3, $4, $5, $6)
                    ON CONFLICT (position_id, bybit_order_id)
                    WHERE order_id IS NULL AND bybit_order_id IS NOT NULL
                    DO UPDATE SET
                        relationship_type = EXCLUDED.relationship_type,
                        size_delta = EXCLUDED.size_delta,
                        execution_price = EXCLUDED.execution_price,
                        executed_at = EXCLUDED.executed_at
                    RETURNING *
                """
                record = await pool.fetchrow(
                    query,
                    position_id,
                    bybit_order_id,
                    relationship_type,
                    str(size_delta),
                    str(execution_price),
                    executed_at,
                )
            
            if not record:
                raise ValueError("Failed to upsert position-order relationship")
            
            result = dict(record)
            logger.debug(
                "position_order_upserted",
                position_id=str(position_id),
                bybit_order_id=bybit_order_id,
                order_id=str(order_id) if order_id else None,
                relationship_type=relationship_type,
            )
            return result
            
        except Exception as e:
            logger.error(
                "position_order_upsert_failed",
                position_id=str(position_id),
                bybit_order_id=bybit_order_id,
                order_id=str(order_id) if order_id else None,
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to upsert position-order relationship: {e}") from e

    async def update_order_id(
        self,
        bybit_order_id: str,
        order_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """
        Update order_id for position_orders records with NULL order_id.

        Called by order-manager after creating order in DB to link position_orders.

        Args:
            bybit_order_id: Bybit order ID
            order_id: Order UUID (internal)

        Returns:
            Updated record or None if not found

        Raises:
            DatabaseError: If update fails
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                UPDATE position_orders
                SET order_id = $1
                WHERE bybit_order_id = $2
                  AND order_id IS NULL
                RETURNING *
            """
            record = await pool.fetchrow(query, order_id, bybit_order_id)
            if record:
                result = dict(record)
                logger.info(
                    "position_order_order_id_updated",
                    bybit_order_id=bybit_order_id,
                    order_id=str(order_id),
                )
                return result
            return None
        except Exception as e:
            logger.error(
                "position_order_update_order_id_failed",
                bybit_order_id=bybit_order_id,
                order_id=str(order_id),
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to update order_id in position_orders: {e}") from e

    async def get_by_position_and_bybit_order(
        self,
        position_id: UUID,
        bybit_order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get position-order relationship by position_id and bybit_order_id.

        Args:
            position_id: Position UUID
            bybit_order_id: Bybit order ID

        Returns:
            Position-order relationship record or None if not found
        """
        query = "SELECT * FROM position_orders WHERE position_id = $1 AND bybit_order_id = $2"
        try:
            pool = await DatabaseConnection.get_pool()
            record = await pool.fetchrow(query, position_id, bybit_order_id)
            return dict(record) if record else None
        except Exception as e:
            logger.error(
                "get_position_order_by_bybit_id_failed",
                position_id=str(position_id),
                bybit_order_id=bybit_order_id,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationship: {e}") from e

    async def get_by_id(self, position_order_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get position-order relationship by ID.

        Args:
            position_order_id: Position-order relationship UUID

        Returns:
            Position-order relationship record or None if not found
        """
        query = "SELECT * FROM position_orders WHERE id = $1"
        try:
            pool = await DatabaseConnection.get_pool()
            record = await pool.fetchrow(query, position_order_id)
            return dict(record) if record else None
        except Exception as e:
            logger.error(
                "get_position_order_by_id_failed",
                position_order_id=str(position_order_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationship: {e}") from e

    async def get_by_position_id(
        self,
        position_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Get all position-order relationships for a position.

        Args:
            position_id: Position UUID

        Returns:
            List of position-order relationship records
        """
        query = """
            SELECT * FROM position_orders
            WHERE position_id = $1
            ORDER BY executed_at DESC
        """
        try:
            pool = await DatabaseConnection.get_pool()
            records = await pool.fetch(query, position_id)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(
                "get_position_orders_by_position_id_failed",
                position_id=str(position_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationships: {e}") from e

    async def get_by_order_id(
        self,
        order_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        Get all position-order relationships for an order.

        Args:
            order_id: Order UUID

        Returns:
            List of position-order relationship records
        """
        query = """
            SELECT * FROM position_orders
            WHERE order_id = $1
            ORDER BY executed_at DESC
        """
        try:
            pool = await DatabaseConnection.get_pool()
            records = await pool.fetch(query, order_id)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(
                "get_position_orders_by_order_id_failed",
                order_id=str(order_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationships: {e}") from e

