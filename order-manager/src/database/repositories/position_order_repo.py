"""
Position Order database repository.

Provides CRUD operations for position_orders table.
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

    async def create(
        self,
        position_id: UUID,
        order_id: UUID,
        relationship_type: str,
        size_delta: Decimal,
        execution_price: Decimal,
        executed_at: datetime,
    ) -> Dict[str, Any]:
        """
        Create a new position-order relationship.

        Args:
            position_id: Position UUID
            order_id: Order UUID
            relationship_type: Type of relationship ('opened', 'increased', 'decreased', 'closed', 'reversed')
            size_delta: Change in position size
            execution_price: Execution price
            executed_at: Execution timestamp

        Returns:
            Created position-order relationship record

        Raises:
            DatabaseError: If creation fails
        """
        query = """
            INSERT INTO position_orders (
                position_id, order_id, relationship_type,
                size_delta, execution_price, executed_at
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *
        """
        try:
            pool = await DatabaseConnection.get_pool()
            record = await pool.fetchrow(
                query,
                position_id,
                order_id,
                relationship_type,
                str(size_delta),
                str(execution_price),
                executed_at,
            )
            if not record:
                raise ValueError("Failed to create position-order relationship")
            result = dict(record)
            logger.info(
                "Position-order relationship created",
                position_id=str(position_id),
                order_id=str(order_id),
                relationship_type=relationship_type,
            )
            return result
        except asyncpg.UniqueViolationError as e:
            logger.warning(
                "Position-order relationship already exists",
                position_id=str(position_id),
                order_id=str(order_id),
                error=str(e),
            )
            # Return existing record
            return await self.get_by_position_and_order(position_id, order_id)
        except Exception as e:
            logger.error(
                "Failed to create position-order relationship",
                position_id=str(position_id),
                order_id=str(order_id),
                error=str(e),
                exc_info=True,
            )
            raise DatabaseError(f"Failed to create position-order relationship: {e}") from e

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
                "Failed to get position-order relationship",
                position_order_id=str(position_order_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationship: {e}") from e

    async def get_by_position_and_order(
        self,
        position_id: UUID,
        order_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """
        Get position-order relationship by position_id and order_id.

        Args:
            position_id: Position UUID
            order_id: Order UUID

        Returns:
            Position-order relationship record or None if not found
        """
        query = "SELECT * FROM position_orders WHERE position_id = $1 AND order_id = $2"
        try:
            pool = await DatabaseConnection.get_pool()
            record = await pool.fetchrow(query, position_id, order_id)
            return dict(record) if record else None
        except Exception as e:
            logger.error(
                "Failed to get position-order relationship",
                position_id=str(position_id),
                order_id=str(order_id),
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
                "Failed to get position-order relationships",
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
                "Failed to get position-order relationships",
                order_id=str(order_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to get position-order relationships: {e}") from e

