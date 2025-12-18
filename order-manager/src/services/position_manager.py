"""Position manager service for position state management."""

from decimal import Decimal
from typing import Optional
from uuid import UUID

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..models.position import Position, PositionSnapshot
from ..exceptions import DatabaseError
from ..database.repositories.position_order_repo import PositionOrderRepository
from datetime import datetime

logger = get_logger(__name__)


class PositionManager:
    """Service for managing position state and queries."""

    async def get_position(self, asset: str, mode: str = "one-way") -> Optional[Position]:
        """Get current position for an asset.

        Args:
            asset: Trading pair symbol
            mode: Trading mode ('one-way' or 'hedge')

        Returns:
            Position object if found, None otherwise
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                       mode, long_size, short_size, long_avg_price, short_avg_price,
                       last_updated, last_snapshot_at
                FROM positions
                WHERE asset = $1 AND mode = $2
            """
            row = await pool.fetchrow(query, asset, mode)

            if row is None:
                logger.debug("position_not_found", asset=asset, mode=mode)
                return None

            position_data = dict(row)
            position = Position.from_dict(position_data)

            logger.debug(
                "position_retrieved",
                asset=asset,
                mode=mode,
                size=float(position.size),
                trace_id=None,
            )

            return position

        except Exception as e:
            logger.error("position_query_failed", asset=asset, mode=mode, error=str(e))
            raise DatabaseError(f"Failed to query position: {e}") from e

    async def update_position(
        self,
        asset: str,
        size_delta: Decimal,
        execution_price: Decimal,
        mode: str = "one-way",
        trace_id: Optional[str] = None,
        order_id: Optional[UUID] = None,
        executed_at: Optional[datetime] = None,
    ) -> Position:
        """Update position based on order execution.

        Args:
            asset: Trading pair symbol
            size_delta: Change in position size (positive for buy, negative for sell)
            execution_price: Price at which order was executed
            mode: Trading mode ('one-way' or 'hedge')
            trace_id: Optional trace ID for request tracking

        Returns:
            Updated Position object
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Get current position
            current_position = await self.get_position(asset, mode)

            if current_position is None:
                # Create new position
                new_size = size_delta
                new_avg_price = execution_price
            else:
                # Update existing position
                current_size = current_position.size
                current_avg_price = current_position.average_entry_price or execution_price

                # Calculate new size
                new_size = current_size + size_delta

                # Calculate new average price (weighted average)
                if new_size != 0:
                    if (current_size > 0 and size_delta > 0) or (current_size < 0 and size_delta < 0):
                        # Same direction: weighted average
                        # Use absolute values for calculation to ensure positive average price
                        total_value = (abs(current_size) * current_avg_price) + (abs(size_delta) * execution_price)
                        new_avg_price = total_value / abs(new_size)
                        # Ensure average price is always positive
                        if new_avg_price < 0:
                            new_avg_price = abs(new_avg_price)
                    else:
                        # Opposite direction: reduce position
                        if abs(size_delta) >= abs(current_size):
                            # Position reversed or closed
                            new_avg_price = execution_price
                        else:
                            # Partial reduction: keep current average price
                            new_avg_price = current_avg_price
                else:
                    # Position closed
                    new_avg_price = None

            # Upsert position
            upsert_query = """
                INSERT INTO positions (asset, size, average_entry_price, mode, last_updated)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (asset, mode)
                DO UPDATE SET
                    size = EXCLUDED.size,
                    average_entry_price = EXCLUDED.average_entry_price,
                    last_updated = NOW()
                RETURNING id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                          mode, long_size, short_size, long_avg_price, short_avg_price,
                          last_updated, last_snapshot_at
            """
            row = await pool.fetchrow(
                upsert_query,
                asset,
                str(new_size),
                str(new_avg_price) if new_avg_price is not None else None,
                mode,
            )

            position_data = dict(row)
            position = Position.from_dict(position_data)

            # Create position-order relationship if order_id provided
            if order_id:
                try:
                    relationship_type = self._determine_relationship_type(
                        current_position=current_position,
                        new_size=new_size,
                        size_delta=size_delta,
                    )
                    position_order_repo = PositionOrderRepository()
                    await position_order_repo.create(
                        position_id=position.id,
                        order_id=order_id,
                        relationship_type=relationship_type,
                        size_delta=size_delta,
                        execution_price=execution_price,
                        executed_at=executed_at or datetime.utcnow(),
                    )
                    logger.debug(
                        "Position-order relationship created",
                        position_id=str(position.id),
                        order_id=str(order_id),
                        relationship_type=relationship_type,
                        trace_id=trace_id,
                    )
                except Exception as e:
                    # Log error but don't fail position update
                    logger.warning(
                        "Failed to create position-order relationship (continuing)",
                        position_id=str(position.id),
                        order_id=str(order_id),
                        error=str(e),
                        trace_id=trace_id,
                    )

            logger.info(
                "position_updated",
                asset=asset,
                mode=mode,
                size_delta=float(size_delta),
                execution_price=float(execution_price),
                new_size=float(position.size),
                new_avg_price=float(position.average_entry_price) if position.average_entry_price else None,
                trace_id=trace_id,
            )

            return position

        except Exception as e:
            logger.error(
                "position_update_failed",
                asset=asset,
                mode=mode,
                size_delta=float(size_delta),
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to update position: {e}") from e

    def _determine_relationship_type(
        self,
        current_position: Optional[Position],
        new_size: Decimal,
        size_delta: Decimal,
    ) -> str:
        """
        Determine relationship type between position and order.

        Args:
            current_position: Current position before update (None if new)
            new_size: New position size after update
            size_delta: Change in position size

        Returns:
            Relationship type: 'opened', 'increased', 'decreased', 'closed', 'reversed'
        """
        if current_position is None:
            # New position
            return "opened"
        
        current_size = current_position.size
        
        if new_size == 0:
            # Position closed
            return "closed"
        elif (current_size > 0 and new_size < 0) or (current_size < 0 and new_size > 0):
            # Position reversed
            return "reversed"
        elif abs(new_size) > abs(current_size):
            # Position increased
            return "increased"
        elif abs(new_size) < abs(current_size):
            # Position decreased
            return "decreased"
        else:
            # Size unchanged (shouldn't happen, but handle gracefully)
            return "increased"  # Default fallback

    async def validate_position(
        self, asset: str, mode: str = "one-way", fix_discrepancies: bool = True
    ) -> tuple[bool, Optional[str], Optional[Position]]:
        """Validate position by computing from order history and comparing with stored state.

        Args:
            asset: Trading pair symbol
            mode: Trading mode ('one-way' or 'hedge')
            fix_discrepancies: If True, update stored position when discrepancy is found

        Returns:
            Tuple of (is_valid, error_message, updated_position)
            - is_valid: True if position is valid, False if discrepancy found
            - error_message: Description of discrepancy if any
            - updated_position: Updated position if discrepancy was fixed, None otherwise
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Compute position from order history (weighted average price by filled quantity)
            compute_query = """
                SELECT
                    SUM(CASE WHEN side = 'Buy' THEN filled_quantity ELSE -filled_quantity END) as computed_size,
                    CASE 
                        WHEN SUM(filled_quantity) > 0 THEN
                            SUM(average_price * filled_quantity) FILTER (WHERE average_price IS NOT NULL) / SUM(filled_quantity) FILTER (WHERE average_price IS NOT NULL)
                        ELSE NULL
                    END as computed_avg_price
                FROM orders
                WHERE asset = $1 AND status IN ('filled', 'partially_filled')
            """
            row = await pool.fetchrow(compute_query, asset)

            computed_size = row["computed_size"] or Decimal("0")
            computed_avg_price = Decimal(str(row["computed_avg_price"])) if row["computed_avg_price"] is not None else None

            # Get stored position
            stored_position = await self.get_position(asset, mode)

            if stored_position is None:
                if computed_size == 0:
                    return (True, None, None)
                error_msg = f"Stored position missing but computed size is {computed_size}"
                logger.warning("position_validation_failed", asset=asset, error=error_msg)
                
                # If discrepancy fixing is enabled and computed position exists, create the position
                if fix_discrepancies and computed_size != 0:
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        action="creating_missing_position",
                        computed_size=float(computed_size),
                    )
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, computed_avg_price, mode
                    )
                    return (True, f"Created missing position: {error_msg}", updated_position)
                
                return (False, error_msg, None)

            # Compare sizes (allow small differences due to rounding)
            size_diff = abs(stored_position.size - computed_size)
            if size_diff > Decimal("0.0001"):
                error_msg = (
                    f"Position size mismatch: stored={stored_position.size}, "
                    f"computed={computed_size}, diff={size_diff}"
                )
                logger.warning("position_validation_failed", asset=asset, error=error_msg)
                
                # Handle discrepancy by updating stored position with computed values
                if fix_discrepancies:
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        action="updating_position",
                        stored_size=float(stored_position.size),
                        computed_size=float(computed_size),
                        size_diff=float(size_diff),
                    )
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, computed_avg_price, mode
                    )
                    return (True, f"Fixed discrepancy: {error_msg}", updated_position)
                
                return (False, error_msg, None)

            logger.debug("position_validation_passed", asset=asset, size=float(stored_position.size))
            return (True, None, None)

        except Exception as e:
            logger.error("position_validation_error", asset=asset, error=str(e))
            return (False, f"Validation error: {e}", None)

    async def _update_position_from_computed(
        self,
        asset: str,
        computed_size: Decimal,
        computed_avg_price: Optional[Decimal],
        mode: str,
    ) -> Position:
        """Update stored position from computed values.

        Args:
            asset: Trading pair symbol
            computed_size: Computed position size from order history
            computed_avg_price: Computed average price from order history
            mode: Trading mode

        Returns:
            Updated Position object
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Upsert position with computed values
            upsert_query = """
                INSERT INTO positions (asset, size, average_entry_price, mode, last_updated)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (asset, mode)
                DO UPDATE SET
                    size = EXCLUDED.size,
                    average_entry_price = EXCLUDED.average_entry_price,
                    last_updated = NOW()
                RETURNING id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                          mode, long_size, short_size, long_avg_price, short_avg_price,
                          last_updated, last_snapshot_at
            """
            row = await pool.fetchrow(
                upsert_query,
                asset,
                str(computed_size),
                str(computed_avg_price) if computed_avg_price is not None else None,
                mode,
            )

            position_data = dict(row)
            updated_position = Position.from_dict(position_data)

            logger.info(
                "position_updated_from_computed",
                asset=asset,
                mode=mode,
                computed_size=float(computed_size),
                computed_avg_price=float(computed_avg_price) if computed_avg_price else None,
            )

            return updated_position

        except Exception as e:
            logger.error(
                "position_update_from_computed_failed",
                asset=asset,
                mode=mode,
                error=str(e),
            )
            raise DatabaseError(f"Failed to update position from computed values: {e}") from e

    async def create_position_snapshot(
        self,
        position: Position,
        trace_id: Optional[str] = None,
    ) -> PositionSnapshot:
        """Create a snapshot of the current position state.

        Args:
            position: Position to snapshot
            trace_id: Optional trace ID for request tracking

        Returns:
            PositionSnapshot object
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Create snapshot from position
            snapshot = PositionSnapshot(
                position_id=position.id,
                asset=position.asset,
                size=position.size,
                average_entry_price=position.average_entry_price,
                unrealized_pnl=position.unrealized_pnl,
                realized_pnl=position.realized_pnl,
                mode=position.mode,
                long_size=position.long_size,
                short_size=position.short_size,
            )

            # Insert snapshot into database
            insert_query = """
                INSERT INTO position_snapshots (
                    id, position_id, asset, size, average_entry_price,
                    unrealized_pnl, realized_pnl, mode, long_size, short_size,
                    snapshot_timestamp
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id, position_id, asset, size, average_entry_price,
                          unrealized_pnl, realized_pnl, mode, long_size, short_size,
                          snapshot_timestamp
            """
            row = await pool.fetchrow(
                insert_query,
                str(snapshot.id),
                str(snapshot.position_id),
                snapshot.asset,
                str(snapshot.size),
                str(snapshot.average_entry_price) if snapshot.average_entry_price is not None else None,
                str(snapshot.unrealized_pnl) if snapshot.unrealized_pnl is not None else None,
                str(snapshot.realized_pnl) if snapshot.realized_pnl is not None else None,
                snapshot.mode,
                str(snapshot.long_size) if snapshot.long_size is not None else None,
                str(snapshot.short_size) if snapshot.short_size is not None else None,
                snapshot.snapshot_timestamp,
            )

            snapshot_data = dict(row)
            created_snapshot = PositionSnapshot.from_dict(snapshot_data)

            # Update position's last_snapshot_at timestamp
            update_query = """
                UPDATE positions
                SET last_snapshot_at = $1
                WHERE id = $2
            """
            await pool.execute(update_query, snapshot.snapshot_timestamp, str(position.id))

            logger.info(
                "position_snapshot_created",
                position_id=str(position.id),
                asset=position.asset,
                size=float(position.size),
                snapshot_id=str(created_snapshot.id),
                trace_id=trace_id,
            )

            return created_snapshot

        except Exception as e:
            logger.error(
                "position_snapshot_creation_failed",
                position_id=str(position.id),
                asset=position.asset,
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to create position snapshot: {e}") from e

    async def get_all_positions(self) -> list[Position]:
        """Get all positions from the database.

        Returns:
            List of Position objects
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, size, average_entry_price, unrealized_pnl, realized_pnl,
                       mode, long_size, short_size, long_avg_price, short_avg_price,
                       last_updated, last_snapshot_at
                FROM positions
                ORDER BY asset, mode
            """
            rows = await pool.fetch(query)

            positions = []
            for row in rows:
                position_data = dict(row)
                position = Position.from_dict(position_data)
                positions.append(position)

            logger.debug("all_positions_retrieved", count=len(positions))
            return positions

        except Exception as e:
            logger.error("get_all_positions_failed", error=str(e))
            raise DatabaseError(f"Failed to retrieve all positions: {e}") from e

