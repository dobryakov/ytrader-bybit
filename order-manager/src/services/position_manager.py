"""Position manager service for position state management."""

from decimal import Decimal
from typing import Optional
from uuid import UUID

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..models.position import Position
from ..exceptions import DatabaseError

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
                        total_value = (current_size * current_avg_price) + (size_delta * execution_price)
                        new_avg_price = total_value / abs(new_size)
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

    async def validate_position(self, asset: str, mode: str = "one-way") -> tuple[bool, Optional[str]]:
        """Validate position by computing from order history and comparing with stored state.

        Args:
            asset: Trading pair symbol
            mode: Trading mode ('one-way' or 'hedge')

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Compute position from order history
            compute_query = """
                SELECT
                    SUM(CASE WHEN side = 'Buy' THEN filled_quantity ELSE -filled_quantity END) as computed_size,
                    AVG(average_price) FILTER (WHERE average_price IS NOT NULL) as computed_avg_price
                FROM orders
                WHERE asset = $1 AND status IN ('filled', 'partially_filled')
            """
            row = await pool.fetchrow(compute_query, asset)

            computed_size = row["computed_size"] or Decimal("0")
            computed_avg_price = row["computed_avg_price"]

            # Get stored position
            stored_position = await self.get_position(asset, mode)

            if stored_position is None:
                if computed_size == 0:
                    return (True, None)
                return (False, f"Stored position missing but computed size is {computed_size}")

            # Compare sizes (allow small differences due to rounding)
            size_diff = abs(stored_position.size - computed_size)
            if size_diff > Decimal("0.0001"):
                error_msg = (
                    f"Position size mismatch: stored={stored_position.size}, "
                    f"computed={computed_size}, diff={size_diff}"
                )
                logger.warning("position_validation_failed", asset=asset, error=error_msg)
                return (False, error_msg)

            logger.debug("position_validation_passed", asset=asset, size=float(stored_position.size))
            return (True, None)

        except Exception as e:
            logger.error("position_validation_error", asset=asset, error=str(e))
            return (False, f"Validation error: {e}")

