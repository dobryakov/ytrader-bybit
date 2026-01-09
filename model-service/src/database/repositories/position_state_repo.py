"""
Order and Position State repository.

Reads current open orders and positions.
Orders are read from shared PostgreSQL database.
Positions are read from Position Manager API (not directly from database).
"""

from typing import Optional, List
from datetime import datetime
from decimal import Decimal
import asyncpg

from ..base import BaseRepository
from ...models.position_state import OrderPositionState, OrderState, PositionState
from ...config.logging import get_logger
from ...services.position_manager_client import position_manager_client

logger = get_logger(__name__)


class PositionStateRepository(BaseRepository[OrderPositionState]):
    """
    Repository for reading order and position state.
    
    Orders are read from shared PostgreSQL database.
    Positions are read from Position Manager API (not directly from database).
    """

    @property
    def table_name(self) -> str:
        """Return the table name (not used for read-only operations)."""
        return "orders"  # Not used, but required by base class

    async def get_order_position_state(
        self,
        strategy_id: Optional[str] = None,
        asset: Optional[str] = None,
    ) -> OrderPositionState:
        """
        Get current order and position state for a strategy.

        Args:
            strategy_id: Trading strategy identifier (None for all strategies)
            asset: Asset filter (None for all assets)

        Returns:
            OrderPositionState with current orders and positions
        """
        # Fetch open orders
        orders = await self._get_open_orders(strategy_id=strategy_id, asset=asset)

        # Fetch open positions
        positions = await self._get_open_positions(asset=asset)

        return OrderPositionState(
            strategy_id=strategy_id or "all",
            orders=orders,
            positions=positions,
            snapshot_timestamp=datetime.utcnow(),
        )

    async def _get_open_orders(
        self,
        strategy_id: Optional[str] = None,
        asset: Optional[str] = None,
    ) -> List[OrderState]:
        """
        Get open orders from database.

        Args:
            strategy_id: Strategy filter (None for all)
            asset: Asset filter (None for all)

        Returns:
            List of OrderState objects
        """
        query = """
            SELECT 
                id, order_id, signal_id, asset, side, order_type,
                quantity, price, status, filled_quantity, average_price,
                fees, created_at, updated_at, executed_at
            FROM orders
            WHERE status IN ('pending', 'partially_filled')
        """
        params = []
        param_index = 1

        # Note: orders table doesn't have strategy_id column directly
        # We filter by signal_id if needed, but for now we'll get all open orders
        # and filter by asset if specified
        if asset:
            query += f" AND asset = ${param_index}"
            params.append(asset)
            param_index += 1

        query += " ORDER BY created_at DESC"

        try:
            records = await self._fetch(query, *params)
            orders = []

            for record in records:
                try:
                    order = OrderState(
                        id=str(record["id"]),
                        order_id=record["order_id"],
                        signal_id=str(record["signal_id"]),
                        asset=record["asset"],
                        side=record["side"],
                        order_type=record["order_type"],
                        quantity=Decimal(str(record["quantity"])),
                        price=Decimal(str(record["price"])) if record["price"] else None,
                        status=record["status"],
                        filled_quantity=Decimal(str(record["filled_quantity"])),
                        average_price=Decimal(str(record["average_price"])) if record["average_price"] else None,
                        fees=Decimal(str(record["fees"])) if record["fees"] else None,
                        created_at=record["created_at"],
                        updated_at=record["updated_at"],
                        executed_at=record["executed_at"],
                    )
                    orders.append(order)
                except Exception as e:
                    logger.warning(
                        "Failed to parse order record",
                        order_id=record.get("order_id"),
                        error=str(e),
                        exc_info=True,
                    )
                    continue

            logger.debug("Fetched open orders", count=len(orders), strategy_id=strategy_id, asset=asset)
            return orders

        except Exception as e:
            logger.error("Failed to fetch open orders", strategy_id=strategy_id, asset=asset, error=str(e), exc_info=True)
            raise

    async def _get_open_positions(self, asset: Optional[str] = None) -> List[PositionState]:
        """
        Get open positions from Position Manager API (not directly from database).

        Args:
            asset: Asset filter (None for all)

        Returns:
            List of PositionState objects
        """
        try:
            # Fetch positions from Position Manager API
            positions_data = await position_manager_client.get_all_positions(asset=asset)
            positions = []

            for pos_data in positions_data:
                try:
                    # Map API response to PositionState model
                    # API returns positions with fields like: id, asset, size, average_entry_price, etc.
                    # Parse timestamps (API returns ISO format strings with 'Z' suffix)
                    last_updated_str = pos_data.get("last_updated")
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                    else:
                        last_updated = datetime.utcnow()
                    
                    last_snapshot_at_str = pos_data.get("last_snapshot_at")
                    last_snapshot_at = None
                    if last_snapshot_at_str:
                        try:
                            last_snapshot_at = datetime.fromisoformat(last_snapshot_at_str.replace("Z", "+00:00"))
                        except (ValueError, AttributeError):
                            pass
                    
                    position = PositionState(
                        id=str(pos_data.get("id", "")),
                        asset=pos_data.get("asset", ""),
                        size=Decimal(str(pos_data.get("size", 0))),
                        average_entry_price=Decimal(str(pos_data.get("average_entry_price")))
                        if pos_data.get("average_entry_price") is not None
                        else None,
                        unrealized_pnl=Decimal(str(pos_data.get("unrealized_pnl", 0))),
                        realized_pnl=Decimal(str(pos_data.get("realized_pnl", 0))),
                        mode=pos_data.get("mode", "one-way"),
                        long_size=Decimal(str(pos_data.get("long_size")))
                        if pos_data.get("long_size") is not None
                        else None,
                        short_size=Decimal(str(pos_data.get("short_size")))
                        if pos_data.get("short_size") is not None
                        else None,
                        long_avg_price=Decimal(str(pos_data.get("long_avg_price")))
                        if pos_data.get("long_avg_price") is not None
                        else None,
                        short_avg_price=Decimal(str(pos_data.get("short_avg_price")))
                        if pos_data.get("short_avg_price") is not None
                        else None,
                        last_updated=last_updated,
                        last_snapshot_at=last_snapshot_at,
                    )
                    positions.append(position)
                except Exception as e:
                    logger.warning(
                        "Failed to parse position from API",
                        position_id=pos_data.get("id"),
                        asset=pos_data.get("asset"),
                        error=str(e),
                        exc_info=True,
                    )
                    continue

            logger.debug("Fetched open positions from Position Manager API", count=len(positions), asset=asset)
            return positions

        except Exception as e:
            logger.error(
                "Failed to fetch open positions from Position Manager API",
                asset=asset,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Return empty list on error to avoid breaking the calling code
            # Critical errors are already logged with circuit breaker in position_manager_client
            return []

    async def get_position_for_asset(self, asset: str) -> Optional[PositionState]:
        """
        Get position for a specific asset.

        Args:
            asset: Trading pair symbol

        Returns:
            PositionState or None if not found
        """
        positions = await self._get_open_positions(asset=asset)
        return positions[0] if positions else None

    async def get_orders_for_asset(self, asset: str) -> List[OrderState]:
        """
        Get open orders for a specific asset.

        Args:
            asset: Trading pair symbol

        Returns:
            List of OrderState objects
        """
        return await self._get_open_orders(asset=asset)

