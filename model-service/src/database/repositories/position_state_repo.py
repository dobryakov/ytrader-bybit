"""
Order and Position State repository.

Reads current open orders and positions from shared PostgreSQL database.
"""

from typing import Optional, List
from datetime import datetime
from decimal import Decimal
import asyncpg

from ..base import BaseRepository
from ...models.position_state import OrderPositionState, OrderState, PositionState
from ...config.logging import get_logger

logger = get_logger(__name__)


class PositionStateRepository(BaseRepository[OrderPositionState]):
    """Repository for reading order and position state from shared database."""

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
        Get open positions from database.

        Args:
            asset: Asset filter (None for all)

        Returns:
            List of PositionState objects
        """
        query = """
            SELECT 
                id, asset, size, average_entry_price, unrealized_pnl,
                realized_pnl, mode, long_size, short_size,
                long_avg_price, short_avg_price, last_updated, last_snapshot_at
            FROM positions
            WHERE size != 0
        """
        params = []
        param_index = 1

        if asset:
            query += f" AND asset = ${param_index}"
            params.append(asset)
            param_index += 1

        query += " ORDER BY last_updated DESC"

        try:
            records = await self._fetch(query, *params)
            positions = []

            for record in records:
                try:
                    position = PositionState(
                        id=str(record["id"]),
                        asset=record["asset"],
                        size=Decimal(str(record["size"])),
                        average_entry_price=Decimal(str(record["average_entry_price"]))
                        if record["average_entry_price"]
                        else None,
                        unrealized_pnl=Decimal(str(record["unrealized_pnl"])),
                        realized_pnl=Decimal(str(record["realized_pnl"])),
                        mode=record["mode"],
                        long_size=Decimal(str(record["long_size"])) if record["long_size"] else None,
                        short_size=Decimal(str(record["short_size"])) if record["short_size"] else None,
                        long_avg_price=Decimal(str(record["long_avg_price"])) if record["long_avg_price"] else None,
                        short_avg_price=Decimal(str(record["short_avg_price"])) if record["short_avg_price"] else None,
                        last_updated=record["last_updated"],
                        last_snapshot_at=record["last_snapshot_at"],
                    )
                    positions.append(position)
                except Exception as e:
                    logger.warning(
                        "Failed to parse position record",
                        position_id=record.get("id"),
                        error=str(e),
                        exc_info=True,
                    )
                    continue

            logger.debug("Fetched open positions", count=len(positions), asset=asset)
            return positions

        except Exception as e:
            logger.error("Failed to fetch open positions", asset=asset, error=str(e), exc_info=True)
            raise

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

