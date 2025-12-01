"""Position manager service for position state management and ML features."""

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Tuple

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..exceptions import DatabaseError
from ..models import Position, PositionSnapshot

logger = get_logger(__name__)


class PositionManager:
    """Service for managing position state, queries, and ML feature helpers."""

    # === Core queries ======================================================

    async def get_position(self, asset: str, mode: str = "one-way") -> Optional[Position]:
        """Get current position for an asset/mode pair."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, version,
                       last_updated, closed_at, created_at
                FROM positions
                WHERE asset = $1 AND mode = $2
            """
            row = await pool.fetchrow(query, asset.upper(), mode.lower())
            if row is None:
                logger.debug("position_not_found", asset=asset, mode=mode)
                return None

            position = Position.from_db_dict(dict(row))
            logger.debug(
                "position_retrieved",
                asset=asset,
                mode=mode,
                size=str(position.size),
            )
            return position
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("position_query_failed", asset=asset, mode=mode, error=str(e))
            raise DatabaseError(f"Failed to query position: {e}") from e

    async def get_all_positions(self) -> List[Position]:
        """Get all positions from the database."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, version,
                       last_updated, closed_at, created_at
                FROM positions
                ORDER BY asset, mode
            """
            rows = await pool.fetch(query)
            positions = [Position.from_db_dict(dict(row)) for row in rows]
            logger.debug("all_positions_retrieved", count=len(positions))
            return positions
        except Exception as e:  # pragma: no cover
            logger.error("get_all_positions_failed", error=str(e))
            raise DatabaseError(f"Failed to retrieve all positions: {e}") from e

    # === Order-fill based update (Phase 3 scope) ===========================

    async def update_position_from_order_fill(
        self,
        asset: str,
        size_delta: Decimal,
        execution_price: Decimal,
        mode: str = "one-way",
    ) -> Position:
        """Update position based on order execution.

        This is a direct adaptation of Order Manager's position update logic,
        extended to work with the shared `positions` schema.
        """
        try:
            current_position = await self.get_position(asset, mode)

            if current_position is None:
                new_size = size_delta
                new_avg_price = execution_price
            else:
                current_size = current_position.size
                current_avg_price = current_position.average_entry_price or execution_price

                new_size = current_size + size_delta

                if new_size != 0:
                    if (current_size > 0 and size_delta > 0) or (
                        current_size < 0 and size_delta < 0
                    ):
                        total_value = (current_size * current_avg_price) + (
                            size_delta * execution_price
                        )
                        new_avg_price = total_value / abs(new_size)
                    else:
                        if abs(size_delta) >= abs(current_size):
                            new_avg_price = execution_price
                        else:
                            new_avg_price = current_avg_price
                else:
                    new_avg_price = None

            pool = await DatabaseConnection.get_pool()
            upsert_query = """
                INSERT INTO positions (
                    asset, mode, size, average_entry_price,
                    unrealized_pnl, realized_pnl,
                    current_price, version, last_updated, created_at
                )
                VALUES ($1, $2, $3, $4, 0, 0, NULL, 1, NOW(), NOW())
                ON CONFLICT (asset, mode)
                DO UPDATE SET
                    size = EXCLUDED.size,
                    average_entry_price = EXCLUDED.average_entry_price,
                    last_updated = NOW()
                RETURNING id, asset, mode, size, average_entry_price, current_price,
                          unrealized_pnl, realized_pnl,
                          long_size, short_size, version,
                          last_updated, closed_at, created_at
            """
            row = await pool.fetchrow(
                upsert_query,
                asset.upper(),
                mode.lower(),
                str(new_size),
                str(new_avg_price) if new_avg_price is not None else None,
            )
            position = Position.from_db_dict(dict(row))

            logger.info(
                "position_updated_from_order_fill",
                asset=asset,
                mode=mode,
                size_delta=str(size_delta),
                execution_price=str(execution_price),
                new_size=str(position.size),
                new_avg_price=str(position.average_entry_price)
                if position.average_entry_price is not None
                else None,
            )
            return position
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_update_from_order_fill_failed",
                asset=asset,
                mode=mode,
                size_delta=str(size_delta),
                error=str(e),
            )
            raise DatabaseError(f"Failed to update position from order fill: {e}") from e

    # === WebSocket-based update (Phase 3 placeholder) ======================

    async def update_position_from_websocket(
        self,
        asset: str,
        mode: str,
        mark_price: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
        realized_pnl: Optional[Decimal] = None,
    ) -> Optional[Position]:
        """Update position from WebSocket event.

        Phase 3 only needs basic current_price + PnL propagation; full
        conflict-resolution and validation is added in later phases.
        """
        try:
            position = await self.get_position(asset, mode)
            if position is None:
                # In Phase 3 we don't auto-create from WS; later phases handle that.
                logger.debug(
                    "ws_position_update_ignored_no_position",
                    asset=asset,
                    mode=mode,
                )
                return None

            # Simple field updates
            if mark_price is not None:
                position.current_price = mark_price
            if unrealized_pnl is not None:
                position.unrealized_pnl = unrealized_pnl
            if realized_pnl is not None:
                position.realized_pnl = realized_pnl

            pool = await DatabaseConnection.get_pool()
            update_query = """
                UPDATE positions
                SET current_price = $1,
                    unrealized_pnl = $2,
                    realized_pnl = $3,
                    last_updated = NOW()
                WHERE asset = $4 AND mode = $5
                RETURNING id, asset, mode, size, average_entry_price, current_price,
                          unrealized_pnl, realized_pnl,
                          long_size, short_size, version,
                          last_updated, closed_at, created_at
            """
            row = await pool.fetchrow(
                update_query,
                str(position.current_price) if position.current_price is not None else None,
                str(position.unrealized_pnl),
                str(position.realized_pnl),
                asset.upper(),
                mode.lower(),
            )
            if row:
                updated = Position.from_db_dict(dict(row))
            else:
                updated = position

            logger.info(
                "position_updated_from_websocket",
                asset=asset,
                mode=mode,
                current_price=str(updated.current_price)
                if updated.current_price is not None
                else None,
            )
            return updated
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_update_from_websocket_failed",
                asset=asset,
                mode=mode,
                error=str(e),
            )
            raise DatabaseError(f"Failed to update position from WebSocket: {e}") from e

    # === Validation & snapshot methods =====================================

    async def validate_position(
        self,
        asset: str,
        mode: str = "one-way",
        fix_discrepancies: bool = True,
    ) -> Tuple[bool, Optional[str], Optional[Position]]:
        """Phase 3 placeholder for validate_position.

        Detailed validation against authoritative sources is implemented in
        later phases. For now we simply check that a stored position exists.
        """
        try:
            stored = await self.get_position(asset, mode)
            if stored is None:
                msg = f"Position not found for asset={asset}, mode={mode}"
                logger.warning("position_validation_missing", asset=asset, mode=mode, error=msg)
                return False, msg, None
            logger.debug(
                "position_validation_passed_basic",
                asset=asset,
                mode=mode,
                size=str(stored.size),
            )
            return True, None, stored
        except Exception as e:  # pragma: no cover
            logger.error("position_validation_error", asset=asset, mode=mode, error=str(e))
            return False, f"Validation error: {e}", None

    async def create_position_snapshot(
        self,
        position: Position,
    ) -> PositionSnapshot:
        """Create a snapshot of the current position state into position_snapshots."""
        try:
            snapshot_payload = position.to_db_dict()
            pool = await DatabaseConnection.get_pool()
            insert_query = """
                INSERT INTO position_snapshots (
                    id, position_id, asset, mode, snapshot_data, created_at
                )
                VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())
                RETURNING id, position_id, asset, mode, snapshot_data, created_at
            """
            row = await pool.fetchrow(
                insert_query,
                str(position.id),
                position.asset,
                position.mode,
                snapshot_payload,
            )

            snapshot = PositionSnapshot.from_db_dict(dict(row))

            logger.info(
                "position_snapshot_created",
                position_id=str(position.id),
                asset=position.asset,
                mode=position.mode,
                snapshot_id=str(snapshot.id),
            )
            return snapshot
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_snapshot_creation_failed",
                position_id=str(position.id),
                asset=position.asset,
                mode=position.mode,
                error=str(e),
            )
            raise DatabaseError(f"Failed to create position snapshot: {e}") from e

    # === ML feature helpers (Phase 3) ======================================

    def calculate_unrealized_pnl_pct(self, position: Position) -> Optional[Decimal]:
        """Delegate to Position.computed field (helper for tests / services)."""
        return position.unrealized_pnl_pct

    def calculate_time_held_minutes(self, position: Position) -> Optional[int]:
        """Delegate to Position.computed field (helper for tests / services)."""
        return position.time_held_minutes

    def calculate_position_size_norm(
        self,
        position: Position,
        total_exposure: Decimal,
    ) -> Optional[Decimal]:
        """Calculate position_size_norm = abs(size * current_price) / total_exposure."""
        if position.current_price is None or total_exposure <= 0:
            return None
        try:
            exposure = abs(position.size * position.current_price)
            if exposure == 0:
                return Decimal("0")
            return exposure / total_exposure
        except Exception:  # pragma: no cover
            return None

    # === Filtering helpers (Phase 3) =======================================

    def filter_by_asset(self, positions: List[Position], asset: Optional[str]) -> List[Position]:
        if not asset:
            return positions
        asset_upper = asset.upper()
        return [p for p in positions if p.asset == asset_upper]

    def filter_by_mode(self, positions: List[Position], mode: Optional[str]) -> List[Position]:
        if not mode:
            return positions
        mode_lower = mode.lower()
        return [p for p in positions if p.mode == mode_lower]

    def filter_by_size(
        self,
        positions: List[Position],
        size_min: Optional[Decimal],
        size_max: Optional[Decimal],
    ) -> List[Position]:
        filtered: List[Position] = []
        for p in positions:
            if size_min is not None and p.size < size_min:
                continue
            if size_max is not None and p.size > size_max:
                continue
            filtered.append(p)
        return filtered



