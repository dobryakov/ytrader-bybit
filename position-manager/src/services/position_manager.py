"""Position manager service for position state management and ML features."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple

import httpx

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..exceptions import DatabaseError
from ..models import Position, PositionSnapshot
from ..publishers import PositionEventPublisher

logger = get_logger(__name__)

# Re-export httpx symbol for easier monkeypatching in unit tests.
__all__ = ["PositionManager", "httpx"]


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
        execution_fees: Optional[Decimal] = None,
        mode: str = "one-way",
    ) -> Position:
        """Update position based on order execution.

        This is a direct adaptation of Order Manager's position update logic,
        extended to work with the shared `positions` schema and optimistic
        locking semantics (version field).
        """
        max_retries = settings.position_manager_optimistic_lock_retries
        backoff_base_ms = settings.position_manager_optimistic_lock_backoff_base

        try:
            pool = await DatabaseConnection.get_pool()

            for attempt in range(max_retries):
                # Fetch the current position (if any)
                current_position = await self.get_position(asset, mode)

                if current_position is None:
                    # Position creation on first order update
                    new_size = size_delta
                    new_avg_price: Optional[Decimal] = execution_price if new_size != 0 else None
                    version = 1

                    insert_query = """
                        INSERT INTO positions (
                            asset, mode, size, average_entry_price,
                            unrealized_pnl, realized_pnl,
                            current_price, version, last_updated, created_at
                        )
                        VALUES ($1, $2, $3, $4, 0, 0, NULL, $5, NOW(), NOW())
                        ON CONFLICT (asset, mode) DO NOTHING
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at
                    """
                    row = await pool.fetchrow(
                        insert_query,
                        asset.upper(),
                        mode.lower(),
                        str(new_size),
                        str(new_avg_price) if new_avg_price is not None else None,
                        version,
                    )
                    if row:
                        position = Position.from_db_dict(dict(row))
                        logger.info(
                            "position_created_from_order_fill",
                            asset=asset,
                            mode=mode,
                            size_delta=str(size_delta),
                            execution_price=str(execution_price),
                            new_size=str(position.size),
                        )
                        # Invalidate portfolio cache on new position
                        try:
                            from .portfolio_manager import default_portfolio_manager

                            default_portfolio_manager.invalidate_cache()
                        except Exception:  # pragma: no cover
                            logger.warning("portfolio_cache_invalidation_failed_from_order_fill")
                        # Best-effort publish position + portfolio events
                        try:
                            await PositionEventPublisher.publish_position_updated(
                                position=position,
                                update_source="order_execution",
                                trace_id=None,
                            )
                        except Exception:  # pragma: no cover
                            logger.warning("position_event_publish_failed_from_order_fill_create")
                        return position

                    # If insert did not return a row, a concurrent creator won; retry.
                    logger.warning(
                        "position_order_fill_insert_conflict_retry",
                        asset=asset,
                        mode=mode,
                        attempt=attempt + 1,
                    )
                else:
                    # Existing position - apply execution-based update rules
                    current_size = current_position.size
                    current_avg_price = (
                        current_position.average_entry_price or execution_price
                    )

                    new_size = current_size + size_delta

                    # Realized PnL for the portion that is being closed
                    realized_pnl_delta = Decimal("0")
                    if (current_size > 0 and size_delta < 0) or (
                        current_size < 0 and size_delta > 0
                    ):
                        closed_qty = min(abs(current_size), abs(size_delta))
                        fees = execution_fees or Decimal("0")
                        if current_size > 0:
                            # Long: profit when execution_price > avg_price
                            realized_pnl_delta = (execution_price - current_avg_price) * closed_qty
                        else:
                            # Short: profit when execution_price < avg_price
                            realized_pnl_delta = (current_avg_price - execution_price) * closed_qty
                        realized_pnl_delta -= fees

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

                    # Closed position handling
                    closed_at_expr = "NOW()" if new_size == 0 else "NULL"

                    update_query = f"""
                        UPDATE positions
                        SET size = $1,
                            average_entry_price = $2,
                            realized_pnl = realized_pnl + $3,
                            version = version + 1,
                            last_updated = NOW(),
                            closed_at = {closed_at_expr}
                        WHERE asset = $4
                          AND mode = $5
                          AND version = $6
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at
                    """
                    row = await pool.fetchrow(
                        update_query,
                        str(new_size),
                        str(new_avg_price) if new_avg_price is not None else None,
                        str(realized_pnl_delta),
                        asset.upper(),
                        mode.lower(),
                        current_position.version,
                    )
                    if row:
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
                        try:
                            from .portfolio_manager import default_portfolio_manager

                            default_portfolio_manager.invalidate_cache()
                        except Exception:  # pragma: no cover
                            logger.warning("portfolio_cache_invalidation_failed_from_order_fill")
                        # Best-effort publish position event
                        try:
                            await PositionEventPublisher.publish_position_updated(
                                position=position,
                                update_source="order_execution",
                                trace_id=None,
                            )
                        except Exception:  # pragma: no cover
                            logger.warning("position_event_publish_failed_from_order_fill_update")
                        return position

                    # Conflict: someone updated the row first; retry with backoff.
                    delay_ms = backoff_base_ms * (2**attempt)
                    logger.warning(
                        "position_order_fill_optimistic_lock_conflict",
                        asset=asset,
                        mode=mode,
                        attempt=attempt + 1,
                        delay_ms=delay_ms,
                    )
                    await asyncio.sleep(delay_ms / 1000.0)

            logger.error(
                "position_order_fill_optimistic_lock_failed",
                asset=asset,
                mode=mode,
                retries=max_retries,
            )
            raise DatabaseError(
                f"Failed to update position from order fill after {max_retries} optimistic-lock retries"
            )
        except DatabaseError:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_update_from_order_fill_failed",
                asset=asset,
                mode=mode,
                size_delta=str(size_delta),
                error=str(e),
            )
            raise DatabaseError(f"Failed to update position from order fill: {e}") from e

    # === WebSocket-based update (Phase 4 enhanced) ==========================

    async def update_position_from_websocket(
        self,
        asset: str,
        mode: str,
        mark_price: Optional[Decimal] = None,
        avg_price: Optional[Decimal] = None,
        size_from_ws: Optional[Decimal] = None,
        unrealized_pnl: Optional[Decimal] = None,
        realized_pnl: Optional[Decimal] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[Position]:
        """Update position from WebSocket event.

        Phase 4 enhancement:
        - Use mark_price as current_price
        - Optionally reconcile avgPrice vs stored average_entry_price
          using POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD
        - Validate size_from_ws vs stored size using
          POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD (log-only)
        - Recalculate ML-related helpers after update
        - Use optimistic locking on version field with retry
        """
        max_retries = settings.position_manager_optimistic_lock_retries
        backoff_base_ms = settings.position_manager_optimistic_lock_backoff_base

        try:
            pool = await DatabaseConnection.get_pool()

            for attempt in range(max_retries):
                position = await self.get_position(asset, mode)

                if position is None:
                    # Position creation on first WebSocket update
                    if mark_price is None:
                        # Try external API if mark_price missing
                        mark_price = await self._get_current_price_from_api(asset, trace_id=trace_id)

                    if mark_price is None:
                        logger.warning(
                            "ws_position_update_missing_price_cannot_create",
                            asset=asset,
                            mode=mode,
                            trace_id=trace_id,
                        )
                        return None

                    # When created from WS, we rely on WebSocket unrealisedPnl & avgPrice
                    new_size = size_from_ws or Decimal("0")
                    new_avg_price = avg_price
                    unreal = unrealized_pnl or Decimal("0")
                    realized = realized_pnl or Decimal("0")

                    insert_query = """
                        INSERT INTO positions (
                            asset, mode, size, average_entry_price,
                            unrealized_pnl, realized_pnl,
                            current_price, version, last_updated, created_at
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, 1, NOW(), NOW())
                        ON CONFLICT (asset, mode) DO NOTHING
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at
                    """
                    row = await pool.fetchrow(
                        insert_query,
                        asset.upper(),
                        mode.lower(),
                        str(new_size),
                        str(new_avg_price) if new_avg_price is not None else None,
                        str(unreal),
                        str(realized),
                        str(mark_price) if mark_price is not None else None,
                    )
                    if row:
                        created = Position.from_db_dict(dict(row))
                        logger.info(
                            "position_created_from_websocket",
                            asset=asset,
                            mode=mode,
                            size=str(created.size),
                            current_price=str(created.current_price)
                            if created.current_price is not None
                            else None,
                            trace_id=trace_id,
                        )
                        try:
                            from .portfolio_manager import default_portfolio_manager

                            default_portfolio_manager.invalidate_cache()
                        except Exception:  # pragma: no cover
                            logger.warning("portfolio_cache_invalidation_failed_from_ws")
                        # Best-effort publish position event
                        try:
                            await PositionEventPublisher.publish_position_updated(
                                position=created,
                                update_source="websocket",
                                trace_id=trace_id,
                            )
                        except Exception:  # pragma: no cover
                            logger.warning("position_event_publish_failed_from_ws_create")
                        return created

                    logger.warning(
                        "ws_position_create_conflict_retry",
                        asset=asset,
                        mode=mode,
                        attempt=attempt + 1,
                        trace_id=trace_id,
                    )
                else:
                    # Conflict resolution for average_entry_price
                    new_avg_price = self._resolve_avg_price(position.average_entry_price, avg_price)
                    if (
                        position.average_entry_price is not None
                        and avg_price is not None
                        and new_avg_price == avg_price
                    ):
                        logger.info(
                            "avg_price_conflict_resolved",
                            asset=asset,
                            mode=mode,
                            old_avg=str(position.average_entry_price),
                            new_avg=str(avg_price),
                            trace_id=trace_id,
                        )

                    # Size validation (Order Manager is source of truth)
                    if self._has_size_discrepancy(position.size, size_from_ws):
                        size_diff = abs(size_from_ws - position.size)  # type: ignore[arg-type]
                        threshold = Decimal(
                            str(settings.position_manager_size_validation_threshold)
                        )
                        logger.warning(
                            "position_size_discrepancy_detected",
                            asset=asset,
                            mode=mode,
                            db_size=str(position.size),
                            ws_size=str(size_from_ws),
                            diff=str(size_diff),
                            threshold=str(threshold),
                            trace_id=trace_id,
                        )

                    # Price and PnL updates
                    current_price = mark_price or position.current_price

                    # Refresh price if stale
                    if current_price is not None and self._is_price_stale(position.last_updated):
                        refreshed = await self._get_current_price_from_api(
                            asset, trace_id=trace_id
                        )
                        if refreshed is not None:
                            current_price = refreshed

                    new_unrealized = unrealized_pnl or position.unrealized_pnl
                    new_realized = realized_pnl or position.realized_pnl

                    update_query = """
                        UPDATE positions
                        SET current_price = $1,
                            unrealized_pnl = $2,
                            realized_pnl = $3,
                            average_entry_price = COALESCE($4, average_entry_price),
                            version = version + 1,
                            last_updated = NOW()
                        WHERE asset = $5
                          AND mode = $6
                          AND version = $7
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at
                    """
                    row = await pool.fetchrow(
                        update_query,
                        str(current_price) if current_price is not None else None,
                        str(new_unrealized),
                        str(new_realized),
                        str(new_avg_price) if new_avg_price is not None else None,
                        asset.upper(),
                        mode.lower(),
                        position.version,
                    )
                    if row:
                        updated = Position.from_db_dict(dict(row))
                        logger.info(
                            "position_updated_from_websocket",
                            asset=asset,
                            mode=mode,
                            current_price=str(updated.current_price)
                            if updated.current_price is not None
                            else None,
                            unrealized_pnl=str(updated.unrealized_pnl),
                            realized_pnl=str(updated.realized_pnl),
                            trace_id=trace_id,
                        )
                        try:
                            from .portfolio_manager import default_portfolio_manager

                            default_portfolio_manager.invalidate_cache()
                        except Exception:  # pragma: no cover
                            logger.warning("portfolio_cache_invalidation_failed_from_ws")
                        # Best-effort publish position event
                        try:
                            await PositionEventPublisher.publish_position_updated(
                                position=updated,
                                update_source="websocket",
                                trace_id=trace_id,
                            )
                        except Exception:  # pragma: no cover
                            logger.warning("position_event_publish_failed_from_ws_update")
                        return updated

                    delay_ms = backoff_base_ms * (2**attempt)
                    logger.warning(
                        "position_ws_optimistic_lock_conflict",
                        asset=asset,
                        mode=mode,
                        attempt=attempt + 1,
                        delay_ms=delay_ms,
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(delay_ms / 1000.0)

            logger.error(
                "position_ws_optimistic_lock_failed",
                asset=asset,
                mode=mode,
                retries=max_retries,
                trace_id=trace_id,
            )
            raise DatabaseError(
                f"Failed to update position from WebSocket after {max_retries} optimistic-lock retries"
            )
        except DatabaseError:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_update_from_websocket_failed",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
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
        trace_id: Optional[str] = None,
    ) -> PositionSnapshot:
        """Create a snapshot of the current position state into position_snapshots.

        The snapshot payload is stored in the JSONB ``snapshot_data`` column and
        contains all DB fields plus ML features (where available) so that past
        states can be reconstructed for analytics and model training.
        """
        try:
            # Base DB payload
            snapshot_payload = position.to_db_dict()
            # Enrich with ML features for historical analysis
            snapshot_payload.update(
                {
                    "unrealized_pnl_pct": str(position.unrealized_pnl_pct)
                    if position.unrealized_pnl_pct is not None
                    else None,
                    "time_held_minutes": position.time_held_minutes,
                    # position_size_norm depends on portfolio exposure; include key
                    # for schema completeness even if not populated here.
                    "position_size_norm": None,
                }
            )

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
                trace_id=trace_id,
            )

            # Best-effort event publishing; failures should not break request.
            try:
                await PositionEventPublisher.publish_snapshot_created(
                    snapshot=snapshot,
                    trace_id=trace_id,
                )
            except Exception:  # pragma: no cover - best-effort path
                logger.warning(
                    "position_snapshot_event_publish_failed",
                    snapshot_id=str(snapshot.id),
                    position_id=str(position.id),
                    trace_id=trace_id,
                )

            return snapshot
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_snapshot_creation_failed",
                position_id=str(position.id),
                asset=position.asset,
                mode=position.mode,
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to create position snapshot: {e}") from e

    # === Snapshot history & cleanup (Phase 6) =================================

    async def get_position_snapshots(
        self,
        position_id,
        limit: int,
        offset: int,
    ) -> List[PositionSnapshot]:
        """Return snapshots for a given position ordered by created_at DESC."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, position_id, asset, mode, snapshot_data, created_at
                FROM position_snapshots
                WHERE position_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """
            rows = await pool.fetch(query, str(position_id), limit, offset)
            snapshots = [PositionSnapshot.from_db_dict(dict(row)) for row in rows]
            logger.debug(
                "position_snapshots_retrieved",
                position_id=str(position_id),
                count=len(snapshots),
            )
            return snapshots
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_snapshots_query_failed",
                position_id=str(position_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to query position snapshots: {e}") from e

    async def cleanup_old_snapshots(self) -> int:
        """Delete snapshots older than the configured retention period.

        Returns the number of deleted rows.
        """
        try:
            retention_days = settings.position_manager_snapshot_retention_days
            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            pool = await DatabaseConnection.get_pool()
            query = """
                DELETE FROM position_snapshots
                WHERE created_at < $1
            """
            command_tag = await pool.execute(query, cutoff)
            # asyncpg returns tags like "DELETE 42"
            try:
                deleted = int(command_tag.split()[-1])
            except Exception:  # pragma: no cover - extremely defensive
                deleted = 0

            logger.info(
                "position_snapshots_cleanup_completed",
                retention_days=retention_days,
                deleted=deleted,
            )
            return deleted
        except Exception as e:  # pragma: no cover
            logger.error(
                "position_snapshots_cleanup_failed",
                error=str(e),
            )
            raise DatabaseError(f"Failed to cleanup old position snapshots: {e}") from e

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

    # === External price & staleness helpers (Phase 4) =======================

    def _is_price_stale(self, last_updated: datetime) -> bool:
        """Check if current_price is stale based on POSITION_MANAGER_PRICE_STALENESS_THRESHOLD."""
        threshold_seconds = settings.position_manager_price_staleness_threshold
        return datetime.utcnow() - last_updated > timedelta(seconds=threshold_seconds)

    # Small pure helpers to simplify reasoning and unit testing of US2 conflict
    # resolution rules (T061).

    def _resolve_avg_price(
        self,
        existing_avg_price: Optional[Decimal],
        ws_avg_price: Optional[Decimal],
    ) -> Optional[Decimal]:
        """Resolve average_entry_price based on WebSocket avgPrice and threshold.

        Returns the value that should be persisted (either existing or ws_avg).
        """
        if (
            not settings.position_manager_use_ws_avg_price
            or existing_avg_price is None
            or ws_avg_price is None
        ):
            return existing_avg_price or ws_avg_price

        threshold = Decimal(str(settings.position_manager_avg_price_diff_threshold))
        diff_ratio = abs(ws_avg_price - existing_avg_price) / existing_avg_price
        if diff_ratio > threshold:
            return ws_avg_price
        return existing_avg_price

    def _has_size_discrepancy(
        self,
        db_size: Decimal,
        ws_size: Optional[Decimal],
    ) -> bool:
        """Check if WebSocket size differs from DB size beyond validation threshold."""
        if ws_size is None:
            return False
        threshold = Decimal(str(settings.position_manager_size_validation_threshold))
        size_diff = abs(ws_size - db_size)
        return size_diff > threshold

    async def _get_current_price_from_api(
        self,
        asset: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Decimal]:
        """Query external price API (Bybit) with retries and backoff."""
        base_url = "https://api.bybit.com/v5/market/tickers"
        retries = settings.position_manager_price_api_retries
        timeout_s = settings.position_manager_price_api_timeout

        params = {"symbol": asset.upper()}

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                # Minimal parsing: expect first ticker entry with "markPrice" or "lastPrice"
                result = data.get("result") or {}
                list_ = result.get("list") or []
                if not list_:
                    raise ValueError("Empty tickers list from price API")
                ticker = list_[0]
                price_str = ticker.get("markPrice") or ticker.get("lastPrice")
                if not price_str:
                    raise ValueError("Price field missing in ticker payload")
                price = Decimal(str(price_str))
                logger.info(
                    "external_price_api_success",
                    asset=asset,
                    price=str(price),
                    attempt=attempt + 1,
                    trace_id=trace_id,
                )
                return price
            except Exception as e:  # pragma: no cover - network failures
                if attempt < retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "external_price_api_retry",
                        asset=asset,
                        attempt=attempt + 1,
                        retries=retries,
                        delay=delay,
                        error=str(e),
                        trace_id=trace_id,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "external_price_api_failed",
                        asset=asset,
                        retries=retries,
                        error=str(e),
                        trace_id=trace_id,
                    )
                    return None




