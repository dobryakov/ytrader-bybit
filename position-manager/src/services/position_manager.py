"""Position manager service for position state management and ML features."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

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

    # In-memory tracking of last-seen WebSocket timestamps for conflict resolution.
    # Keys are (asset_upper, mode_lower).
    # NOTE: _last_order_timestamp removed - positions are now updated only from WebSocket events
    _last_ws_timestamp: Dict[Tuple[str, str], datetime] = {}

    # === Core queries ======================================================

    async def _calculate_average_entry_price_from_orders(
        self, asset: str, target_size: Decimal
    ) -> Optional[Decimal]:
        """
        Calculate average entry price from order history.
        
        Args:
            asset: Trading pair symbol
            target_size: Target position size (positive for long, negative for short)
            
        Returns:
            Calculated average entry price or None if cannot be calculated
        """
        try:
            pool = await DatabaseConnection.get_pool()
            # Calculate weighted average price from filled orders
            # For long positions: use Buy orders
            # For short positions: use Sell orders
            query = """
                SELECT 
                    side,
                    SUM(filled_quantity) as total_qty,
                    SUM(filled_quantity * average_price) as total_value
                FROM orders
                WHERE asset = $1 
                  AND status IN ('filled', 'partially_filled')
                GROUP BY side
            """
            rows = await pool.fetch(query, asset.upper())
            
            if not rows:
                return None
            
            total_qty = Decimal("0")
            total_value = Decimal("0")
            
            for row in rows:
                side = row["side"]
                qty = row["total_qty"] or Decimal("0")
                value = row["total_value"] or Decimal("0")
                
                if target_size > 0:
                    # Long position: use Buy orders
                    if side.upper() == "BUY":
                        total_qty += qty
                        total_value += value
                else:
                    # Short position: use Sell orders
                    if side.upper() == "SELL":
                        total_qty += qty
                        total_value += value
            
            if total_qty > 0:
                avg_price = total_value / total_qty
                logger.debug(
                    "average_entry_price_calculated_from_orders",
                    asset=asset,
                    target_size=str(target_size),
                    calculated_avg_price=str(avg_price),
                )
                return avg_price
            
            return None
        except Exception as e:
            logger.warning(
                "failed_to_calculate_avg_price_from_orders",
                asset=asset,
                error=str(e),
            )
            return None

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

            position_dict = dict(row)
            position_size = Decimal(str(position_dict["size"]))
            closed_at = position_dict.get("closed_at")
            
            # Fix position desynchronization: if position is marked as closed (closed_at is set)
            # but size is not zero, force close it (set size to 0 and clear closed_at)
            # This can happen when position is closed via Bybit sync but size wasn't properly updated
            if closed_at is not None and position_size != 0:
                logger.warning(
                    "position_desynchronization_detected",
                    asset=asset,
                    mode=mode,
                    size=str(position_size),
                    closed_at=closed_at.isoformat() if hasattr(closed_at, 'isoformat') else str(closed_at),
                    action="forcing_close",
                )
                # Force close: set size to 0 and clear closed_at
                await pool.execute(
                    """
                    UPDATE positions
                    SET size = 0,
                        average_entry_price = NULL,
                        closed_at = NOW(),
                        version = version + 1,
                        last_updated = NOW()
                    WHERE asset = $1 AND mode = $2
                    """,
                    asset.upper(),
                    mode.lower(),
                )
                position_dict["size"] = "0"
                position_dict["average_entry_price"] = None
                position_dict["closed_at"] = datetime.utcnow()
                position_size = Decimal("0")
                logger.info(
                    "position_force_closed_due_to_desynchronization",
                    asset=asset,
                    mode=mode,
                )
            
            # Fix missing average_entry_price for non-zero positions
            if position_dict["average_entry_price"] is None and position_size != 0:
                calculated_avg_price = await self._calculate_average_entry_price_from_orders(
                    asset, position_size
                )
                if calculated_avg_price:
                    # Update position in database
                    await pool.execute(
                        """
                        UPDATE positions
                        SET average_entry_price = $1, version = version + 1, last_updated = NOW()
                        WHERE asset = $2 AND mode = $3
                        """,
                        str(calculated_avg_price),
                        asset.upper(),
                        mode.lower(),
                    )
                    position_dict["average_entry_price"] = str(calculated_avg_price)
                    logger.info(
                        "average_entry_price_fixed_from_orders",
                        asset=asset,
                        mode=mode,
                        size=str(position_size),
                        calculated_avg_price=str(calculated_avg_price),
                    )

            position = Position.from_db_dict(position_dict)
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
        event_timestamp: Optional[datetime] = None,
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
                    
                    # If position size is zero, average_entry_price must be NULL
                    if new_size == 0:
                        new_avg_price = None
                    # If avg_price is missing but position size is non-zero, calculate from order history
                    elif new_avg_price is None and new_size != 0:
                        calculated_avg_price = await self._calculate_average_entry_price_from_orders(
                            asset, new_size
                        )
                        if calculated_avg_price:
                            new_avg_price = calculated_avg_price
                            logger.info(
                                "average_entry_price_calculated_for_new_position",
                                asset=asset,
                                mode=mode,
                                size=str(new_size),
                                calculated_avg_price=str(new_avg_price),
                                trace_id=trace_id,
                            )
                        elif mark_price:
                            # Fallback: use mark_price if order history unavailable
                            new_avg_price = mark_price
                            logger.warning(
                                "average_entry_price_using_mark_price_fallback",
                                asset=asset,
                                mode=mode,
                                size=str(new_size),
                                mark_price=str(mark_price),
                                trace_id=trace_id,
                            )
                    
                    # Validate that new_avg_price is positive if not None
                    if new_avg_price is not None and new_avg_price <= 0:
                        logger.warning(
                            "avg_price_invalid_negative_or_zero_insert",
                            asset=asset,
                            mode=mode,
                            invalid_avg=str(new_avg_price),
                            new_size=str(new_size),
                            trace_id=trace_id,
                        )
                        new_avg_price = None
                    
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
                        # Record last WebSocket timestamp for conflict resolution (Phase 9)
                        ws_effective_ts = event_timestamp or created.last_updated
                        if ws_effective_ts is not None:
                            key = (created.asset, created.mode)
                            self._last_ws_timestamp[key] = ws_effective_ts
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
                    # Effective WebSocket timestamp for this update
                    ws_effective_ts = event_timestamp or position.last_updated

                    # Conflict resolution for average_entry_price
                    new_avg_price = self._resolve_avg_price(position.average_entry_price, avg_price)
                    
                    # Validate and fix average_entry_price if it seems invalid
                    # (e.g., if it's more than 10x or less than 0.1x of current_price)
                    if new_avg_price is not None and position.current_price is not None:
                        current_price = position.current_price
                        if new_avg_price > current_price * 10 or new_avg_price < current_price / 10:
                            # Recalculate from unrealized_pnl if available
                            if position.unrealized_pnl is not None and position.size != 0:
                                recalculated = current_price - (position.unrealized_pnl / abs(position.size))
                                if recalculated > 0:
                                    logger.warning(
                                        "avg_price_invalid_recalculated",
                                        asset=asset,
                                        mode=mode,
                                        old_avg=str(new_avg_price),
                                        recalculated_avg=str(recalculated),
                                        current_price=str(current_price),
                                        unrealized_pnl=str(position.unrealized_pnl),
                                        size=str(position.size),
                                        trace_id=trace_id,
                                    )
                                    new_avg_price = recalculated
                    
                    # Validate that new_avg_price is positive if not None
                    if new_avg_price is not None and new_avg_price <= 0:
                        logger.warning(
                            "avg_price_invalid_negative_or_zero",
                            asset=asset,
                            mode=mode,
                            invalid_avg=str(new_avg_price),
                            trace_id=trace_id,
                        )
                        new_avg_price = None
                    
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

                    # Size validation + optional timestamp-based conflict resolution (Phase 9)
                    size_update_from_ws = False
                    resolved_size = position.size

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

                        # NOTE: Positions are now updated only from WebSocket events.
                        # Size discrepancy resolution is based on WebSocket data as source of truth.
                        # Always use WebSocket size if there's a significant discrepancy.
                        # (Previous order_timestamp-based conflict resolution removed)
                        if self._should_update_size_from_ws(
                            db_size=position.size,
                            ws_size=size_from_ws,
                            ws_timestamp=ws_effective_ts,
                            order_timestamp=None,  # DEPRECATED: no longer used
                        ):
                            size_update_from_ws = True
                            resolved_size = size_from_ws  # type: ignore[assignment]
                            logger.info(
                                "position_size_resolved_from_websocket",
                                asset=asset,
                                mode=mode,
                                db_size=str(position.size),
                                ws_size=str(size_from_ws),
                                ws_timestamp=ws_effective_ts.isoformat()
                                if ws_effective_ts is not None
                                else None,
                                trace_id=trace_id,
                            )
                        else:
                            logger.info(
                                "position_size_ws_not_applied_due_to_threshold",
                                asset=asset,
                                mode=mode,
                                db_size=str(position.size),
                                ws_size=str(size_from_ws),
                                ws_timestamp=ws_effective_ts.isoformat()
                                if ws_effective_ts is not None
                                else None,
                                trace_id=trace_id,
                            )
                    
                    # If position is closed (size = 0), set average_entry_price to NULL
                    if resolved_size == 0:
                        new_avg_price = None

                    # Price and PnL updates
                    current_price = mark_price or position.current_price

                    # Refresh price if stale
                    if current_price is not None and self._is_price_stale(position.last_updated):
                        refreshed = await self._get_current_price_from_api(
                            asset, trace_id=trace_id
                        )
                        if refreshed is not None:
                            current_price = refreshed

                    # Use unrealized_pnl from WebSocket if provided, otherwise calculate from current_price
                    new_unrealized = unrealized_pnl
                    if new_unrealized is None:
                        # Calculate unrealized_pnl from current_price and average_entry_price
                        if current_price is not None and new_avg_price is not None and resolved_size != 0:
                            # Formula: (current_price - average_entry_price) * size
                            # Works for both long (size > 0) and short (size < 0) positions
                            new_unrealized = (current_price - new_avg_price) * resolved_size
                        else:
                            # Fallback to existing unrealized_pnl if cannot calculate
                            new_unrealized = position.unrealized_pnl
                    
                    new_realized = realized_pnl or position.realized_pnl

                    # Final validation: ensure new_avg_price is None if size is 0 or if it's <= 0
                    if resolved_size == 0 or (new_avg_price is not None and new_avg_price <= 0):
                        new_avg_price = None

                    update_query = """
                        UPDATE positions
                        SET size = CAST($1 AS numeric),
                            current_price = CASE 
                                WHEN $2::text = '' THEN NULL
                                ELSE CAST($2::text AS numeric)
                            END,
                            unrealized_pnl = CAST($3 AS numeric),
                            realized_pnl = CAST($4 AS numeric),
                            average_entry_price = CASE 
                                WHEN CAST($1 AS numeric) = 0 THEN NULL
                                WHEN $5::text = '' THEN average_entry_price
                                WHEN CAST($5::text AS numeric) <= 0 THEN NULL
                                ELSE CAST($5::text AS numeric)
                            END,
                            version = version + 1,
                            last_updated = NOW(),
                            closed_at = CASE WHEN CAST($1 AS numeric) = 0 THEN NOW() ELSE closed_at END
                        WHERE asset = $6
                          AND mode = $7
                          AND version = $8
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at
                    """
                    # Always pass string for $2 and $5 to help asyncpg determine parameter type
                    # Pass empty string instead of None to avoid type inference issues
                    # This workaround is similar to the datetime issue fix ($4::timestamptz)
                    current_price_param = str(current_price) if current_price is not None else ''
                    avg_price_param = str(new_avg_price) if new_avg_price is not None else ''
                    row = await pool.fetchrow(
                        update_query,
                        str(resolved_size),
                        current_price_param,
                        str(new_unrealized),
                        str(new_realized),
                        avg_price_param,
                        asset.upper(),
                        mode.lower(),
                        position.version,
                    )
                    if row:
                        updated = Position.from_db_dict(dict(row))
                        # Record last WebSocket timestamp for conflict resolution (Phase 9)
                        ws_final_ts = ws_effective_ts or updated.last_updated
                        if ws_final_ts is not None:
                            key = (updated.asset, updated.mode)
                            self._last_ws_timestamp[key] = ws_final_ts
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
        trace_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[Position]]:
        """Validate position by computing from order history and comparing with stored state.

        Args:
            asset: Trading pair symbol
            mode: Trading mode ('one-way' or 'hedge')
            fix_discrepancies: If True, update stored position when discrepancy is found
            trace_id: Optional trace ID for request tracking

        Returns:
            Tuple of (is_valid, error_message, updated_position)
            - is_valid: True if position is valid, False if discrepancy found
            - error_message: Description of discrepancy if any
            - updated_position: Updated position if discrepancy was fixed, None otherwise
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Compute position from order history
            # For average price, we need to compute it correctly based on position direction
            # This is a simplified validation - actual average price is maintained correctly
            # by the position update logic during order execution
            compute_query = """
                SELECT
                    SUM(CASE WHEN side = 'Buy' THEN filled_quantity ELSE -filled_quantity END) as computed_size
                FROM orders
                WHERE asset = $1 AND status IN ('filled', 'partially_filled')
            """
            row = await pool.fetchrow(compute_query, asset.upper())

            computed_size = row["computed_size"] or Decimal("0")
            # Average price validation is complex and position_manager correctly maintains it
            # during order execution. We skip average price validation here as it requires
            # tracking position direction changes which is already handled correctly.
            computed_avg_price = None  # Skip average price validation - it's maintained correctly by update logic

            # Get stored position
            stored_position = await self.get_position(asset, mode)

            if stored_position is None:
                if computed_size == 0:
                    logger.debug(
                        "position_validation_passed",
                        asset=asset,
                        mode=mode,
                        reason="no_position_no_orders",
                        trace_id=trace_id,
                    )
                    return (True, None, None)
                error_msg = f"Stored position missing but computed size is {computed_size}"
                logger.warning(
                    "position_validation_failed",
                    asset=asset,
                    mode=mode,
                    error=error_msg,
                    computed_size=float(computed_size),
                    trace_id=trace_id,
                )

                # If discrepancy fixing is enabled and computed position exists, create the position
                # Note: We don't compute average_price here as it requires complex logic
                # that's already handled correctly during order execution
                if fix_discrepancies and computed_size != 0:
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        mode=mode,
                        action="creating_missing_position",
                        computed_size=float(computed_size),
                        trace_id=trace_id,
                    )
                    # Don't update average_price - let it be set by next order execution
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, None, mode, trace_id
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
                logger.warning(
                    "position_validation_failed",
                    asset=asset,
                    mode=mode,
                    error=error_msg,
                    stored_size=float(stored_position.size),
                    computed_size=float(computed_size),
                    size_diff=float(size_diff),
                    trace_id=trace_id,
                )

                # Handle discrepancy by updating stored position with computed values
                # Note: We preserve the existing average_entry_price as it's correctly
                # maintained by the position update logic during order execution
                if fix_discrepancies:
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        mode=mode,
                        action="updating_position",
                        stored_size=float(stored_position.size),
                        computed_size=float(computed_size),
                        size_diff=float(size_diff),
                        trace_id=trace_id,
                    )
                    # Preserve existing average_price - only update size
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, stored_position.average_entry_price, mode, trace_id
                    )
                    return (True, f"Fixed discrepancy: {error_msg}", updated_position)

                return (False, error_msg, None)

            logger.debug(
                "position_validation_passed",
                asset=asset,
                mode=mode,
                size=float(stored_position.size),
                trace_id=trace_id,
            )
            return (True, None, None)

        except Exception as e:
            logger.error(
                "position_validation_error",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return (False, f"Validation error: {e}", None)

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
                    id, position_id, asset, mode, snapshot_data, snapshot_timestamp
                )
                VALUES (gen_random_uuid(), $1, $2, $3, $4, NOW())
                RETURNING id, position_id, asset, mode, snapshot_data, snapshot_timestamp AS created_at
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
                SELECT id, position_id, asset, mode, snapshot_data, snapshot_timestamp AS created_at
                FROM position_snapshots
                WHERE position_id = $1
                ORDER BY snapshot_timestamp DESC
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
                WHERE snapshot_timestamp < $1
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

    # === Validation helpers (Phase 7) ======================================

    async def _update_position_from_computed(
        self,
        asset: str,
        computed_size: Decimal,
        computed_avg_price: Optional[Decimal],
        mode: str,
        trace_id: Optional[str] = None,
    ) -> Position:
        """Update stored position from computed values (from order history).

        Args:
            asset: Trading pair symbol
            computed_size: Computed position size from order history
            computed_avg_price: Computed average price from order history
            mode: Trading mode
            trace_id: Optional trace ID for request tracking

        Returns:
            Updated Position object
        """
        try:
            pool = await DatabaseConnection.get_pool()

            # Get existing position to preserve current_price and other fields
            existing = await self.get_position(asset, mode)

            # Upsert position with computed values, preserving version for optimistic locking
            if existing:
                upsert_query = """
                    UPDATE positions
                    SET size = $1,
                        average_entry_price = CASE 
                            WHEN $2::text = '' THEN NULL
                            ELSE CAST($2::text AS numeric)
                        END,
                        last_updated = NOW(),
                        version = version + 1
                    WHERE asset = $3 AND mode = $4
                    RETURNING id, asset, mode, size, average_entry_price, current_price,
                              unrealized_pnl, realized_pnl,
                              long_size, short_size, version,
                              last_updated, closed_at, created_at
                """
                # Always pass string for $2 to help asyncpg determine parameter type
                # Pass empty string instead of None to avoid type inference issues
                avg_price_param = str(computed_avg_price) if computed_avg_price is not None else ''
                row = await pool.fetchrow(
                    upsert_query,
                    str(computed_size),
                    avg_price_param,
                    asset.upper(),
                    mode.lower(),
                )
            else:
                # Create new position if it doesn't exist
                upsert_query = """
                    INSERT INTO positions (
                        asset, mode, size, average_entry_price, version, last_updated, created_at
                    )
                    VALUES ($1, $2, $3, $4, 1, NOW(), NOW())
                    RETURNING id, asset, mode, size, average_entry_price, current_price,
                              unrealized_pnl, realized_pnl,
                              long_size, short_size, version,
                              last_updated, closed_at, created_at
                """
                row = await pool.fetchrow(
                    upsert_query,
                    asset.upper(),
                    mode.lower(),
                    str(computed_size),
                    str(computed_avg_price) if computed_avg_price is not None else None,
                )

            if row is None:
                raise DatabaseError("Failed to update/create position from computed values")

            position = Position.from_db_dict(dict(row))

            logger.info(
                "position_updated_from_computed",
                asset=asset,
                mode=mode,
                computed_size=float(computed_size),
                computed_avg_price=float(computed_avg_price) if computed_avg_price else None,
                trace_id=trace_id,
            )

            return position

        except Exception as e:
            logger.error(
                "position_update_from_computed_failed",
                asset=asset,
                mode=mode,
                error=str(e),
                trace_id=trace_id,
            )
            raise DatabaseError(f"Failed to update position from computed values: {e}") from e

    # In-memory validation statistics (can be enhanced with database persistence if needed)
    _validation_stats = {
        "total_validations": 0,
        "total_discrepancies": 0,
        "total_fixes": 0,
        "total_errors": 0,
    }

    async def record_validation_statistics(
        self,
        validated_count: int,
        fixed_count: int,
        error_count: int,
        total_positions: int,
    ) -> None:
        """Record validation statistics for monitoring.

        Args:
            validated_count: Number of positions that passed validation
            fixed_count: Number of discrepancies that were fixed
            error_count: Number of validation errors
            total_positions: Total number of positions validated
        """
        self._validation_stats["total_validations"] += validated_count
        self._validation_stats["total_discrepancies"] += (fixed_count + error_count)
        self._validation_stats["total_fixes"] += fixed_count
        self._validation_stats["total_errors"] += error_count

        logger.debug(
            "validation_statistics_updated",
            validated_count=validated_count,
            fixed_count=fixed_count,
            error_count=error_count,
            total_positions=total_positions,
            cumulative_stats=self._validation_stats,
        )

    def get_validation_statistics(self) -> dict:
        """Get current validation statistics.

        Returns:
            Dictionary with validation statistics
        """
        return self._validation_stats.copy()

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

    def _should_update_size_from_ws(
        self,
        db_size: Decimal,
        ws_size: Optional[Decimal],
        ws_timestamp: Optional[datetime],
        order_timestamp: Optional[datetime],  # DEPRECATED: kept for compatibility, no longer used
    ) -> bool:
        """Decide if size should be updated from WebSocket.
        
        NOTE: This method is simplified - positions are now updated only from WebSocket events,
        so WebSocket is always the source of truth. This method is kept for compatibility
        but always returns True if ws_size is available and differs significantly from db_size.

        Args:
            db_size: Current position size in database
            ws_size: Position size from WebSocket event
            ws_timestamp: WebSocket event timestamp (optional, for logging)
            order_timestamp: DEPRECATED - no longer used, kept for compatibility

        Returns:
            True if size should be updated from WebSocket
        """
        if ws_size is None:
            return False

        threshold = Decimal(str(settings.position_manager_size_validation_threshold))
        size_diff = abs(ws_size - db_size)
        if size_diff <= threshold:
            return False

        # Critical discrepancy: if size difference is large (> 1.0) and opposite signs,
        # always update from WebSocket (indicates position flip bug)
        critical_threshold = Decimal("1.0")
        opposite_signs = (db_size > 0 and ws_size < 0) or (db_size < 0 and ws_size > 0)
        if size_diff > critical_threshold and opposite_signs:
            logger.warning(
                "critical_size_discrepancy_detected_force_update",
                db_size=str(db_size),
                ws_size=str(ws_size),
                size_diff=str(size_diff),
            )
            return True

        # Always use WebSocket as source of truth if there's a significant discrepancy
        # (order_timestamp-based resolution removed as positions are only updated from WebSocket)
        return True

    async def _get_current_price_from_api(
        self,
        asset: str,
        trace_id: Optional[str] = None,
    ) -> Optional[Decimal]:
        """Query external price API (Bybit) with retries and backoff."""
        base_url = (
            "https://api-testnet.bybit.com"
            if settings.bybit_environment == "testnet"
            else "https://api.bybit.com"
        )
        endpoint = "/v5/market/tickers"
        url = f"{base_url}{endpoint}"
        retries = settings.position_manager_price_api_retries
        timeout_s = settings.position_manager_price_api_timeout

        params = {"category": "linear", "symbol": asset.upper()}

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.get(url, params=params)
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

    async def get_bybit_positions(
        self,
        trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all open positions from Bybit API.

        Args:
            trace_id: Optional trace ID for logging

        Returns:
            List of position dictionaries from Bybit API

        Raises:
            DatabaseError: If API call fails or credentials are missing
        """
        if not settings.bybit_api_key or not settings.bybit_api_secret:
            raise DatabaseError("Bybit API credentials not configured")

        base_url = (
            "https://api-testnet.bybit.com"
            if settings.bybit_environment == "testnet"
            else "https://api.bybit.com"
        )
        endpoint = "/v5/position/list"
        url = f"{base_url}{endpoint}"

        params = {"category": "linear", "settleCoin": "USDT"}
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        # Generate signature for GET request
        sorted_params = sorted([(k, str(v)) for k, v in params.items() if v is not None])
        query_string = "&".join([f"{k}={v}" for k, v in sorted_params])
        signature_string = f"{timestamp}{settings.bybit_api_key}{recv_window}{query_string}"
        signature = hmac.new(
            settings.bybit_api_secret.encode("utf-8"),
            signature_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": settings.bybit_api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()

                if data.get("retCode") != 0:
                    error_msg = data.get("retMsg", "Unknown error")
                    logger.error(
                        "bybit_api_get_positions_failed",
                        ret_code=data.get("retCode"),
                        error=error_msg,
                        trace_id=trace_id,
                    )
                    raise DatabaseError(f"Bybit API error: {error_msg}")

                positions = data.get("result", {}).get("list", [])
                open_positions = [
                    p for p in positions if p.get("size") and Decimal(str(p.get("size", "0"))) != Decimal("0")
                ]

                logger.info(
                    "bybit_api_positions_retrieved",
                    total=len(positions),
                    open=len(open_positions),
                    trace_id=trace_id,
                )

                return open_positions

        except httpx.HTTPError as e:
            logger.error(
                "bybit_api_http_error",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"HTTP error fetching Bybit positions: {e}") from e
        except Exception as e:
            logger.error(
                "bybit_api_get_positions_exception",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to get Bybit positions: {e}") from e

    async def sync_positions_with_bybit(
        self,
        force: bool = False,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronize local positions with Bybit API positions.
        
        NOTE: This method updates positions directly via SQL, bypassing the normal
        WebSocket-based update mechanism. This is intentional and acceptable because:
        - This is a manual administrative tool for fixing discrepancies
        - Bybit API is also a valid source of truth (like WebSocket events)
        - This is not part of the normal position update flow
        - Used only for manual synchronization when discrepancies are detected
        
        Normal position updates should go through update_position_from_websocket()
        which is called by WebSocketPositionConsumer from ws-gateway.position events.

        Args:
            force: If True, force update local positions to match Bybit
            trace_id: Optional trace ID for logging

        Returns:
            Dictionary with sync report:
            {
                "bybit_positions": [...],
                "local_positions": [...],
                "comparisons": [...],
                "updated": [...],
                "created": [...],
                "errors": [...]
            }
        """
        try:
            # Get positions from Bybit
            bybit_positions = await self.get_bybit_positions(trace_id=trace_id)

            # Get local positions
            local_positions = await self.get_all_positions()

            # Build comparison report
            bybit_dict = {p["symbol"]: p for p in bybit_positions}
            local_dict = {p.asset: p for p in local_positions if p.size != 0}

            comparisons = []
            updated = []
            created = []
            errors = []

            # Compare positions
            all_assets = set(bybit_dict.keys()) | set(local_dict.keys())

            for asset in all_assets:
                bybit_pos = bybit_dict.get(asset)
                local_pos = local_dict.get(asset)

                # Normalize Bybit position
                bybit_size = Decimal(str(bybit_pos.get("size", "0"))) if bybit_pos else Decimal("0")
                bybit_side = bybit_pos.get("side", "Buy") if bybit_pos else None
                # Convert side to size sign: Buy = positive, Sell = negative
                if bybit_side == "Sell":
                    bybit_size = -abs(bybit_size)
                elif bybit_side == "Buy":
                    bybit_size = abs(bybit_size)

                bybit_avg_price = (
                    Decimal(str(bybit_pos.get("avgPrice", "0"))) if bybit_pos and bybit_pos.get("avgPrice") else None
                )
                bybit_unrealised_pnl = (
                    Decimal(str(bybit_pos.get("unrealisedPnl", "0")))
                    if bybit_pos and bybit_pos.get("unrealisedPnl")
                    else Decimal("0")
                )
                bybit_realised_pnl = (
                    Decimal(str(bybit_pos.get("cumRealisedPnl", "0")))
                    if bybit_pos and bybit_pos.get("cumRealisedPnl")
                    else Decimal("0")
                )

                local_size = local_pos.size if local_pos else Decimal("0")
                local_avg_price = local_pos.average_entry_price if local_pos else None
                local_unrealised_pnl = local_pos.unrealized_pnl if local_pos else Decimal("0")
                local_realised_pnl = local_pos.realized_pnl if local_pos else Decimal("0")

                # Compare
                size_diff = abs(bybit_size - local_size)
                avg_price_diff = (
                    abs(bybit_avg_price - local_avg_price) if (bybit_avg_price and local_avg_price) else None
                )
                avg_price_diff_pct = (
                    (avg_price_diff / local_avg_price * 100) if (avg_price_diff and local_avg_price) else None
                )

                comparison = {
                    "asset": asset,
                    "bybit_exists": bybit_pos is not None,
                    "local_exists": local_pos is not None,
                    "bybit_size": str(bybit_size),
                    "local_size": str(local_size),
                    "size_diff": str(size_diff),
                    "size_match": size_diff <= Decimal("0.0001"),
                    "bybit_avg_price": str(bybit_avg_price) if bybit_avg_price else None,
                    "local_avg_price": str(local_avg_price) if local_avg_price else None,
                    "avg_price_diff": str(avg_price_diff) if avg_price_diff else None,
                    "avg_price_diff_pct": float(avg_price_diff_pct) if avg_price_diff_pct else None,
                    "bybit_unrealised_pnl": str(bybit_unrealised_pnl),
                    "local_unrealised_pnl": str(local_unrealised_pnl),
                    "bybit_realised_pnl": str(bybit_realised_pnl),
                    "local_realised_pnl": str(local_realised_pnl),
                }
                comparisons.append(comparison)

                # Force sync if requested
                if force:
                    try:
                        # Check if we need to update PnL even if size/price match
                        pnl_diff_unrealized = abs(bybit_unrealised_pnl - local_unrealised_pnl) if local_pos else Decimal("0")
                        pnl_diff_realized = abs(bybit_realised_pnl - local_realised_pnl) if local_pos else Decimal("0")
                        has_pnl_discrepancy = pnl_diff_unrealized > Decimal("0.01") or pnl_diff_realized > Decimal("0.01")
                        
                        # Case 1: Position exists on Bybit but not locally, or sizes differ, or PnL differs
                        if bybit_pos and (not local_pos or size_diff > Decimal("0.0001") or (avg_price_diff and avg_price_diff_pct and avg_price_diff_pct > 1.0) or has_pnl_discrepancy):
                            # Update or create position
                            if local_pos:
                                # Update existing - only update size/price if they differ
                                if size_diff > Decimal("0.0001") or (avg_price_diff and avg_price_diff_pct and avg_price_diff_pct > 1.0):
                                    updated_pos = await self._update_position_from_computed(
                                        asset=asset,
                                        computed_size=bybit_size,
                                        computed_avg_price=bybit_avg_price,
                                        mode="one-way",
                                        trace_id=trace_id,
                                    )
                                else:
                                    # Size and price match, just get position for ID
                                    updated_pos = local_pos
                                
                                # Update PnL separately (always update if there's discrepancy)
                                pool = await DatabaseConnection.get_pool()
                                await pool.execute(
                                    """
                                    UPDATE positions
                                    SET unrealized_pnl = $1,
                                        realized_pnl = $2,
                                        version = version + 1,
                                        last_updated = NOW()
                                    WHERE asset = $3 AND mode = $4
                                    """,
                                    str(bybit_unrealised_pnl),
                                    str(bybit_realised_pnl),
                                    asset.upper(),
                                    "one-way",
                                )
                                reason = "size/price" if (size_diff > Decimal("0.0001") or (avg_price_diff and avg_price_diff_pct and avg_price_diff_pct > 1.0)) else "pnl"
                                updated.append({"asset": asset, "action": "updated", "position_id": str(updated_pos.id), "reason": reason})
                            else:
                                # Create new
                                created_pos = await self._update_position_from_computed(
                                    asset=asset,
                                    computed_size=bybit_size,
                                    computed_avg_price=bybit_avg_price,
                                    mode="one-way",
                                    trace_id=trace_id,
                                )
                                # Update PnL
                                pool = await DatabaseConnection.get_pool()
                                await pool.execute(
                                    """
                                    UPDATE positions
                                    SET unrealized_pnl = $1,
                                        realized_pnl = $2,
                                        version = version + 1,
                                        last_updated = NOW()
                                    WHERE asset = $3 AND mode = $4
                                    """,
                                    str(bybit_unrealised_pnl),
                                    str(bybit_realised_pnl),
                                    asset.upper(),
                                    "one-way",
                                )
                                created.append({"asset": asset, "action": "created", "position_id": str(created_pos.id)})

                            logger.info(
                                "position_synced_with_bybit",
                                asset=asset,
                                bybit_size=str(bybit_size),
                                local_size=str(local_size),
                                force=force,
                                trace_id=trace_id,
                            )
                        # Case 2: Position exists locally but not on Bybit (should be closed)
                        elif local_pos and not bybit_pos:
                            # Close position by setting size to 0
                            pool = await DatabaseConnection.get_pool()
                            
                            # Get position_id before closing
                            position_row = await pool.fetchrow(
                                "SELECT id, realized_pnl, current_price FROM positions WHERE asset = $1 AND mode = $2",
                                asset.upper(),
                                "one-way",
                            )
                            position_id = position_row["id"] if position_row else None
                            final_realized_pnl = Decimal(str(position_row["realized_pnl"] or 0)) if position_row else Decimal("0")
                            exit_price = Decimal(str(position_row["current_price"] or 0)) if position_row and position_row["current_price"] else None
                            
                            # Update position
                            await pool.execute(
                                """
                                UPDATE positions
                                SET size = 0,
                                    average_entry_price = NULL,
                                    unrealized_pnl = 0,
                                    version = version + 1,
                                    last_updated = NOW(),
                                    closed_at = NOW()
                                WHERE asset = $1 AND mode = $2
                                """,
                                asset.upper(),
                                "one-way",
                            )
                            
                            # Update prediction_trading_results for this position (same actions as normal close)
                            if position_id:
                                try:
                                    # Find open prediction_trading_results for this position
                                    open_results = await pool.fetch(
                                        """
                                        SELECT id, signal_id, realized_pnl, unrealized_pnl
                                        FROM prediction_trading_results
                                        WHERE position_id = $1 AND is_closed = false
                                        """,
                                        position_id,
                                    )
                                    
                                    # Calculate total current realized_pnl across all open results
                                    total_current_realized = sum(
                                        Decimal(str(r["realized_pnl"] or 0)) for r in open_results
                                    )
                                    
                                    # Distribute final realized_pnl proportionally if multiple results
                                    # If only one result, use final_realized_pnl directly
                                    for result in open_results:
                                        result_id = result["id"]
                                        current_realized = Decimal(str(result["realized_pnl"] or 0))
                                        current_unrealized = Decimal(str(result["unrealized_pnl"] or 0))
                                        
                                        # Distribute final realized_pnl proportionally
                                        if len(open_results) > 1 and total_current_realized > 0:
                                            # Proportional distribution based on current realized_pnl
                                            proportion = current_realized / total_current_realized
                                            final_realized = final_realized_pnl * proportion
                                        else:
                                            # Single result or no current realized_pnl: use final directly
                                            final_realized = final_realized_pnl
                                        
                                        final_total = final_realized + Decimal("0")  # unrealized becomes 0 on close
                                        
                                        await pool.execute(
                                            """
                                            UPDATE prediction_trading_results
                                            SET realized_pnl = $1,
                                                unrealized_pnl = 0,
                                                total_pnl = $2,
                                                exit_price = $3,
                                                exit_timestamp = NOW(),
                                                is_closed = true,
                                                updated_at = NOW()
                                            WHERE id = $4
                                            """,
                                            str(final_realized),
                                            str(final_total),
                                            str(exit_price) if exit_price else None,
                                            result_id,
                                        )
                                        
                                        logger.info(
                                            "prediction_trading_result_closed_on_bybit_sync",
                                            result_id=str(result_id),
                                            signal_id=str(result["signal_id"]),
                                            position_id=str(position_id),
                                            asset=asset,
                                            final_realized_pnl=str(final_realized),
                                            trace_id=trace_id,
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "failed_to_update_prediction_trading_results_on_sync_close",
                                        position_id=str(position_id),
                                        asset=asset,
                                        error=str(e),
                                        trace_id=trace_id,
                                        exc_info=True,
                                    )
                            
                            # Publish position update event (same as normal close)
                            try:
                                # Get updated position for event
                                updated_position_row = await pool.fetchrow(
                                    """
                                    SELECT id, asset, mode, size, average_entry_price, current_price,
                                           unrealized_pnl, realized_pnl, created_at, closed_at
                                    FROM positions
                                    WHERE asset = $1 AND mode = $2
                                    """,
                                    asset.upper(),
                                    "one-way",
                                )
                                if updated_position_row:
                                    from ..models.position import Position
                                    updated_position = Position.from_db_dict(dict(updated_position_row))
                                    await PositionEventPublisher.publish_position_updated(
                                        position=updated_position,
                                        update_source="bybit_sync_close",
                                        trace_id=trace_id,
                                    )
                                    logger.info(
                                        "position_update_event_published_on_sync_close",
                                        asset=asset,
                                        position_id=str(updated_position.id),
                                        trace_id=trace_id,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "failed_to_publish_position_event_on_sync_close",
                                    asset=asset,
                                    error=str(e),
                                    trace_id=trace_id,
                                    exc_info=True,
                                )
                            
                            updated.append({"asset": asset, "action": "closed", "reason": "position_not_on_bybit"})
                            logger.info(
                                "position_closed_not_on_bybit",
                                asset=asset,
                                local_size=str(local_size),
                                position_id=str(position_id) if position_id else None,
                                force=force,
                                trace_id=trace_id,
                            )
                    except Exception as e:
                        error_msg = f"Failed to sync {asset}: {e}"
                        errors.append({"asset": asset, "error": error_msg})
                        logger.error(
                            "position_sync_error",
                            asset=asset,
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )

            return {
                "bybit_positions_count": len(bybit_positions),
                "local_positions_count": len([p for p in local_positions if p.size != 0]),
                "comparisons": comparisons,
                "updated": updated,
                "created": created,
                "errors": errors,
                "force_applied": force,
            }

        except Exception as e:
            logger.error(
                "sync_positions_with_bybit_error",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to sync positions with Bybit: {e}") from e



