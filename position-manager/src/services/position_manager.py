"""Position manager service for position state management and ML features."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
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

    @staticmethod
    def _convert_datetime_to_iso_string(data: Any) -> Any:
        """
        Recursively convert all datetime objects to ISO format strings.
        
        This is needed for JSON serialization, as datetime objects cannot be
        directly serialized to JSON. All datetime objects are converted to ISO
        format strings with timezone information.
        
        Args:
            data: Dictionary, list, or other data structure that may contain datetime objects
            
        Returns:
            Data structure with all datetime objects converted to ISO format strings
        """
        if data is None:
            return None
        
        if isinstance(data, datetime):
            # Convert to ISO format string with timezone
            return data.isoformat()
        
        if isinstance(data, dict):
            return {
                key: PositionManager._convert_datetime_to_iso_string(value)
                for key, value in data.items()
            }
        
        if isinstance(data, (list, tuple)):
            converted_list = [
                PositionManager._convert_datetime_to_iso_string(item)
                for item in data
            ]
            return type(data)(converted_list) if isinstance(data, tuple) else converted_list
        
        return data

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

    async def get_active_position(self, asset: str, mode: str = "one-way") -> Optional[Position]:
        """Get current active position for an asset/mode pair.
        
        Returns only active positions (closed_at IS NULL).
        This is an internal method - use get_position() for public API.
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, long_avg_price, short_avg_price,
                       leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                       cum_realised_pnl, cum_unrealised_pnl,
                       total_fees, opening_fees, closing_fees,
                       margin_used, available_margin, maintenance_margin,
                       max_size, min_size, total_volume_traded,
                       first_entry_price, last_entry_price, exit_price,
                       peak_unrealized_pnl, peak_unrealized_pnl_at,
                       worst_unrealized_pnl, worst_unrealized_pnl_at,
                       created_at, last_updated, closed_at,
                       version, source, last_sync_with_bybit, bybit_position_data
                FROM positions
                WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
            """
            row = await pool.fetchrow(query, asset.upper(), mode.lower())
            if row is None:
                logger.debug("active_position_not_found", asset=asset, mode=mode)
                return None

            position_dict = dict(row)
            position_size = Decimal(str(position_dict["size"]))
            
            # Fix position desynchronization: if size is zero but position is active (closed_at IS NULL),
            # this is incorrect state - should be closed
            if position_size == 0:
                logger.warning(
                    "active_position_with_zero_size",
                    asset=asset,
                    mode=mode,
                    action="closing_position",
                )
                # Close the position
                await pool.execute(
                    """
                    UPDATE positions
                    SET closed_at = NOW(),
                        exit_price = current_price,
                        version = version + 1,
                        last_updated = NOW()
                    WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                    """,
                    asset.upper(),
                    mode.lower(),
                )
                # Return None as position is now closed
                return None
            
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

    async def get_position(self, asset: str, mode: str = "one-way") -> Optional[Position]:
        """Get current active position for an asset/mode pair (public API).
        
        Returns only active positions (closed_at IS NULL).
        For historical positions, use get_position_history() or get_closed_position_by_id().
        """
        return await self.get_active_position(asset, mode)

    async def get_all_active_positions(self) -> List[Position]:
        """Get all active positions (closed_at IS NULL).
        
        This is an internal method - use get_all_positions() for public API.
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, long_avg_price, short_avg_price,
                       leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                       cum_realised_pnl, cum_unrealised_pnl,
                       total_fees, opening_fees, closing_fees,
                       margin_used, available_margin, maintenance_margin,
                       max_size, min_size, total_volume_traded,
                       first_entry_price, last_entry_price, exit_price,
                       peak_unrealized_pnl, peak_unrealized_pnl_at,
                       worst_unrealized_pnl, worst_unrealized_pnl_at,
                       created_at, last_updated, closed_at,
                       version, source, last_sync_with_bybit, bybit_position_data
                FROM positions
                WHERE closed_at IS NULL AND size != 0
                ORDER BY asset, mode
            """
            rows = await pool.fetch(query)
            positions = []
            for row in rows:
                try:
                    position = Position.from_db_dict(dict(row))
                    positions.append(position)
                except Exception as e:
                    # Skip positions with validation errors
                    row_dict = dict(row)
                    logger.warning(
                        "position_validation_error_skipped",
                        asset=row_dict.get("asset"),
                        mode=row_dict.get("mode"),
                        size=str(row_dict.get("size")),
                        average_entry_price=str(row_dict.get("average_entry_price")) if row_dict.get("average_entry_price") else None,
                        error=str(e),
                    )
            logger.debug("all_active_positions_retrieved", count=len(positions), skipped=len(rows) - len(positions))
            return positions
        except Exception as e:  # pragma: no cover
            logger.error("get_all_active_positions_failed", error=str(e))
            raise DatabaseError(f"Failed to retrieve all active positions: {e}") from e

    async def get_all_positions(self) -> List[Position]:
        """Get all active positions from the database (public API).
        
        Returns only active positions (closed_at IS NULL) by default.
        For historical positions, use get_position_history().
        """
        return await self.get_all_active_positions()

    async def get_position_history(
        self,
        asset: str,
        mode: str = "one-way",
        limit: int = 100,
        offset: int = 0,
    ) -> List[Position]:
        """Get history of closed positions for an asset/mode pair.
        
        Args:
            asset: Trading pair symbol
            mode: Trading mode
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of closed Position objects ordered by closed_at DESC
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, long_avg_price, short_avg_price,
                       leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                       cum_realised_pnl, cum_unrealised_pnl,
                       total_fees, opening_fees, closing_fees,
                       margin_used, available_margin, maintenance_margin,
                       max_size, min_size, total_volume_traded,
                       first_entry_price, last_entry_price, exit_price,
                       peak_unrealized_pnl, peak_unrealized_pnl_at,
                       worst_unrealized_pnl, worst_unrealized_pnl_at,
                       created_at, last_updated, closed_at,
                       version, source, last_sync_with_bybit, bybit_position_data
                FROM positions
                WHERE asset = $1 AND mode = $2 AND closed_at IS NOT NULL
                ORDER BY closed_at DESC
                LIMIT $3 OFFSET $4
            """
            rows = await pool.fetch(query, asset.upper(), mode.lower(), limit, offset)
            positions = [Position.from_db_dict(dict(row)) for row in rows]
            logger.debug(
                "position_history_retrieved",
                asset=asset,
                mode=mode,
                count=len(positions),
            )
            return positions
        except Exception as e:
            logger.error(
                "position_history_query_failed",
                asset=asset,
                mode=mode,
                error=str(e),
            )
            raise DatabaseError(f"Failed to query position history: {e}") from e

    async def get_closed_position_by_id(self, position_id: UUID) -> Optional[Position]:
        """Get a closed position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Closed Position object or None if not found
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, long_avg_price, short_avg_price,
                       leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                       cum_realised_pnl, cum_unrealised_pnl,
                       total_fees, opening_fees, closing_fees,
                       margin_used, available_margin, maintenance_margin,
                       max_size, min_size, total_volume_traded,
                       first_entry_price, last_entry_price, exit_price,
                       peak_unrealized_pnl, peak_unrealized_pnl_at,
                       worst_unrealized_pnl, worst_unrealized_pnl_at,
                       created_at, last_updated, closed_at,
                       version, source, last_sync_with_bybit, bybit_position_data
                FROM positions
                WHERE id = $1 AND closed_at IS NOT NULL
            """
            row = await pool.fetchrow(query, str(position_id))
            if row is None:
                logger.debug("closed_position_not_found", position_id=str(position_id))
                return None
            
            position = Position.from_db_dict(dict(row))
            logger.debug("closed_position_retrieved", position_id=str(position_id))
            return position
        except Exception as e:
            logger.error(
                "closed_position_query_failed",
                position_id=str(position_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to query closed position: {e}") from e

    async def get_position_by_id(
        self,
        position_id: UUID,
    ) -> Optional[Position]:
        """Get a position by its ID (active or closed).
        
        Args:
            position_id: UUID of the position
            
        Returns:
            Position object (active or closed) or None if not found
        """
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT id, asset, mode, size, average_entry_price, current_price,
                       unrealized_pnl, realized_pnl,
                       long_size, short_size, long_avg_price, short_avg_price,
                       leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                       cum_realised_pnl, cum_unrealised_pnl,
                       total_fees, opening_fees, closing_fees,
                       margin_used, available_margin, maintenance_margin,
                       max_size, min_size, total_volume_traded,
                       first_entry_price, last_entry_price, exit_price,
                       peak_unrealized_pnl, peak_unrealized_pnl_at,
                       worst_unrealized_pnl, worst_unrealized_pnl_at,
                       created_at, last_updated, closed_at,
                       version, source, last_sync_with_bybit, bybit_position_data
                FROM positions
                WHERE id = $1
            """
            row = await pool.fetchrow(query, str(position_id))
            if row is None:
                logger.debug("position_not_found_by_id", position_id=str(position_id))
                return None
            
            position = Position.from_db_dict(dict(row))
            logger.debug("position_retrieved_by_id", position_id=str(position_id), is_closed=position.is_closed)
            return position
        except Exception as e:
            logger.error(
                "position_query_by_id_failed",
                position_id=str(position_id),
                error=str(e),
            )
            raise DatabaseError(f"Failed to query position by ID: {e}") from e

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
                # Use transaction with SELECT FOR UPDATE for race condition protection
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        # Check for existing active position with lock
                        existing_row = await conn.fetchrow(
                            """
                            SELECT id, size, version, closed_at
                            FROM positions
                            WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                            FOR UPDATE
                            """,
                            asset.upper(),
                            mode.lower(),
                        )

                        if existing_row is None:
                            # No active position - create new one or handle reopen
                            # Check if there's a closed position (reopen scenario)
                            closed_row = await conn.fetchrow(
                                """
                                SELECT id, closed_at
                                FROM positions
                                WHERE asset = $1 AND mode = $2 AND closed_at IS NOT NULL
                                ORDER BY closed_at DESC
                                LIMIT 1
                                """,
                                asset.upper(),
                                mode.lower(),
                            )
                            
                            # Position creation on first WebSocket update or reopen
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
                            
                            # If position size is zero, don't create position
                            if new_size == 0:
                                return None
                            
                            # If avg_price is missing but position size is non-zero, calculate from order history
                            if new_avg_price is None and new_size != 0:
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

                            # Create new position (reopen scenario)
                            insert_query = """
                                INSERT INTO positions (
                                    asset, mode, size, average_entry_price,
                                    unrealized_pnl, realized_pnl, total_fees,
                                    current_price, version, last_updated, 
                                    created_at, closed_at
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, 0, $7, 1, NOW(), NOW(), NULL)
                                RETURNING id, asset, mode, size, average_entry_price, current_price,
                                          unrealized_pnl, realized_pnl,
                                          long_size, short_size, long_avg_price, short_avg_price,
                                          leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                                          cum_realised_pnl, cum_unrealised_pnl,
                                          total_fees, opening_fees, closing_fees,
                                          margin_used, available_margin, maintenance_margin,
                                          max_size, min_size, total_volume_traded,
                                          first_entry_price, last_entry_price, exit_price,
                                          peak_unrealized_pnl, peak_unrealized_pnl_at,
                                          worst_unrealized_pnl, worst_unrealized_pnl_at,
                                          created_at, last_updated, closed_at,
                                          version, source, last_sync_with_bybit, bybit_position_data
                            """
                            try:
                                row = await conn.fetchrow(
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
                                    # Record last WebSocket timestamp for conflict resolution
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
                            except Exception as e:
                                # Handle UniqueViolationError (race condition - active position created by another process)
                                if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                                    logger.warning(
                                        "active_position_exists_on_reopen",
                                        asset=asset,
                                        mode=mode,
                                        reason="Race condition: active position found during reopen",
                                        trace_id=trace_id,
                                    )
                                    # Position was created by another process - retry to get it
                                    # Break out of INSERT block and continue to UPDATE path
                                    pass
                                else:
                                    raise
                        
                        # Active position exists (either found initially or created by another process)
                        # Read it with lock for update
                        position_row = await conn.fetchrow(
                            """
                            SELECT id, asset, mode, size, average_entry_price, current_price,
                                   unrealized_pnl, realized_pnl,
                                   long_size, short_size, long_avg_price, short_avg_price,
                                   leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                                   cum_realised_pnl, cum_unrealised_pnl,
                                   total_fees, opening_fees, closing_fees,
                                   margin_used, available_margin, maintenance_margin,
                                   max_size, min_size, total_volume_traded,
                                   first_entry_price, last_entry_price, exit_price,
                                   peak_unrealized_pnl, peak_unrealized_pnl_at,
                                   worst_unrealized_pnl, worst_unrealized_pnl_at,
                                   created_at, last_updated, closed_at,
                                   version, source, last_sync_with_bybit, bybit_position_data
                            FROM positions
                            WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                            FOR UPDATE
                            """,
                            asset.upper(),
                            mode.lower(),
                        )
                        
                        if position_row is None:
                            # Position was closed between check and read, or still doesn't exist
                            # If we just tried to create it and got UniqueViolationError, retry
                            if attempt < max_retries - 1:
                                await asyncio.sleep(backoff_base_ms / 1000.0 * (2 ** attempt))
                                continue
                            # If position size is zero, don't create position
                            if size_from_ws is None or size_from_ws == 0:
                                return None
                            # Last attempt: try one more time to get position
                            # If still not found, return None (position might have been closed)
                            return None
                        
                        position = Position.from_db_dict(dict(position_row))
                        
                        # Check if position is closed (size = 0) - should not happen for active position
                        if position.size == 0:
                            # Auto-fix: close the position
                            await conn.execute(
                                """
                                UPDATE positions
                                SET closed_at = NOW(),
                                    exit_price = current_price,
                                    version = version + 1,
                                    last_updated = NOW()
                                WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                                """,
                                asset.upper(),
                                mode.lower(),
                            )
                            return None
                        
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
                        # Also set current_price to NULL for closed positions
                        current_price = None
                    else:
                        # Price and PnL updates
                        current_price = mark_price or position.current_price

                        # If current_price is NULL, try to fetch it from API
                        if current_price is None:
                            refreshed = await self._get_current_price_from_api(
                                asset, trace_id=trace_id
                            )
                            if refreshed is not None:
                                current_price = refreshed
                        # Otherwise, refresh price if stale
                        elif self._is_price_stale(position.last_updated):
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
                    
                    # Handle realized_pnl: validate value from WebSocket for open positions
                    # cumRealisedPnl from Bybit can be cumulative across all positions for the asset,
                    # so for open positions (size != 0) we need to validate it's reasonable
                    if realized_pnl is not None:
                        # For open positions, validate that realized_pnl is reasonable
                        # It should not be orders of magnitude larger than unrealized_pnl
                        if resolved_size != 0:
                            # Calculate expected maximum reasonable realized_pnl
                            # For an open position, realized_pnl should be small compared to position value
                            position_value = abs(resolved_size * (new_avg_price or current_price or Decimal("1")))
                            # Allow realized_pnl up to 10x position value (for partial closes)
                            max_reasonable_realized = abs(position_value * Decimal("10"))
                            
                            if abs(realized_pnl) > max_reasonable_realized:
                                # Value is unreasonably large, likely cumulative from previous positions
                                # For open positions, reset to 0 (existing value might also be incorrect)
                                logger.warning(
                                    "realized_pnl_unreasonable_for_open_position",
                                    asset=asset,
                                    mode=mode,
                                    size=str(resolved_size),
                                    realized_pnl_from_ws=str(realized_pnl),
                                    existing_realized_pnl=str(position.realized_pnl),
                                    max_reasonable=str(max_reasonable_realized),
                                    position_value=str(position_value),
                                    action="resetting_to_zero",
                                    trace_id=trace_id,
                                )
                                # Reset to 0 for open positions when value is unreasonable
                                # This handles cases where existing value is also incorrect (cumulative from previous positions)
                                new_realized = Decimal("0")
                            else:
                                # Value is reasonable, use it
                                new_realized = realized_pnl
                        else:
                            # Position is closed (size == 0)
                            # Note: This new_realized is used for updating positions table.
                                # For closed positions, we use unrealized_pnl_at_close instead
                                # (see logic below when closing position)
                            new_realized = realized_pnl
                    else:
                        # No realized_pnl from WebSocket, keep existing value
                        new_realized = position.realized_pnl or Decimal("0")

                    # Final validation: ensure new_avg_price is None if size is 0 or if it's <= 0
                    if resolved_size == 0 or (new_avg_price is not None and new_avg_price <= 0):
                        new_avg_price = None

                        # If position is being closed (size becomes 0, but wasn't 0 before), close it
                    if resolved_size == 0 and position.size != 0:
                        # Get exit_price: use mark_price if available, otherwise current_price from position
                        exit_price = mark_price if mark_price is not None else position.current_price
                        # Get average_entry_price before it's cleared (use existing position value)
                        entry_price = position.average_entry_price
                        # Use unrealized_pnl from current position (before closing), not the new one (which will be 0)
                        unrealized_at_close = position.unrealized_pnl or Decimal("0")
                        
                        # For a fully closed position, realized_pnl should be the unrealized_pnl_at_close
                        # because all unrealized PnL becomes realized when position is closed.
                        # We don't use cumulative realized_pnl from Bybit (new_realized) as it's cumulative
                        # across all positions for the asset, not specific to this position.
                        position_realized_pnl = unrealized_at_close
                        
                        # Get total_fees if available
                        total_fees = position.total_fees

                        # Close position with optimistic locking
                        close_result = await conn.execute(
                                """
                                UPDATE positions
                                SET size = 0,
                                    closed_at = NOW(),
                                    exit_price = CASE 
                                        WHEN $1::text = '' THEN NULL
                                        ELSE CAST($1::text AS numeric)
                                    END,
                                    average_entry_price = NULL,
                                    current_price = NULL,
                                    unrealized_pnl = 0,
                                    realized_pnl = CAST($2 AS numeric),
                                    total_fees = CASE 
                                        WHEN $3::text = '' THEN 0
                                        ELSE CAST($3::text AS numeric)
                                    END,
                                    version = version + 1,
                                    last_updated = NOW()
                                WHERE asset = $4 
                                  AND mode = $5 
                                  AND closed_at IS NULL
                                  AND version = $6
                                """,
                                str(exit_price) if exit_price is not None else '',
                                str(position_realized_pnl),
                                str(total_fees) if total_fees is not None else '',
                            asset.upper(),
                            mode.lower(),
                                position.version,
                            )
                            
                        if close_result == "UPDATE 0":
                            # Version conflict - retry
                            if attempt < max_retries - 1:
                                await asyncio.sleep(backoff_base_ms / 1000.0 * (2 ** attempt))
                                continue
                            logger.error(
                                "position_close_version_conflict",
                                asset=asset,
                                mode=mode,
                                trace_id=trace_id,
                            )
                            return None
                        
                        # Position closed successfully
                        closed_position = await self.get_closed_position_by_id(position.id)
                        if closed_position:
                            logger.info(
                                "position_closed_from_websocket",
                                asset=asset,
                                mode=mode,
                                exit_price=str(exit_price) if exit_price else None,
                                realized_pnl=str(position_realized_pnl),
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
                                    position=closed_position,
                                    update_source="websocket",
                                    trace_id=trace_id,
                                )
                            except Exception:  # pragma: no cover
                                logger.warning("position_event_publish_failed_from_ws_close")
                            return closed_position
                        return None

                        # Position opening is tracked via created_at (each record is created once)
                        
                        # Update position with optimistic locking
                        update_query = f"""
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
                            total_fees = CASE 
                                WHEN CAST($1 AS numeric) = 0 THEN 0
                                ELSE total_fees
                            END,
                            version = version + 1,
                            last_updated = NOW(),
                            closed_at = CASE 
                                WHEN CAST($1 AS numeric) = 0 THEN NOW() 
                                ELSE NULL 
                                END
                            WHERE asset = $6
                              AND mode = $7
                              AND closed_at IS NULL
                              AND version = $8
                            RETURNING id, asset, mode, size, average_entry_price, current_price,
                                      unrealized_pnl, realized_pnl,
                                      long_size, short_size, long_avg_price, short_avg_price,
                                      leverage, position_value, liq_price, bust_price, take_profit, stop_loss,
                                      cum_realised_pnl, cum_unrealised_pnl,
                                      total_fees, opening_fees, closing_fees,
                                      margin_used, available_margin, maintenance_margin,
                                      max_size, min_size, total_volume_traded,
                                      first_entry_price, last_entry_price, exit_price,
                                      peak_unrealized_pnl, peak_unrealized_pnl_at,
                                      worst_unrealized_pnl, worst_unrealized_pnl_at,
                                      created_at, last_updated, closed_at,
                                      version, source, last_sync_with_bybit, bybit_position_data
                        """
                        # Always pass string for $2 and $5 to help asyncpg determine parameter type
                        # Pass empty string instead of None to avoid type inference issues
                        # This workaround is similar to the datetime issue fix ($4::timestamptz)
                        current_price_param = str(current_price) if current_price is not None else ''
                        avg_price_param = str(new_avg_price) if new_avg_price is not None else ''
                        row = await conn.fetchrow(
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
                                size=str(updated.size),
                                resolved_size=str(resolved_size),
                                ws_size=str(size_from_ws) if size_from_ws is not None else None,
                                size_update_from_ws=size_update_from_ws,
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
                        if attempt < max_retries - 1:
                            await asyncio.sleep(delay_ms / 1000.0)
                            continue
                        else:
                            logger.error(
                                "position_ws_update_max_retries_exceeded",
                                asset=asset,
                                mode=mode,
                                max_retries=max_retries,
                                trace_id=trace_id,
                            )
                            return None
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
                            # Also set current_price to NULL for closed positions
                            current_price = None
                        else:
                            # Price and PnL updates
                            current_price = mark_price or position.current_price

                            # If current_price is NULL, try to fetch it from API
                            if current_price is None:
                                refreshed = await self._get_current_price_from_api(
                                    asset, trace_id=trace_id
                                )
                                if refreshed is not None:
                                    current_price = refreshed
                            # Otherwise, refresh price if stale
                            elif self._is_price_stale(position.last_updated):
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
                        
                        # Handle realized_pnl: validate value from WebSocket for open positions
                        # cumRealisedPnl from Bybit can be cumulative across all positions for the asset,
                        # so for open positions (size != 0) we need to validate it's reasonable
                        if realized_pnl is not None:
                            # For open positions, validate that realized_pnl is reasonable
                            # It should not be orders of magnitude larger than unrealized_pnl
                            if resolved_size != 0:
                                # Calculate expected maximum reasonable realized_pnl
                                # For an open position, realized_pnl should be small compared to position value
                                position_value = abs(resolved_size * (new_avg_price or current_price or Decimal("1")))
                                # Allow realized_pnl up to 10x position value (for partial closes)
                                max_reasonable_realized = abs(position_value * Decimal("10"))
                                
                                if abs(realized_pnl) > max_reasonable_realized:
                                    # Value is unreasonably large, likely cumulative from previous positions
                                    # For open positions, reset to 0 (existing value might also be incorrect)
                                    logger.warning(
                                        "realized_pnl_unreasonable_for_open_position",
                                        asset=asset,
                                        mode=mode,
                                        size=str(resolved_size),
                                        realized_pnl_from_ws=str(realized_pnl),
                                        existing_realized_pnl=str(position.realized_pnl),
                                        max_reasonable=str(max_reasonable_realized),
                                        position_value=str(position_value),
                                        action="resetting_to_zero",
                                        trace_id=trace_id,
                                    )
                                    # Reset to 0 for open positions when value is unreasonable
                                    # This handles cases where existing value is also incorrect (cumulative from previous positions)
                                    new_realized = Decimal("0")
                                else:
                                    # Value is reasonable, use it
                                    new_realized = realized_pnl
                            else:
                                # Position is closed (size == 0)
                                # Note: This new_realized is used for updating positions table.
                                # Closed positions are stored in the same positions table with closed_at IS NOT NULL
                                new_realized = realized_pnl
                        else:
                            # No realized_pnl from WebSocket, keep existing value
                            new_realized = position.realized_pnl or Decimal("0")

                        # Final validation: ensure new_avg_price is None if size is 0 or if it's <= 0
                        if resolved_size == 0 or (new_avg_price is not None and new_avg_price <= 0):
                            new_avg_price = None

                        # If position is being closed (size becomes 0, but wasn't 0 before), set closed_at
                        exit_price = None
                        entry_price = None
                        position_realized_pnl = new_realized
                        if resolved_size == 0 and position.size != 0:
                            # Get exit_price: use mark_price if available, otherwise current_price from position
                            exit_price = mark_price if mark_price is not None else position.current_price
                            # Get average_entry_price before it's cleared (use existing position value)
                            entry_price = position.average_entry_price
                            # Use unrealized_pnl from current position (before closing), not the new one (which will be 0)
                            unrealized_at_close = position.unrealized_pnl or Decimal("0")
                            
                            # For a fully closed position, realized_pnl should be the unrealized_pnl_at_close
                            # because all unrealized PnL becomes realized when position is closed.
                            # We don't use cumulative realized_pnl from Bybit (new_realized) as it's cumulative
                            # across all positions for the asset, not specific to this position.
                            position_realized_pnl = unrealized_at_close

                        # Track position opening: if size changes from 0 to non-zero, set opened_at
                        # This happens when position.size == 0 and resolved_size != 0
                        # Position opening is tracked via created_at (each record is created once)
                        
                        update_query = f"""
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
                            exit_price = CASE 
                                WHEN CAST($1 AS numeric) = 0 AND $9::text != '' THEN CAST($9::text AS numeric)
                                WHEN CAST($1 AS numeric) = 0 THEN exit_price
                                ELSE NULL
                            END,
                            total_fees = CASE 
                                WHEN CAST($1 AS numeric) = 0 THEN 0
                                ELSE total_fees
                            END,
                            version = version + 1,
                            last_updated = NOW(),
                            closed_at = CASE 
                                WHEN CAST($1 AS numeric) = 0 THEN NOW() 
                                ELSE NULL 
                            END
                        WHERE asset = $6
                          AND mode = $7
                          AND closed_at IS NULL
                          AND version = $8
                        RETURNING id, asset, mode, size, average_entry_price, current_price,
                                  unrealized_pnl, realized_pnl,
                                  long_size, short_size, version,
                                  last_updated, closed_at, created_at, exit_price
                    """
                    # Always pass string for $2, $5, and $9 to help asyncpg determine parameter type
                    # Pass empty string instead of None to avoid type inference issues
                    # This workaround is similar to the datetime issue fix ($4::timestamptz)
                    current_price_param = str(current_price) if current_price is not None else ''
                    avg_price_param = str(new_avg_price) if new_avg_price is not None else ''
                    exit_price_param = str(exit_price) if exit_price is not None else ''
                    row = await conn.fetchrow(
                        update_query,
                        str(resolved_size),
                        current_price_param,
                        str(new_unrealized),
                        str(position_realized_pnl),  # Use position_realized_pnl for closed positions
                        avg_price_param,
                        asset.upper(),
                        mode.lower(),
                        position.version,
                        exit_price_param,
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
                            size=str(updated.size),
                            resolved_size=str(resolved_size),
                            ws_size=str(size_from_ws) if size_from_ws is not None else None,
                            size_update_from_ws=size_update_from_ws,
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
        """Validate active position by computing from order history and comparing with stored state.
        
        CRITICAL: Only validates active positions (closed_at IS NULL).
        Validation MUST be performed BEFORE closing a position.
        After closing, positions are "frozen" and not validated.

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
            # Calculate average entry price from order history if position size is non-zero
            # This allows us to fix discrepancies even when stored average_entry_price is None
            computed_avg_price = None
            if computed_size != 0:
                computed_avg_price = await self._calculate_average_entry_price_from_orders(
                    asset, computed_size
                )
                if computed_avg_price:
                    logger.debug(
                        "computed_avg_price_from_orders",
                        asset=asset,
                        mode=mode,
                        computed_size=float(computed_size),
                        computed_avg_price=float(computed_avg_price),
                        trace_id=trace_id,
                    )

            # Get stored position
            stored_position = await self.get_active_position(asset, mode)

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
                if fix_discrepancies and computed_size != 0:
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        mode=mode,
                        action="creating_missing_position",
                        computed_size=float(computed_size),
                        computed_avg_price=float(computed_avg_price) if computed_avg_price else None,
                        trace_id=trace_id,
                    )
                    # Try to create position with computed average_entry_price
                    if computed_avg_price is None:
                        logger.warning(
                            "cannot_create_position_without_avg_price",
                            asset=asset,
                            mode=mode,
                            computed_size=float(computed_size),
                            reason="could not calculate average_entry_price from orders",
                            trace_id=trace_id,
                        )
                        return (
                            False,
                            f"Cannot create position: {error_msg}. Could not calculate average entry price from order history.",
                            None,
                        )
                    # Create position with computed values
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, computed_avg_price, mode, trace_id
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
                if fix_discrepancies:
                    # Determine which average_entry_price to use:
                    # 1. If computed_size == 0, use None (position is closed)
                    # 2. If stored average_entry_price exists, prefer it (it's maintained by update logic)
                    # 3. Otherwise, use computed_avg_price from order history
                    if computed_size == 0:
                        avg_price_to_use = None
                    elif stored_position.average_entry_price is not None:
                        # Prefer existing average_entry_price as it's maintained by position update logic
                        avg_price_to_use = stored_position.average_entry_price
                        logger.debug(
                            "using_existing_avg_price_for_fix",
                            asset=asset,
                            mode=mode,
                            existing_avg_price=float(avg_price_to_use),
                            trace_id=trace_id,
                        )
                    elif computed_avg_price is not None:
                        # Use computed average_entry_price from order history
                        avg_price_to_use = computed_avg_price
                        logger.info(
                            "using_computed_avg_price_for_fix",
                            asset=asset,
                            mode=mode,
                            computed_avg_price=float(avg_price_to_use),
                            trace_id=trace_id,
                        )
                    else:
                        # Cannot fix: no average_entry_price available
                        error_msg_with_avg = (
                            f"{error_msg}. Cannot fix: position has non-zero size but average_entry_price cannot be determined."
                        )
                        logger.warning(
                            "cannot_fix_position_discrepancy_missing_avg_price",
                            asset=asset,
                            mode=mode,
                            stored_size=float(stored_position.size),
                            computed_size=float(computed_size),
                            stored_avg_price=None,
                            computed_avg_price=None,
                            trace_id=trace_id,
                        )
                        return (False, error_msg_with_avg, None)
                    
                    logger.info(
                        "fixing_position_discrepancy",
                        asset=asset,
                        mode=mode,
                        action="updating_position",
                        stored_size=float(stored_position.size),
                        computed_size=float(computed_size),
                        size_diff=float(size_diff),
                        avg_price_to_use=float(avg_price_to_use) if avg_price_to_use else None,
                        trace_id=trace_id,
                    )
                    updated_position = await self._update_position_from_computed(
                        asset, computed_size, avg_price_to_use, mode, trace_id
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
                    "time_held_minutes": position.holding_time_minutes,
                    # position_size_norm depends on portfolio exposure; include key
                    # for schema completeness even if not populated here.
                    "position_size_norm": None,
                }
            )

            # Convert datetime objects to ISO strings for JSON serialization
            snapshot_payload_json = self._convert_datetime_to_iso_string(snapshot_payload)
            # Convert dict to JSON string for JSONB column (asyncpg requires this)
            snapshot_payload_json_str = json.dumps(snapshot_payload_json)

            pool = await DatabaseConnection.get_pool()
            # Extract size from snapshot_data for legacy column compatibility
            # The size column is NOT NULL in the table schema (migration 008)
            # even though we primarily use snapshot_data (migration 044)
            # Ensure size is never None - use 0 as fallback
            if position.size is None:
                logger.warning(
                    "position_size_is_none",
                    position_id=str(position.id),
                    asset=position.asset,
                    mode=position.mode,
                    trace_id=trace_id,
                )
                size_value = Decimal("0")
            else:
                size_value = position.size
            
            insert_query = """
                INSERT INTO position_snapshots (
                    id, position_id, asset, mode, size, snapshot_data, snapshot_timestamp
                )
                VALUES (gen_random_uuid(), $1, $2, $3, $4::decimal, $5::jsonb, NOW())
                RETURNING id, position_id, asset, mode, snapshot_data, snapshot_timestamp AS created_at
            """
            row = await pool.fetchrow(
                insert_query,
                str(position.id),
                position.asset,
                position.mode,
                size_value,  # Pass Decimal directly - asyncpg supports it
                snapshot_payload_json_str,
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
            existing = await self.get_active_position(asset, mode)

            # Validate: if size is non-zero, average_entry_price must be set
            if computed_size != 0 and computed_avg_price is None:
                # Try to preserve existing average_entry_price if available
                if existing and existing.average_entry_price is not None:
                    computed_avg_price = existing.average_entry_price
                    logger.warning(
                        "computed_avg_price is None for non-zero size, using existing value",
                        asset=asset,
                        mode=mode,
                        computed_size=float(computed_size),
                        existing_avg_price=float(existing.average_entry_price),
                        trace_id=trace_id,
                    )
                else:
                    # Cannot create/update position with non-zero size without average_entry_price
                    raise ValueError(
                        f"Cannot update position: size={computed_size} is non-zero but average_entry_price is None"
                    )

            # Upsert position with computed values, preserving version for optimistic locking
            if existing:
                # SQL logic: if avg_price_param is empty string and size != 0, preserve existing value
                # Otherwise, use new value or set NULL if size == 0
                # Track position opening: if size changes from 0 to non-zero, set opened_at
                # Check if position was closed (size == 0) and is now being opened
                existing_size = existing.size if existing else Decimal("0")
                # Position opening is tracked via created_at (each record is created once)
                
                upsert_query = f"""
                    UPDATE positions
                    SET size = $1,
                        average_entry_price = CASE 
                            WHEN $2::text = '' AND CAST($1 AS numeric) != 0 THEN average_entry_price
                            WHEN $2::text = '' THEN NULL
                            ELSE CAST($2::text AS numeric)
                        END,
                        closed_at = CASE WHEN CAST($1 AS numeric) = 0 THEN closed_at ELSE NULL END,
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
                # Validate: cannot create position with non-zero size without average_entry_price
                if computed_size != 0 and computed_avg_price is None:
                    raise ValueError(
                        f"Cannot create position: size={computed_size} is non-zero but average_entry_price is None"
                    )
                
                # Position opening is tracked via created_at (each record is created once)
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
        return position.holding_time_minutes

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

        params = {"category": settings.bybit_market_category, "symbol": asset.upper()}

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

        params = {"category": settings.bybit_market_category, "settleCoin": "USDT"}
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
        asset: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronize local active positions with Bybit API positions.
        
        CRITICAL: Only synchronizes active positions (closed_at IS NULL).
        By default, relies on WebSocket events (data is correct and timely).
        Synchronization through API is performed ONLY in case of:
        - Detected problems (discrepancies between WebSocket and computed positions)
        - Manual request through API endpoint
        - Periodic check (optional, with large interval)
        
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
            asset: Optional asset filter to sync only specific asset (e.g., "BTCUSDT")
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

            # Filter by asset if specified
            if asset:
                bybit_positions = [p for p in bybit_positions if p.get("symbol", "").upper() == asset.upper()]
                logger.debug(
                    "position_sync_bybit_asset_filtered",
                    asset=asset,
                    filtered_count=len(bybit_positions),
                    trace_id=trace_id,
                )

            # Get local active positions (only sync active positions)
            local_positions = await self.get_all_active_positions()
            
            # Filter local positions by asset if specified
            if asset:
                local_positions = [p for p in local_positions if p.asset.upper() == asset.upper()]

            # Build comparison report
            bybit_dict = {p["symbol"]: p for p in bybit_positions}
            local_dict = {p.asset: p for p in local_positions if p.size != 0}
            
            # If asset filter is specified and no positions found, return early
            if asset and not bybit_dict and not local_dict:
                logger.info(
                    "position_sync_bybit_no_positions_for_asset",
                    asset=asset,
                    trace_id=trace_id,
                )
                return {
                    "bybit_positions_count": 0,
                    "local_positions_count": 0,
                    "comparisons": [],
                    "updated": [],
                    "created": [],
                    "errors": [],
                }

            comparisons = []
            updated = []
            created = []
            errors = []

            # Compare positions
            all_assets = set(bybit_dict.keys()) | set(local_dict.keys())
            
            # If asset filter is specified, only process that asset
            if asset:
                all_assets = {asset.upper()} & all_assets

            for asset_symbol in all_assets:
                bybit_pos = bybit_dict.get(asset_symbol)
                local_pos = local_dict.get(asset_symbol)

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
                    "asset": asset_symbol,
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
                            # Validate: if size is non-zero, average_entry_price must be set
                            # Use existing local average_entry_price if Bybit doesn't provide it
                            avg_price_to_use = bybit_avg_price
                            if bybit_size != 0 and avg_price_to_use is None:
                                if local_pos and local_pos.average_entry_price is not None:
                                    avg_price_to_use = local_pos.average_entry_price
                                    logger.warning(
                                        "bybit_avg_price_missing_using_local",
                                        asset=asset_symbol,
                                        bybit_size=str(bybit_size),
                                        local_avg_price=str(avg_price_to_use),
                                        trace_id=trace_id,
                                    )
                                else:
                                    errors.append(
                                        f"Cannot sync {asset_symbol}: size={bybit_size} is non-zero but average_entry_price is missing from Bybit and local position"
                                    )
                                    logger.error(
                                        "bybit_sync_skipped_missing_avg_price",
                                        asset=asset_symbol,
                                        bybit_size=str(bybit_size),
                                        trace_id=trace_id,
                                    )
                                    continue
                            
                            # Update or create position
                            if local_pos:
                                # Update existing - only update size/price if they differ
                                if size_diff > Decimal("0.0001") or (avg_price_diff and avg_price_diff_pct and avg_price_diff_pct > 1.0):
                                    updated_pos = await self._update_position_from_computed(
                                        asset=asset_symbol,
                                        computed_size=bybit_size,
                                        computed_avg_price=avg_price_to_use,
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
                                    asset_symbol.upper(),
                                    "one-way",
                                )
                                reason = "size/price" if (size_diff > Decimal("0.0001") or (avg_price_diff and avg_price_diff_pct and avg_price_diff_pct > 1.0)) else "pnl"
                                updated.append({"asset": asset_symbol, "action": "updated", "position_id": str(updated_pos.id), "reason": reason})
                            else:
                                # Create new - avg_price_to_use already validated above
                                created_pos = await self._update_position_from_computed(
                                    asset=asset_symbol,
                                    computed_size=bybit_size,
                                    computed_avg_price=avg_price_to_use,
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
                                    asset_symbol.upper(),
                                    "one-way",
                                )
                                created.append({"asset": asset_symbol, "action": "created", "position_id": str(created_pos.id)})

                            logger.info(
                                "position_synced_with_bybit",
                                asset=asset_symbol,
                                bybit_size=str(bybit_size),
                                local_size=str(local_size),
                                force=force,
                                trace_id=trace_id,
                            )
                        # Case 2: Position exists locally but not on Bybit (should be closed)
                        elif local_pos and not bybit_pos:
                            # Close position by setting size to 0
                            # Use asset_symbol instead of asset parameter
                            pool = await DatabaseConnection.get_pool()
                            
                            # Get position_id before closing
                            position_row = await pool.fetchrow(
                                "SELECT id, realized_pnl, current_price FROM positions WHERE asset = $1 AND mode = $2",
                                asset_symbol.upper(),
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
                                    exit_price = CASE 
                                        WHEN $3::text = '' THEN NULL
                                        ELSE CAST($3::text AS numeric)
                                    END,
                                    unrealized_pnl = 0,
                                    version = version + 1,
                                    last_updated = NOW(),
                                    closed_at = NOW()
                                WHERE asset = $1 AND mode = $2
                                """,
                                asset_symbol.upper(),
                                "one-way",
                                str(exit_price) if exit_price is not None else '',
                            )
                            
                            # Update prediction_trading_results for this position (same actions as normal close)
                            if position_id:
                                try:
                                    # Find open prediction_trading_results for this position
                                    # Join through signal_id -> orders -> position_orders -> positions
                                    # Note: position_id column was removed from prediction_trading_results (migration 033)
                                    # so we join through position_orders table instead
                                    open_results = await pool.fetch(
                                        """
                                        SELECT DISTINCT ptr.id, ptr.signal_id, ptr.realized_pnl, ptr.unrealized_pnl
                                        FROM prediction_trading_results ptr
                                        INNER JOIN orders o ON o.signal_id = ptr.signal_id
                                        INNER JOIN position_orders po ON po.order_id = o.id
                                        WHERE po.position_id = $1 AND ptr.is_closed = false
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
                                            asset=asset_symbol,
                                            final_realized_pnl=str(final_realized),
                                            trace_id=trace_id,
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "failed_to_update_prediction_trading_results_on_sync_close",
                                        position_id=str(position_id),
                                        asset=asset_symbol,
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
                                    asset_symbol.upper(),
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
                                        asset=asset_symbol,
                                        position_id=str(updated_position.id),
                                        trace_id=trace_id,
                                    )
                            except Exception as e:
                                logger.warning(
                                    "failed_to_publish_position_event_on_sync_close",
                                    asset=asset_symbol,
                                    error=str(e),
                                    trace_id=trace_id,
                                    exc_info=True,
                                )
                            
                            updated.append({"asset": asset_symbol, "action": "closed", "reason": "position_not_on_bybit"})
                            logger.info(
                                "position_closed_not_on_bybit",
                                asset=asset_symbol,
                                local_size=str(local_size),
                                position_id=str(position_id) if position_id else None,
                                force=force,
                                trace_id=trace_id,
                            )
                    except Exception as e:
                        error_msg = f"Failed to sync {asset_symbol}: {e}"
                        errors.append({"asset": asset_symbol, "error": error_msg})
                        logger.error(
                            "position_sync_error",
                            asset=asset_symbol,
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



