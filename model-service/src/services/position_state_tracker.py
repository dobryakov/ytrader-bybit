"""
Position state tracker service.

Tracks position lifecycle data including entry price, entry time, peak price,
highest unrealized PnL, and last exit signal timestamp for exit strategy evaluation.
Persists state to database for recovery across service restarts.
"""

from typing import Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..models.position_state_tracker import PositionState
from ..database.connection import db_pool
from ..config.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)


class PositionStateTracker:
    """
    Tracks position state for exit strategy evaluation.

    Maintains position lifecycle data including entry price, entry time,
    peak price, highest unrealized PnL, and last exit signal timestamp.
    Persists state to database for recovery across service restarts.
    """

    def __init__(self):
        """Initialize position state tracker."""
        # In-memory cache for fast access (keyed by asset)
        self._state_cache: Dict[str, PositionState] = {}

    async def get_or_create_state(
        self,
        asset: str,
        position_data: Dict[str, Any],
    ) -> PositionState:
        """
        Get existing position state or create new one.

        Args:
            asset: Trading pair symbol
            position_data: Position data from update event

        Returns:
            PositionState instance
        """
        # Check cache first
        if asset in self._state_cache:
            state = self._state_cache[asset]
            # Update peak price and highest PnL from current position data
            await self._update_state_from_position_data(state, position_data)
            return state

        # Try to load from database
        state = await self._load_from_database(asset)
        if state:
            self._state_cache[asset] = state
            await self._update_state_from_position_data(state, position_data)
            return state

        # Create new state from position data
        state = await self._create_from_position_data(asset, position_data)
        self._state_cache[asset] = state
        await self._save_to_database(state)
        return state

    async def update_state(
        self,
        asset: str,
        position_data: Dict[str, Any],
    ) -> PositionState:
        """
        Update position state from position data.

        Args:
            asset: Trading pair symbol
            position_data: Position data from update event

        Returns:
            Updated PositionState instance
        """
        state = await self.get_or_create_state(asset, position_data)
        await self._save_to_database(state)
        return state

    async def mark_exit_signal_sent(self, asset: str) -> None:
        """
        Mark that an exit signal was sent for this position.

        Args:
            asset: Trading pair symbol
        """
        if asset in self._state_cache:
            self._state_cache[asset].last_exit_signal_time = datetime.utcnow()
            await self._save_to_database(self._state_cache[asset])

    async def remove_state(self, asset: str) -> None:
        """
        Remove position state (when position is closed).

        Args:
            asset: Trading pair symbol
        """
        if asset in self._state_cache:
            del self._state_cache[asset]

        # Remove from database
        await self._remove_from_database(asset)

    async def _create_from_position_data(
        self,
        asset: str,
        position_data: Dict[str, Any],
    ) -> PositionState:
        """
        Create new PositionState from position data.

        Args:
            asset: Trading pair symbol
            position_data: Position data from update event

        Returns:
            New PositionState instance
        """
        # Try to extract entry price from position data
        # If not available, use current price as approximation
        avg_price = position_data.get("avg_price") or position_data.get("average_price")
        if avg_price:
            entry_price = float(avg_price)
        else:
            # Fallback: estimate from unrealized PnL and current price
            # This is approximate - ideally we'd track entry price from order execution
            unrealized_pnl_pct = position_data.get("unrealized_pnl_pct", 0.0)
            # We can't determine entry price without current price, so use a default
            # In practice, this should come from order execution events
            entry_price = 1.0  # Placeholder - should be improved with actual entry tracking
            logger.warning(
                "Cannot determine entry price from position data, using placeholder",
                asset=asset,
                position_data_keys=list(position_data.keys()),
            )

        # Use current time as entry time (approximation)
        # In practice, this should come from order execution events
        entry_time = datetime.utcnow()

        state = PositionState(
            asset=asset,
            entry_price=entry_price,
            entry_time=entry_time,
            peak_price=entry_price,  # Initialize with entry price
            highest_unrealized_pnl=0.0,  # Initialize with 0
            last_exit_signal_time=None,
        )

        # Update with current position data
        await self._update_state_from_position_data(state, position_data)

        return state

    async def _update_state_from_position_data(
        self,
        state: PositionState,
        position_data: Dict[str, Any],
    ) -> None:
        """
        Update position state from current position data.

        Args:
            state: PositionState to update
            position_data: Position data from update event
        """
        # Update peak price if we can determine current price
        unrealized_pnl_pct = position_data.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is not None:
            unrealized_pnl_pct = float(unrealized_pnl_pct)
            # Calculate current price from entry price and PnL
            current_price = state.entry_price * (1 + unrealized_pnl_pct / 100.0)
            state.update_peak_price(current_price)
            state.update_highest_pnl(unrealized_pnl_pct)

    async def _load_from_database(self, asset: str) -> Optional[PositionState]:
        """
        Load position state from database.

        Args:
            asset: Trading pair symbol

        Returns:
            PositionState if found, None otherwise
        """
        try:
            pool = await db_pool.get_pool()
            query = """
                SELECT asset, entry_price, entry_time, peak_price, highest_unrealized_pnl, last_exit_signal_time
                FROM position_states
                WHERE asset = $1
                LIMIT 1
            """
            row = await pool.fetchrow(query, asset)

            if row:
                return PositionState(
                    asset=row["asset"],
                    entry_price=float(row["entry_price"]),
                    entry_time=row["entry_time"],
                    peak_price=float(row["peak_price"]),
                    highest_unrealized_pnl=float(row["highest_unrealized_pnl"]),
                    last_exit_signal_time=row["last_exit_signal_time"],
                )
            return None
        except Exception as e:
            logger.error("Error loading position state from database", asset=asset, error=str(e), exc_info=True)
            return None

    async def _save_to_database(self, state: PositionState) -> None:
        """
        Save position state to database.

        Args:
            state: PositionState to save
        """
        try:
            pool = await db_pool.get_pool()
            query = """
                INSERT INTO position_states
                (asset, entry_price, entry_time, peak_price, highest_unrealized_pnl, last_exit_signal_time, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (asset) DO UPDATE SET
                    entry_price = EXCLUDED.entry_price,
                    entry_time = EXCLUDED.entry_time,
                    peak_price = EXCLUDED.peak_price,
                    highest_unrealized_pnl = EXCLUDED.highest_unrealized_pnl,
                    last_exit_signal_time = EXCLUDED.last_exit_signal_time,
                    updated_at = NOW()
            """
            await pool.execute(
                query,
                state.asset,
                state.entry_price,
                state.entry_time,
                state.peak_price,
                state.highest_unrealized_pnl,
                state.last_exit_signal_time,
            )
        except Exception as e:
            logger.error(
                "Error saving position state to database",
                asset=state.asset,
                error=str(e),
                exc_info=True,
            )
            # Don't raise - continue operating even if persistence fails

    async def _remove_from_database(self, asset: str) -> None:
        """
        Remove position state from database.

        Args:
            asset: Trading pair symbol
        """
        try:
            pool = await db_pool.get_pool()
            query = "DELETE FROM position_states WHERE asset = $1"
            await pool.execute(query, asset)
        except Exception as e:
            logger.error("Error removing position state from database", asset=asset, error=str(e), exc_info=True)


# Global position state tracker instance
position_state_tracker = PositionStateTracker()

