"""Instrument info client for model service.

Provides access to instrument_info table data (min_order_value, etc.)
for use in signal generation logic.
"""

from typing import Optional
from decimal import Decimal
import asyncio

from ..database.connection import db_pool
from ..config.logging import get_logger

logger = get_logger(__name__)


class InstrumentInfoClient:
    """Client for accessing instrument_info table data."""

    def __init__(self):
        """Initialize instrument info client with cache."""
        # Cache: symbol -> min_order_value
        self._cache: dict[str, Optional[Decimal]] = {}
        self._cache_lock = asyncio.Lock()

    async def get_min_order_value(self, symbol: str) -> Optional[Decimal]:
        """Get minimum order value for a symbol from instrument_info table.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT', 'BTCUSDT')

        Returns:
            Minimum order value in USDT, or None if not found in database
        """
        # Check cache first
        async with self._cache_lock:
            if symbol in self._cache:
                cached_value = self._cache[symbol]
                logger.debug(
                    "min_order_value_cache_hit",
                    symbol=symbol,
                    min_order_value=float(cached_value) if cached_value else None,
                )
                return cached_value

        # Query database
        try:
            pool = await db_pool.get_pool()
            query = """
                SELECT min_order_value
                FROM instrument_info
                WHERE symbol = $1
            """
            row = await pool.fetchrow(query, symbol)

            min_order_value: Optional[Decimal] = None
            if row and row["min_order_value"] is not None:
                min_order_value = Decimal(str(row["min_order_value"]))

            # Update cache
            async with self._cache_lock:
                self._cache[symbol] = min_order_value

            if min_order_value is not None:
                logger.debug(
                    "min_order_value_fetched_from_db",
                    symbol=symbol,
                    min_order_value=float(min_order_value),
                )
            else:
                logger.debug(
                    "min_order_value_not_found_in_db",
                    symbol=symbol,
                )

            return min_order_value

        except Exception as e:
            logger.warning(
                "min_order_value_query_failed",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )
            # Cache None to avoid repeated failed queries
            async with self._cache_lock:
                self._cache[symbol] = None
            return None

    async def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for a specific symbol or all symbols.

        Args:
            symbol: Symbol to clear cache for, or None to clear all
        """
        async with self._cache_lock:
            if symbol:
                self._cache.pop(symbol, None)
                logger.debug("min_order_value_cache_cleared", symbol=symbol)
            else:
                self._cache.clear()
                logger.debug("min_order_value_cache_cleared_all")


# Global instance
instrument_info_client = InstrumentInfoClient()

