"""Instrument info manager service for persisting Bybit instruments-info data.

This service is responsible for:
  - Fetching instruments-info from Bybit REST API (/v5/market/instruments-info)
  - Normalizing and upserting data into the shared instrument_info table
  - Providing a cached source of truth for order/risk validation logic

Implements T072 from specs/004-order-manager/tasks.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..utils.bybit_client import get_bybit_client
from ..utils.tracing import generate_trace_id, set_trace_id
from ..exceptions import DatabaseError, OrderExecutionError


logger = get_logger(__name__)


@dataclass
class InstrumentInfo:
    """Normalized view of instrument limits used by Order Manager."""

    symbol: str
    base_coin: Optional[str]
    quote_coin: Optional[str]
    status: Optional[str]
    lot_size: Decimal
    min_order_qty: Decimal
    max_order_qty: Decimal
    min_order_value: Decimal
    price_tick_size: Decimal
    price_limit_ratio_x: Optional[Decimal]
    price_limit_ratio_y: Optional[Decimal]


class InstrumentInfoManager:
    """Service for managing instruments-info persistence and refresh."""

    async def refresh_all_instruments(
        self,
        category: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> int:
        """Fetch instruments-info for the given category and upsert into database.

        Uses Bybit REST API `/v5/market/instruments-info` with pagination support.

        Args:
            category: Bybit category (e.g. "linear", "inverse", "spot").
            trace_id: Optional trace ID for logging; if None, a new one is created.

        Returns:
            Number of instruments upserted.

        Raises:
            OrderExecutionError: If Bybit API returns an error.
            DatabaseError: If database upsert fails.
        """
        if trace_id is None:
            trace_id = generate_trace_id()
            set_trace_id(trace_id)

        # Use settings default if category not provided
        if category is None:
            category = settings.bybit_market_category

        logger.info(
            "instrument_info_refresh_started",
            category=category,
            trace_id=trace_id,
        )

        bybit_client = get_bybit_client()
        total_upserted = 0
        cursor: Optional[str] = None

        try:
            while True:
                params: Dict[str, Any] = {"category": category}
                if cursor:
                    params["cursor"] = cursor

                response = await bybit_client.get(
                    "/v5/market/instruments-info",
                    params=params,
                    authenticated=False,
                )

                ret_code = response.get("retCode", 0)
                if ret_code != 0:
                    error_msg = response.get("retMsg", "Unknown error")
                    logger.error(
                        "instrument_info_api_error",
                        ret_code=ret_code,
                        ret_msg=error_msg,
                        category=category,
                        trace_id=trace_id,
                    )
                    raise OrderExecutionError(f"Bybit instruments-info error: {error_msg} (code: {ret_code})")

                result = response.get("result", {}) or {}
                symbols: List[Dict[str, Any]] = result.get("list", []) or []
                cursor = result.get("nextPageCursor") or None

                if symbols:
                    upserted = await self._upsert_instruments(symbols, trace_id)
                    total_upserted += upserted

                # Break when there is no next page cursor
                if not cursor:
                    break

            logger.info(
                "instrument_info_refresh_completed",
                category=category,
                upserted=total_upserted,
                trace_id=trace_id,
            )
            return total_upserted
        except OrderExecutionError:
            raise
        except Exception as e:
            logger.error(
                "instrument_info_refresh_failed",
                error=str(e),
                category=category,
                trace_id=trace_id,
                exc_info=True,
            )
            raise OrderExecutionError(f"Failed to refresh instruments-info: {e}") from e

    async def _upsert_instruments(
        self,
        symbols: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> int:
        """Upsert a batch of instruments into the instrument_info table."""
        try:
            pool = await DatabaseConnection.get_pool()
            upsert_query = """
                INSERT INTO instrument_info (
                    symbol,
                    base_coin,
                    quote_coin,
                    status,
                    lot_size,
                    min_order_qty,
                    max_order_qty,
                    min_order_value,
                    price_tick_size,
                    price_limit_ratio_x,
                    price_limit_ratio_y,
                    raw_response,
                    updated_at
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12, NOW()
                )
                ON CONFLICT (symbol)
                DO UPDATE SET
                    base_coin = EXCLUDED.base_coin,
                    quote_coin = EXCLUDED.quote_coin,
                    status = EXCLUDED.status,
                    lot_size = EXCLUDED.lot_size,
                    min_order_qty = EXCLUDED.min_order_qty,
                    max_order_qty = EXCLUDED.max_order_qty,
                    min_order_value = EXCLUDED.min_order_value,
                    price_tick_size = EXCLUDED.price_tick_size,
                    price_limit_ratio_x = EXCLUDED.price_limit_ratio_x,
                    price_limit_ratio_y = EXCLUDED.price_limit_ratio_y,
                    raw_response = EXCLUDED.raw_response,
                    updated_at = NOW()
            """

            count = 0
            async with pool.acquire() as connection:
                async with connection.transaction():
                    for symbol_data in symbols:
                        lot_size_filter = symbol_data.get("lotSizeFilter", {}) or {}
                        price_filter = symbol_data.get("priceFilter", {}) or {}

                        symbol = symbol_data.get("symbol")
                        if not symbol:
                            continue

                        values = (
                            symbol,
                            symbol_data.get("baseCoin"),
                            symbol_data.get("quoteCoin"),
                            symbol_data.get("status"),
                            Decimal(str(lot_size_filter.get("qtyStep", "0.001"))),
                            Decimal(str(lot_size_filter.get("minQty", "0.001"))),
                            Decimal(str(lot_size_filter.get("maxQty", "999999999"))),
                            Decimal(str(symbol_data.get("minOrderValue", "5"))),
                            Decimal(str(price_filter.get("tickSize", "0.01"))),
                            Decimal(str(price_filter.get("priceLimitRatioX", "0.1")))
                            if price_filter.get("priceLimitRatioX") is not None
                            else None,
                            Decimal(str(price_filter.get("priceLimitRatioY", "0.1")))
                            if price_filter.get("priceLimitRatioY") is not None
                            else None,
                            symbol_data,
                        )

                        await connection.execute(upsert_query, *values)
                        count += 1

            logger.debug(
                "instrument_info_batch_upserted",
                count=count,
                trace_id=trace_id,
            )
            return count
        except Exception as e:
            logger.error(
                "instrument_info_upsert_failed",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to upsert instruments-info: {e}") from e

    async def get_instrument(self, symbol: str) -> Optional[InstrumentInfo]:
        """Get normalized instrument info for a symbol from the database."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT
                    symbol,
                    base_coin,
                    quote_coin,
                    status,
                    lot_size,
                    min_order_qty,
                    max_order_qty,
                    min_order_value,
                    price_tick_size,
                    price_limit_ratio_x,
                    price_limit_ratio_y
                FROM instrument_info
                WHERE symbol = $1
            """
            row = await pool.fetchrow(query, symbol)
            if row is None:
                return None

            return InstrumentInfo(
                symbol=row["symbol"],
                base_coin=row["base_coin"],
                quote_coin=row["quote_coin"],
                status=row["status"],
                lot_size=Decimal(str(row["lot_size"])),
                min_order_qty=Decimal(str(row["min_order_qty"])),
                max_order_qty=Decimal(str(row["max_order_qty"])),
                min_order_value=Decimal(str(row["min_order_value"])),
                price_tick_size=Decimal(str(row["price_tick_size"])),
                price_limit_ratio_x=Decimal(str(row["price_limit_ratio_x"]))
                if row["price_limit_ratio_x"] is not None
                else None,
                price_limit_ratio_y=Decimal(str(row["price_limit_ratio_y"]))
                if row["price_limit_ratio_y"] is not None
                else None,
            )
        except Exception as e:
            logger.error(
                "instrument_info_query_failed",
                symbol=symbol,
                error=str(e),
            )
            raise DatabaseError(f"Failed to query instruments-info for {symbol}: {e}") from e


class InstrumentInfoRefreshTask:
    """Background task that periodically refreshes instruments-info from Bybit."""

    def __init__(self) -> None:
        self._manager = InstrumentInfoManager()
        self._should_run = False
        self._task: Optional["asyncio.Task[None]"] = None

    async def start(self) -> None:
        """Start periodic refresh loop."""
        import asyncio

        if self._should_run:
            return

        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

        logger.info(
            "instrument_info_refresh_task_started",
            interval=settings.order_manager_instrument_info_refresh_interval,
        )

    async def stop(self) -> None:
        """Stop periodic refresh loop."""
        if not self._should_run:
            return

        import asyncio

        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("instrument_info_refresh_task_stopped")

    async def _run(self) -> None:
        """Internal loop that triggers refresh at configured interval."""
        import asyncio

        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.order_manager_instrument_info_refresh_interval)
                if not self._should_run:
                    break

                refreshed = await self._manager.refresh_all_instruments(trace_id=trace_id)
                logger.info(
                    "instrument_info_periodic_refresh_completed",
                    refreshed=refreshed,
                    trace_id=trace_id,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "instrument_info_periodic_refresh_failed",
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Wait a bit before retrying to avoid tight error loop
                await asyncio.sleep(60)


