"""Bybit fee rate manager service for caching and retrieving trading fees.

This service is responsible for:
  - Fetching maker/taker fee rates from Bybit REST API (/v5/account/fee-rate)
  - Caching them in the shared bybit_fee_rates table
  - Providing a TTL-based cached view for order/risk logic
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional, List

from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..config.settings import settings
from ..exceptions import DatabaseError, OrderExecutionError
from ..utils.bybit_client import get_bybit_client
from ..utils.tracing import generate_trace_id, set_trace_id


logger = get_logger(__name__)


@dataclass
class FeeRate:
    """Normalized view of maker/taker fee rates used by Order Manager."""

    symbol: str
    market_type: str
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    last_synced_at: datetime


class FeeRateManager:
    """Service for managing Bybit fee rates persistence and refresh."""

    async def get_fee_rate(
        self,
        symbol: str,
        market_type: str = "linear",
        trace_id: Optional[str] = None,
        allow_api_fallback: bool = True,
    ) -> Optional[FeeRate]:
        """Get fee rate for a symbol, using DB cache with TTL and optional API fallback.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            market_type: Bybit market category (e.g., 'linear', 'inverse', 'spot')
            trace_id: Optional trace ID for logging
            allow_api_fallback: Whether to query Bybit API when cached data is missing/stale
        """
        if trace_id is None:
            trace_id = generate_trace_id()
            set_trace_id(trace_id)

        symbol = symbol.upper()
        market_type = market_type.lower()

        # 1) Try DB cache
        cached = await self._get_fee_rate_from_db(symbol, market_type)
        ttl_seconds = int(settings.order_manager_fee_data_ttl_seconds)
        now = datetime.now(timezone.utc)
        if cached is not None:
            age = (now - cached.last_synced_at).total_seconds()
            if age <= ttl_seconds:
                logger.debug(
                    "fee_rate_cache_hit",
                    symbol=symbol,
                    market_type=market_type,
                    age_seconds=age,
                    ttl_seconds=ttl_seconds,
                    trace_id=trace_id,
                )
                return cached

            logger.info(
                "fee_rate_cache_stale",
                symbol=symbol,
                market_type=market_type,
                age_seconds=age,
                ttl_seconds=ttl_seconds,
                trace_id=trace_id,
            )

        if not allow_api_fallback:
            return cached

        # 2) Fallback to Bybit API
        try:
            api_rate = await self._fetch_fee_rate_from_api(symbol, market_type, trace_id=trace_id)
        except OrderExecutionError:
            # Already logged
            api_rate = None
        except Exception as e:  # pragma: no cover - extremely defensive
            logger.error(
                "fee_rate_api_unexpected_error",
                symbol=symbol,
                market_type=market_type,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            api_rate = None

        if api_rate is None:
            return cached

        # 3) Upsert and return
        try:
            await self._upsert_fee_rate(api_rate, trace_id=trace_id)
        except DatabaseError:
            # Already logged; still return api_rate for current call
            pass
        except Exception as e:  # pragma: no cover
            logger.error(
                "fee_rate_upsert_unexpected_error",
                symbol=symbol,
                market_type=market_type,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )

        return api_rate

    async def _get_fee_rate_from_db(self, symbol: str, market_type: str) -> Optional[FeeRate]:
        """Fetch fee rate from bybit_fee_rates table."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                SELECT
                    symbol,
                    market_type,
                    maker_fee_rate,
                    taker_fee_rate,
                    last_synced_at
                FROM bybit_fee_rates
                WHERE symbol = $1 AND market_type = $2
            """
            row = await pool.fetchrow(query, symbol, market_type)
            if row is None:
                return None

            return FeeRate(
                symbol=row["symbol"],
                market_type=row["market_type"],
                maker_fee_rate=Decimal(str(row["maker_fee_rate"])),
                taker_fee_rate=Decimal(str(row["taker_fee_rate"])),
                last_synced_at=row["last_synced_at"].replace(tzinfo=timezone.utc)
                if row["last_synced_at"].tzinfo is None
                else row["last_synced_at"],
            )
        except Exception as e:
            logger.error(
                "fee_rate_db_query_failed",
                symbol=symbol,
                market_type=market_type,
                error=str(e),
            )
            raise DatabaseError(f"Failed to query fee rates for {symbol}: {e}") from e

    async def _upsert_fee_rate(self, fee_rate: FeeRate, trace_id: Optional[str] = None) -> None:
        """Upsert a single fee rate into bybit_fee_rates table."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = """
                INSERT INTO bybit_fee_rates (
                    symbol,
                    market_type,
                    maker_fee_rate,
                    taker_fee_rate,
                    last_synced_at,
                    created_at,
                    updated_at
                )
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                ON CONFLICT (symbol, market_type)
                DO UPDATE SET
                    maker_fee_rate = EXCLUDED.maker_fee_rate,
                    taker_fee_rate = EXCLUDED.taker_fee_rate,
                    last_synced_at = EXCLUDED.last_synced_at,
                    updated_at = NOW()
            """
            await pool.execute(
                query,
                fee_rate.symbol,
                fee_rate.market_type,
                str(fee_rate.maker_fee_rate),
                str(fee_rate.taker_fee_rate),
                fee_rate.last_synced_at,
            )
            logger.info(
                "fee_rate_upserted",
                symbol=fee_rate.symbol,
                market_type=fee_rate.market_type,
                maker_fee_rate=str(fee_rate.maker_fee_rate),
                taker_fee_rate=str(fee_rate.taker_fee_rate),
                trace_id=trace_id,
            )
        except Exception as e:
            logger.error(
                "fee_rate_upsert_failed",
                symbol=fee_rate.symbol,
                market_type=fee_rate.market_type,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to upsert fee rate for {fee_rate.symbol}: {e}") from e

    async def _fetch_fee_rate_from_api(
        self,
        symbol: str,
        market_type: str,
        trace_id: Optional[str] = None,
    ) -> Optional[FeeRate]:
        """Query Bybit REST API for fee rate for a specific symbol/category.

        Uses /v5/account/fee-rate endpoint which returns maker/taker fee rates
        for the requested symbol and category.
        """
        bybit_client = get_bybit_client()
        endpoint = "/v5/account/fee-rate"

        params: Dict[str, Any] = {
            "category": market_type,
            "symbol": symbol,
        }

        logger.info(
            "fee_rate_api_request",
            symbol=symbol,
            market_type=market_type,
            trace_id=trace_id,
        )

        response = await bybit_client.get(endpoint, params=params, authenticated=True)

        ret_code = response.get("retCode", 0)
        if ret_code != 0:
            error_msg = response.get("retMsg", "Unknown error")
            logger.error(
                "fee_rate_api_error",
                symbol=symbol,
                market_type=market_type,
                ret_code=ret_code,
                ret_msg=error_msg,
                trace_id=trace_id,
            )
            raise OrderExecutionError(f"Bybit fee-rate error: {error_msg} (code: {ret_code})")

        result = response.get("result", {}) or {}
        list_: List[Dict[str, Any]] = result.get("list", []) or []
        if not list_:
            logger.warning(
                "fee_rate_api_empty_list",
                symbol=symbol,
                market_type=market_type,
                trace_id=trace_id,
            )
            return None

        entry = list_[0]
        try:
            maker = Decimal(str(entry.get("makerFeeRate")))
            taker = Decimal(str(entry.get("takerFeeRate")))
        except Exception as e:
            logger.error(
                "fee_rate_api_parse_error",
                symbol=symbol,
                market_type=market_type,
                raw_entry=entry,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return None

        now = datetime.now(timezone.utc)
        logger.info(
            "fee_rate_api_success",
            symbol=symbol,
            market_type=market_type,
            maker_fee_rate=str(maker),
            taker_fee_rate=str(taker),
            trace_id=trace_id,
        )

        return FeeRate(
            symbol=symbol,
            market_type=market_type,
            maker_fee_rate=maker,
            taker_fee_rate=taker,
            last_synced_at=now,
        )


class FeeRateRefreshTask:
    """Background task that periodically refreshes fee rates for known instruments."""

    def __init__(self) -> None:
        self._manager = FeeRateManager()
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
            "fee_rate_refresh_task_started",
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

        logger.info("fee_rate_refresh_task_stopped")

    async def _run(self) -> None:
        """Internal loop that refreshes fee rates for all known instruments."""
        import asyncio

        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        # Reuse instrument_info refresh interval to avoid too many knobs
        interval = settings.order_manager_instrument_info_refresh_interval

        while self._should_run:
            try:
                await asyncio.sleep(interval)
                if not self._should_run:
                    break

                refreshed = await self._refresh_all_fee_rates(trace_id=trace_id)
                logger.info(
                    "fee_rate_periodic_refresh_completed",
                    refreshed=refreshed,
                    trace_id=trace_id,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "fee_rate_periodic_refresh_failed",
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )
                await asyncio.sleep(60)

    async def _refresh_all_fee_rates(self, trace_id: Optional[str] = None) -> int:
        """Refresh fee rates for all symbols present in instrument_info table."""
        try:
            pool = await DatabaseConnection.get_pool()
            query = "SELECT symbol FROM instrument_info"
            rows = await pool.fetch(query)
            symbols = {row["symbol"] for row in rows if row.get("symbol")}
        except Exception as e:
            logger.error(
                "fee_rate_refresh_symbols_query_failed",
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            return 0

        if not symbols:
            return 0

        from ..config.settings import settings
        category = settings.bybit_market_category
        refreshed = 0

        for symbol in symbols:
            try:
                rate = await self._manager.get_fee_rate(
                    symbol=symbol,
                    market_type=category,
                    trace_id=trace_id,
                    allow_api_fallback=True,
                )
                if rate is not None:
                    refreshed += 1
            except Exception as e:
                logger.warning(
                    "fee_rate_refresh_symbol_failed",
                    symbol=symbol,
                    error=str(e),
                    trace_id=trace_id,
                )
                # Avoid hammering API on continuous failures
                await asyncio.sleep(0.1)

        return refreshed



