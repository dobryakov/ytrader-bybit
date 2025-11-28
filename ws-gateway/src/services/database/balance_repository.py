"""Database operations for AccountBalance entity."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from .connection import DatabaseConnection
from ...config.logging import get_logger
from ...models.account_balance import AccountBalance

logger = get_logger(__name__)


class BalanceRepository:
    """Repository for CRUD operations on account_balances table."""

    @staticmethod
    async def create_balance(balance: AccountBalance) -> AccountBalance:
        """Insert a new balance record into the database.

        Args:
            balance: AccountBalance instance to persist

        Returns:
            The persisted AccountBalance instance

        Raises:
            DatabaseError: If database operation fails
        """
        query = """
            INSERT INTO account_balances (
                id, coin, wallet_balance, available_balance, frozen,
                event_timestamp, received_at, trace_id,
                equity, usd_value, margin_collateral, total_order_im, total_position_im
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        await DatabaseConnection.execute(
            query,
            balance.id,
            balance.coin,
            balance.wallet_balance,
            balance.available_balance,
            balance.frozen,
            balance.event_timestamp,
            balance.received_at,
            balance.trace_id,
            balance.equity,
            balance.usd_value,
            balance.margin_collateral,
            balance.total_order_im,
            balance.total_position_im,
        )
        logger.debug(
            "balance_created",
            balance_id=str(balance.id),
            coin=balance.coin,
            wallet_balance=str(balance.wallet_balance),
            trace_id=balance.trace_id,
        )
        return balance

    @staticmethod
    async def get_latest_balance(coin: str) -> Optional[AccountBalance]:
        """Get the latest balance record for a specific coin.

        Args:
            coin: Coin symbol (e.g., 'USDT', 'BTC')

        Returns:
            Latest AccountBalance for the coin, or None if not found
        """
        query = """
            SELECT id, coin, wallet_balance, available_balance, frozen,
                   event_timestamp, received_at, trace_id,
                   equity, usd_value, margin_collateral, total_order_im, total_position_im
            FROM account_balances
            WHERE coin = $1
            ORDER BY received_at DESC
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, coin)
        if not row:
            return None

        # Convert row to AccountBalance
        return AccountBalance(
            id=row["id"],
            coin=row["coin"],
            wallet_balance=Decimal(str(row["wallet_balance"])),
            available_balance=Decimal(str(row["available_balance"])),
            frozen=Decimal(str(row["frozen"])),
            event_timestamp=row["event_timestamp"],
            received_at=row["received_at"],
            trace_id=row["trace_id"],
            equity=Decimal(str(row["equity"])) if row["equity"] is not None else None,
            usd_value=Decimal(str(row["usd_value"])) if row["usd_value"] is not None else None,
            margin_collateral=row["margin_collateral"] if row["margin_collateral"] is not None else False,
            total_order_im=Decimal(str(row["total_order_im"])) if row["total_order_im"] is not None else Decimal("0"),
            total_position_im=Decimal(str(row["total_position_im"])) if row["total_position_im"] is not None else Decimal("0"),
        )

    @staticmethod
    async def list_balances(
        coin: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[AccountBalance]:
        """List balance records with optional filters.

        Args:
            coin: Optional coin filter
            limit: Optional limit on number of records
            offset: Optional offset for pagination

        Returns:
            List of AccountBalance records
        """
        conditions = []
        params = []

        if coin is not None:
            conditions.append(f"coin = ${len(params) + 1}")
            params.append(coin)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"
            if offset is not None:
                limit_clause += f" OFFSET {offset}"

        query = f"""
            SELECT id, coin, wallet_balance, available_balance, frozen,
                   event_timestamp, received_at, trace_id,
                   equity, usd_value, margin_collateral, total_order_im, total_position_im
            FROM account_balances
            {where_clause}
            ORDER BY received_at DESC
            {limit_clause}
        """
        rows = await DatabaseConnection.fetch(query, *params)

        # Convert rows to AccountBalance instances
        balances = []
        for row in rows:
            balances.append(
                AccountBalance(
                    id=row["id"],
                    coin=row["coin"],
                    wallet_balance=Decimal(str(row["wallet_balance"])),
                    available_balance=Decimal(str(row["available_balance"])),
                    frozen=Decimal(str(row["frozen"])),
                    event_timestamp=row["event_timestamp"],
                    received_at=row["received_at"],
                    trace_id=row["trace_id"],
                    equity=Decimal(str(row["equity"])) if row["equity"] is not None else None,
                    usd_value=Decimal(str(row["usd_value"])) if row["usd_value"] is not None else None,
                    margin_collateral=row["margin_collateral"] if row["margin_collateral"] is not None else False,
                    total_order_im=Decimal(str(row["total_order_im"])) if row["total_order_im"] is not None else Decimal("0"),
                    total_position_im=Decimal(str(row["total_position_im"])) if row["total_position_im"] is not None else Decimal("0"),
                )
            )
        return balances

    @staticmethod
    async def get_by_id(balance_id: str) -> Optional[AccountBalance]:
        """Fetch a balance record by its ID.

        Args:
            balance_id: UUID string of the balance record

        Returns:
            AccountBalance if found, None otherwise
        """
        query = """
            SELECT id, coin, wallet_balance, available_balance, frozen,
                   event_timestamp, received_at, trace_id,
                   equity, usd_value, margin_collateral, total_order_im, total_position_im
            FROM account_balances
            WHERE id = $1
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query, balance_id)
        if not row:
            return None

        return AccountBalance(
            id=row["id"],
            coin=row["coin"],
            wallet_balance=Decimal(str(row["wallet_balance"])),
            available_balance=Decimal(str(row["available_balance"])),
            frozen=Decimal(str(row["frozen"])),
            event_timestamp=row["event_timestamp"],
            received_at=row["received_at"],
            trace_id=row["trace_id"],
            equity=Decimal(str(row["equity"])) if row["equity"] is not None else None,
            usd_value=Decimal(str(row["usd_value"])) if row["usd_value"] is not None else None,
            margin_collateral=row["margin_collateral"] if row["margin_collateral"] is not None else False,
            total_order_im=Decimal(str(row["total_order_im"])) if row["total_order_im"] is not None else Decimal("0"),
            total_position_im=Decimal(str(row["total_position_im"])) if row["total_position_im"] is not None else Decimal("0"),
        )

