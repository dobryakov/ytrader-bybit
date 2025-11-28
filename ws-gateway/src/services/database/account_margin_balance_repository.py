"""Database operations for AccountMarginBalance entity."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from .connection import DatabaseConnection
from ...config.logging import get_logger
from ...models.account_margin_balance import AccountMarginBalance

logger = get_logger(__name__)


class AccountMarginBalanceRepository:
    """Repository for CRUD operations on account_margin_balances table."""

    @staticmethod
    async def create_margin_balance(balance: AccountMarginBalance) -> AccountMarginBalance:
        """Insert a new account margin balance record into the database.

        Args:
            balance: AccountMarginBalance instance to persist

        Returns:
            The persisted AccountMarginBalance instance

        Raises:
            DatabaseError: If database operation fails
        """
        query = """
            INSERT INTO account_margin_balances (
                id, account_type, total_equity, total_wallet_balance, total_margin_balance,
                total_available_balance, total_initial_margin, total_maintenance_margin,
                total_order_im, base_currency, event_timestamp, received_at, trace_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        await DatabaseConnection.execute(
            query,
            balance.id,
            balance.account_type,
            balance.total_equity,
            balance.total_wallet_balance,
            balance.total_margin_balance,
            balance.total_available_balance,
            balance.total_initial_margin,
            balance.total_maintenance_margin,
            balance.total_order_im,
            balance.base_currency,
            balance.event_timestamp,
            balance.received_at,
            balance.trace_id,
        )
        logger.debug(
            "account_margin_balance_created",
            balance_id=str(balance.id),
            account_type=balance.account_type,
            base_currency=balance.base_currency,
            total_available_balance=str(balance.total_available_balance),
            trace_id=balance.trace_id,
        )
        return balance

    @staticmethod
    async def get_latest_margin_balance() -> Optional[AccountMarginBalance]:
        """Get the latest account margin balance record.

        Returns:
            Latest AccountMarginBalance, or None if not found
        """
        query = """
            SELECT id, account_type, total_equity, total_wallet_balance, total_margin_balance,
                   total_available_balance, total_initial_margin, total_maintenance_margin,
                   total_order_im, base_currency, event_timestamp, received_at, trace_id
            FROM account_margin_balances
            ORDER BY received_at DESC
            LIMIT 1
        """
        row = await DatabaseConnection.fetchrow(query)
        if not row:
            return None

        # Convert row to AccountMarginBalance
        return AccountMarginBalance(
            id=row["id"],
            account_type=row["account_type"],
            total_equity=Decimal(str(row["total_equity"])),
            total_wallet_balance=Decimal(str(row["total_wallet_balance"])),
            total_margin_balance=Decimal(str(row["total_margin_balance"])),
            total_available_balance=Decimal(str(row["total_available_balance"])),
            total_initial_margin=Decimal(str(row["total_initial_margin"])),
            total_maintenance_margin=Decimal(str(row["total_maintenance_margin"])),
            total_order_im=Decimal(str(row["total_order_im"])),
            base_currency=row["base_currency"],
            event_timestamp=row["event_timestamp"],
            received_at=row["received_at"],
            trace_id=row["trace_id"],
        )

