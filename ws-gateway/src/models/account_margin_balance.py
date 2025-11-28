"""AccountMarginBalance model for persisted account-level margin data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class AccountMarginBalance:
    """Represents account-level margin balance information persisted to PostgreSQL.

    This stores aggregate margin and balance information for unified accounts,
    allowing order-manager to check available margin without calling Bybit API.
    """

    id: UUID
    account_type: str
    total_equity: Decimal
    total_wallet_balance: Decimal
    total_margin_balance: Decimal
    total_available_balance: Decimal  # Available margin for trading
    total_initial_margin: Decimal  # Locked in positions
    total_maintenance_margin: Decimal
    total_order_im: Decimal  # Locked in orders
    base_currency: str  # Base currency for margin (USDT, USD, etc.)
    event_timestamp: datetime
    received_at: datetime
    trace_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        account_type: str,
        total_equity: Decimal,
        total_wallet_balance: Decimal,
        total_margin_balance: Decimal,
        total_available_balance: Decimal,
        base_currency: str,
        event_timestamp: datetime,
        total_initial_margin: Decimal = Decimal("0"),
        total_maintenance_margin: Decimal = Decimal("0"),
        total_order_im: Decimal = Decimal("0"),
        trace_id: Optional[str] = None,
    ) -> "AccountMarginBalance":
        """Factory for creating a new AccountMarginBalance instance."""
        return cls(
            id=uuid4(),
            account_type=account_type,
            total_equity=total_equity,
            total_wallet_balance=total_wallet_balance,
            total_margin_balance=total_margin_balance,
            total_available_balance=total_available_balance,
            total_initial_margin=total_initial_margin,
            total_maintenance_margin=total_maintenance_margin,
            total_order_im=total_order_im,
            base_currency=base_currency,
            event_timestamp=event_timestamp,
            received_at=datetime.utcnow(),
            trace_id=trace_id,
        )

    def validate(self) -> bool:
        """Validate margin balance constraints.

        Returns:
            True if validation passes, False otherwise.

        Validation rules:
        - All balance values except total_available_balance must be non-negative
        - total_available_balance can be negative (when positions are open and margin is used)
        """
        return (
            self.total_equity >= 0
            and self.total_wallet_balance >= 0
            and self.total_margin_balance >= 0
            # total_available_balance can be negative when margin is used for positions
            and self.total_initial_margin >= 0
            and self.total_maintenance_margin >= 0
            and self.total_order_im >= 0
        )

