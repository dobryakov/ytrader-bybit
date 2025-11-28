"""AccountBalance model for persisted balance data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class AccountBalance:
    """Represents account balance information persisted to PostgreSQL.

    This mirrors the `account_balances` table defined in data-model.md and is used
    for storing critical balance data that requires immediate persistence.
    """

    id: UUID
    coin: str
    wallet_balance: Decimal
    available_balance: Decimal
    frozen: Decimal
    event_timestamp: datetime
    received_at: datetime
    trace_id: Optional[str] = None
    # Additional fields for margin calculations
    equity: Optional[Decimal] = None
    usd_value: Optional[Decimal] = None
    margin_collateral: bool = False
    total_order_im: Decimal = Decimal("0")
    total_position_im: Decimal = Decimal("0")

    @classmethod
    def create(
        cls,
        coin: str,
        wallet_balance: Decimal,
        available_balance: Decimal,
        frozen: Decimal,
        event_timestamp: datetime,
        trace_id: Optional[str] = None,
        equity: Optional[Decimal] = None,
        usd_value: Optional[Decimal] = None,
        margin_collateral: bool = False,
        total_order_im: Decimal = Decimal("0"),
        total_position_im: Decimal = Decimal("0"),
    ) -> "AccountBalance":
        """Factory for creating a new AccountBalance instance."""
        return cls(
            id=uuid4(),
            coin=coin,
            wallet_balance=wallet_balance,
            available_balance=available_balance,
            frozen=frozen,
            event_timestamp=event_timestamp,
            received_at=datetime.utcnow(),
            trace_id=trace_id,
            equity=equity,
            usd_value=usd_value,
            margin_collateral=margin_collateral,
            total_order_im=total_order_im,
            total_position_im=total_position_im,
        )

    def validate(self) -> bool:
        """Validate balance constraints.

        Returns:
            True if validation passes, False otherwise.

        Validation rules:
        - All balance values must be non-negative
        - wallet_balance must equal available_balance + frozen
        """
        if self.wallet_balance < 0 or self.available_balance < 0 or self.frozen < 0:
            return False

        # Check sum consistency (allow small floating point differences)
        expected_sum = self.available_balance + self.frozen
        difference = abs(self.wallet_balance - expected_sum)
        # Allow for small floating point differences (0.00000001)
        return difference <= Decimal("0.00000001")

