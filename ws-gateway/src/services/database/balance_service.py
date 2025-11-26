"""Balance persistence service that handles validation and error recovery."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from ...config.logging import get_logger
from ...exceptions import DatabaseError
from ...models.account_balance import AccountBalance
from ...models.event import Event
from .balance_repository import BalanceRepository

logger = get_logger(__name__)


class BalanceService:
    """Service for persisting balance events to PostgreSQL.

    This service implements User Story 5: Store Critical Data Directly to Database.
    It handles balance validation, persistence, and graceful error handling per FR-017.
    """

    @staticmethod
    def _parse_balance_from_event(event: Event) -> Optional[AccountBalance]:
        """Parse AccountBalance from an Event payload.

        Bybit wallet messages have structure:
        - data[0].coin[] - array of coin balances
        - Each coin has: coin, walletBalance, availableToWithdraw (or availableBalance), locked (frozen)

        Args:
            event: Event with event_type='balance' and balance data in payload

        Returns:
            AccountBalance instance if parsing succeeds, None otherwise
        """
        payload = event.payload

        # Bybit wallet messages have data array with coin array inside
        # Handle both direct coin fields (for simpler events) and nested structure
        coin = payload.get("coin")
        wallet_balance = payload.get("wallet_balance") or payload.get("walletBalance")
        available_balance = payload.get("available_balance") or payload.get("availableBalance") or payload.get("availableToWithdraw")
        frozen = payload.get("frozen") or payload.get("locked", 0)

        # If coin is not directly in payload, try to extract from data array structure
        if not coin:
            # Check if payload has data array (Bybit wallet message format)
            data = payload.get("data")
            if isinstance(data, list) and len(data) > 0:
                first_data = data[0]
                coins = first_data.get("coin", [])
                if isinstance(coins, list) and len(coins) > 0:
                    # Take first coin from array (can be extended to handle multiple coins)
                    coin_data = coins[0]
                    coin = coin_data.get("coin")
                    wallet_balance = coin_data.get("walletBalance")
                    available_balance = coin_data.get("availableToWithdraw") or coin_data.get("availableBalance")
                    frozen = coin_data.get("locked", 0)

        if not coin:
            logger.warning(
                "balance_parse_missing_coin",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                payload_keys=list(payload.keys()) if isinstance(payload, dict) else [],
            )
            return None

        # Parse decimal values with error handling
        try:
            wallet_balance_decimal = Decimal(str(wallet_balance)) if wallet_balance is not None else Decimal(0)
            available_balance_decimal = Decimal(str(available_balance)) if available_balance is not None else Decimal(0)
            frozen_decimal = Decimal(str(frozen)) if frozen is not None else Decimal(0)
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.warning(
                "balance_parse_invalid_decimal",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                error=str(e),
                coin=coin,
                wallet_balance=wallet_balance,
                available_balance=available_balance,
                frozen=frozen,
            )
            return None

        # Create AccountBalance instance
        balance = AccountBalance.create(
            coin=coin,
            wallet_balance=wallet_balance_decimal,
            available_balance=available_balance_decimal,
            frozen=frozen_decimal,
            event_timestamp=event.timestamp,
            trace_id=event.trace_id,
        )

        return balance

    @staticmethod
    def _validate_balance(balance: AccountBalance) -> bool:
        """Validate balance constraints (non-negative, sum consistency).

        Args:
            balance: AccountBalance instance to validate

        Returns:
            True if validation passes, False otherwise
        """
        return balance.validate()

    @staticmethod
    async def persist_balance_from_event(event: Event) -> bool:
        """Persist a balance event to the database.

        This method implements the core balance persistence logic:
        1. Parse balance data from event payload
        2. Validate balance constraints
        3. Persist to database
        4. Handle errors gracefully (per FR-017)

        Args:
            event: Event with event_type='balance' and balance data in payload

        Returns:
            True if persistence succeeded, False otherwise

        Note:
            Per FR-017, database write failures must not block WebSocket processing.
            This method logs errors and returns False on failure, allowing the
            event processing pipeline to continue.
        """
        if event.event_type != "balance":
            logger.debug(
                "balance_persist_skipped_not_balance_event",
                event_id=str(event.event_id),
                event_type=event.event_type,
                trace_id=event.trace_id,
            )
            return False

        try:
            # Parse balance from event
            balance = BalanceService._parse_balance_from_event(event)
            if not balance:
                logger.warning(
                    "balance_persist_parse_failed",
                    event_id=str(event.event_id),
                    trace_id=event.trace_id,
                )
                return False

            # Validate balance constraints
            if not BalanceService._validate_balance(balance):
                logger.warning(
                    "balance_persist_validation_failed",
                    event_id=str(event.event_id),
                    coin=balance.coin,
                    wallet_balance=str(balance.wallet_balance),
                    available_balance=str(balance.available_balance),
                    frozen=str(balance.frozen),
                    trace_id=event.trace_id,
                )
                return False

            # Persist to database
            await BalanceRepository.create_balance(balance)

            logger.info(
                "balance_persisted",
                event_id=str(event.event_id),
                balance_id=str(balance.id),
                coin=balance.coin,
                wallet_balance=str(balance.wallet_balance),
                available_balance=str(balance.available_balance),
                frozen=str(balance.frozen),
                trace_id=event.trace_id,
            )
            return True

        except DatabaseError as e:
            # Database errors are logged but don't block processing (per FR-017)
            logger.error(
                "balance_persist_database_error",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

        except Exception as e:
            # Unexpected errors are logged but don't block processing (per FR-017)
            logger.error(
                "balance_persist_unexpected_error",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    @staticmethod
    async def get_latest_balance(coin: str) -> Optional[AccountBalance]:
        """Get the latest balance for a specific coin.

        Args:
            coin: Coin symbol (e.g., 'USDT', 'BTC')

        Returns:
            Latest AccountBalance for the coin, or None if not found
        """
        try:
            return await BalanceRepository.get_latest_balance(coin)
        except Exception as e:
            logger.error(
                "balance_get_latest_error",
                coin=coin,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

