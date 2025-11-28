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

        # Ensure coin is a string (not a list)
        coin_str = str(coin) if coin and not isinstance(coin, list) else None
        if not coin_str:
            logger.warning(
                "balance_parse_invalid_coin_type",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                coin=coin,
                coin_type=type(coin).__name__,
            )
            return None
        
        # Create AccountBalance instance
        balance = AccountBalance.create(
            coin=coin_str,
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
    def _parse_all_balances_from_event(event: Event) -> list[AccountBalance]:
        """Parse all AccountBalance records from an Event payload.
        
        Bybit wallet messages contain an array of coins, each needs to be saved separately.

        Args:
            event: Event with event_type='balance' and balance data in payload

        Returns:
            List of AccountBalance instances (one per coin)
        """
        balances = []
        payload = event.payload
        
        # Extract data array from Bybit wallet message
        # Bybit wallet messages always have data as an array
        data = payload.get("data")
        if not isinstance(data, list) or len(data) == 0:
            # No data array - skip parsing
            logger.warning(
                "balance_parse_no_data_array",
                event_id=str(event.event_id),
                trace_id=event.trace_id,
                payload_keys=list(payload.keys()) if isinstance(payload, dict) else [],
            )
            return balances
        
        # Process each account data entry
        for account_data in data:
            coins = account_data.get("coin", [])
            if not isinstance(coins, list):
                continue
                
            # Process each coin in the array
            for coin_data in coins:
                if not isinstance(coin_data, dict):
                    logger.warning(
                        "balance_parse_invalid_coin_data_type",
                        event_id=str(event.event_id),
                        trace_id=event.trace_id,
                        coin_data_type=type(coin_data).__name__,
                    )
                    continue
                
                coin = coin_data.get("coin")
                if not coin or isinstance(coin, list):
                    logger.warning(
                        "balance_parse_invalid_coin",
                        event_id=str(event.event_id),
                        trace_id=event.trace_id,
                        coin=coin,
                        coin_type=type(coin).__name__ if coin else None,
                    )
                    continue
                
                # Ensure coin is a string
                coin = str(coin)
                
                wallet_balance = coin_data.get("walletBalance")
                locked = coin_data.get("locked", 0) or 0
                frozen = locked  # locked is the frozen amount
                
                # Get available balance - availableToWithdraw can be empty string in Bybit messages
                available_balance = coin_data.get("availableToWithdraw")
                if available_balance == "" or available_balance is None:
                    # Try availableBalance as fallback
                    available_balance = coin_data.get("availableBalance")
                
                # If still empty/None, calculate from walletBalance - locked
                if available_balance == "" or available_balance is None:
                    # Calculate: available = walletBalance - locked
                    try:
                        wallet_bal = Decimal(str(wallet_balance)) if wallet_balance else Decimal(0)
                        locked_decimal = Decimal(str(locked)) if locked else Decimal(0)
                        available_balance = wallet_bal - locked_decimal
                    except (InvalidOperation, ValueError, TypeError):
                        # If calculation fails, default to 0
                        available_balance = Decimal(0)
                
                # Parse decimal values
                try:
                    wallet_balance_decimal = Decimal(str(wallet_balance)) if wallet_balance else Decimal(0)
                    # available_balance might already be a Decimal if we calculated it above
                    if isinstance(available_balance, Decimal):
                        available_balance_decimal = available_balance
                    else:
                        available_balance_decimal = Decimal(str(available_balance)) if available_balance else Decimal(0)
                    frozen_decimal = Decimal(str(frozen)) if frozen else Decimal(0)
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
                    continue
                
                # Create AccountBalance instance
                balance = AccountBalance.create(
                    coin=str(coin),  # Ensure it's a string
                    wallet_balance=wallet_balance_decimal,
                    available_balance=available_balance_decimal,
                    frozen=frozen_decimal,
                    event_timestamp=event.timestamp,
                    trace_id=event.trace_id,
                )
                balances.append(balance)
        
        return balances

    @staticmethod
    async def persist_balance_from_event(event: Event) -> bool:
        """Persist a balance event to the database.

        This method implements the core balance persistence logic:
        1. Parse balance data from event payload (handles multiple coins)
        2. Validate balance constraints
        3. Persist to database
        4. Handle errors gracefully (per FR-017)

        Args:
            event: Event with event_type='balance' and balance data in payload

        Returns:
            True if at least one balance was persisted successfully, False otherwise

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
            # Parse all balances from event (wallet messages contain multiple coins)
            balances = BalanceService._parse_all_balances_from_event(event)
            if not balances:
                logger.warning(
                    "balance_persist_parse_failed",
                    event_id=str(event.event_id),
                    trace_id=event.trace_id,
                )
                return False

            # Persist each balance to database
            success_count = 0
            for balance in balances:
                try:
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
                        continue

                    # Persist to database
                    await BalanceRepository.create_balance(balance)
                    success_count += 1

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
                except DatabaseError as e:
                    logger.error(
                        "balance_persist_database_error",
                        event_id=str(event.event_id),
                        trace_id=event.trace_id,
                        coin=balance.coin,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue with next balance
                    continue
                except Exception as e:
                    logger.error(
                        "balance_persist_unexpected_error",
                        event_id=str(event.event_id),
                        trace_id=event.trace_id,
                        coin=balance.coin,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue with next balance
                    continue

            return success_count > 0

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

