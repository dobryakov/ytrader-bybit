"""
Account Balance database repository.

Reads latest available balance for coins from account_balances table.
"""

from typing import Optional, Dict, List
from decimal import Decimal
import asyncpg

from ..base import BaseRepository
from ...config.logging import get_logger

logger = get_logger(__name__)


class AccountBalanceRepository(BaseRepository[Dict[str, any]]):
    """Repository for reading account balances from shared database."""

    @property
    def table_name(self) -> str:
        """Return the table name."""
        return "account_balances"

    async def get_latest_balance(self, coin: str) -> Optional[Dict[str, any]]:
        """
        Get latest available balance for a coin.

        Args:
            coin: Coin symbol (e.g., 'USDT', 'BTC')

        Returns:
            Dictionary with balance data or None if not found:
            {
                'coin': str,
                'available_balance': Decimal,
                'wallet_balance': Decimal,
                'frozen': Decimal,
                'received_at': datetime,
                'event_timestamp': datetime
            }
        """
        query = f"""
            SELECT 
                coin, available_balance, wallet_balance, frozen,
                received_at, event_timestamp
            FROM {self.table_name}
            WHERE coin = $1
            ORDER BY received_at DESC
            LIMIT 1
        """
        try:
            record = await self._fetchrow(query, coin)
            if not record:
                logger.debug("No balance data found for coin", coin=coin)
                return None

            result = {
                "coin": record["coin"],
                "available_balance": Decimal(str(record["available_balance"])),
                "wallet_balance": Decimal(str(record["wallet_balance"])),
                "frozen": Decimal(str(record["frozen"])),
                "received_at": record["received_at"],
                "event_timestamp": record["event_timestamp"],
            }
            logger.debug("Retrieved latest balance", coin=coin, available_balance=result["available_balance"])
            return result
        except Exception as e:
            logger.error("Failed to fetch latest balance", coin=coin, error=str(e), exc_info=True)
            # Handle missing balance data gracefully - return None instead of raising
            return None

    async def get_latest_balances(self, coins: List[str]) -> Dict[str, Optional[Dict[str, any]]]:
        """
        Get latest available balance for multiple coins.

        Args:
            coins: List of coin symbols (e.g., ['USDT', 'BTC'])

        Returns:
            Dictionary mapping coin to balance data (or None if not found):
            {
                'USDT': {...balance data...},
                'BTC': {...balance data...} or None
            }
        """
        if not coins:
            return {}

        # Use IN clause for efficient batch query
        query = f"""
            SELECT DISTINCT ON (coin)
                coin, available_balance, wallet_balance, frozen,
                received_at, event_timestamp
            FROM {self.table_name}
            WHERE coin = ANY($1::VARCHAR[])
            ORDER BY coin, received_at DESC
        """
        try:
            records = await self._fetch(query, coins)
            result = {}

            # Initialize all coins to None
            for coin in coins:
                result[coin] = None

            # Fill in found balances
            for record in records:
                coin = record["coin"]
                result[coin] = {
                    "coin": coin,
                    "available_balance": Decimal(str(record["available_balance"])),
                    "wallet_balance": Decimal(str(record["wallet_balance"])),
                    "frozen": Decimal(str(record["frozen"])),
                    "received_at": record["received_at"],
                    "event_timestamp": record["event_timestamp"],
                }
                logger.debug("Retrieved latest balance", coin=coin, available_balance=result[coin]["available_balance"])

            # Log missing coins
            missing_coins = [coin for coin in coins if result[coin] is None]
            if missing_coins:
                logger.debug("No balance data found for coins", coins=missing_coins)

            return result
        except Exception as e:
            logger.error("Failed to fetch latest balances", coins=coins, error=str(e), exc_info=True)
            # Handle gracefully - return dict with all None values
            return {coin: None for coin in coins}

