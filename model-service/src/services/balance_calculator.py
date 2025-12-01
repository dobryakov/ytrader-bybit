"""
Balance-aware signal amount calculator.

Calculates maximum affordable amount for trading signals based on available balance.
"""

from typing import Optional
from decimal import Decimal

from ..database.repositories.account_balance_repo import AccountBalanceRepository
from ..config.logging import get_logger
from ..config.settings import settings

logger = get_logger(__name__)


class BalanceCalculator:
    """Calculates signal amounts based on available balance."""

    def __init__(self, safety_margin: float = 0.95):
        """
        Initialize balance calculator.

        Args:
            safety_margin: Safety margin to leave (0.95 = use 95% of available balance)
        """
        self.safety_margin = safety_margin
        self.balance_repo = AccountBalanceRepository()

    def _extract_currencies(self, trading_pair: str) -> tuple[str, str]:
        """
        Extract base and quote currency from trading pair.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')

        Returns:
            Tuple of (base_currency, quote_currency)
        """
        # Common quote currencies (usually 4 characters)
        quote_currencies = ["USDT", "USDC", "BUSD", "DAI", "TUSD"]
        
        # Try to match quote currency from the end
        for quote in quote_currencies:
            if trading_pair.endswith(quote):
                base = trading_pair[:-len(quote)]
                return (base, quote)
        
        # Fallback: assume last 4 characters are quote currency
        # This handles most cases like BTCUSDT, ETHUSDT
        if len(trading_pair) > 4:
            quote = trading_pair[-4:]
            base = trading_pair[:-4]
            return (base, quote)
        
        # If we can't determine, assume USDT as quote (most common)
        logger.warning("Could not determine currencies from trading pair, assuming USDT as quote", trading_pair=trading_pair)
        return (trading_pair, "USDT")

    def _get_required_currency(self, trading_pair: str, signal_type: str) -> str:
        """
        Determine which currency is required for the order type.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            signal_type: Signal type ('buy' or 'sell')

        Returns:
            Required currency symbol (e.g., 'USDT' for buy, 'BTC' for sell)
        """
        base_currency, quote_currency = self._extract_currencies(trading_pair)
        
        if signal_type.lower() == "buy":
            # Buy order requires quote currency (USDT to buy BTC)
            return quote_currency
        else:  # sell
            # Sell order requires base currency (BTC to sell)
            return base_currency

    async def calculate_affordable_amount(
        self,
        trading_pair: str,
        signal_type: str,
        requested_amount: float,
    ) -> Optional[float]:
        """
        Calculate maximum affordable amount based on available balance.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            signal_type: Signal type ('buy' or 'sell')
            requested_amount: Requested signal amount in quote currency

        Returns:
            Adapted amount that fits available balance, or None if insufficient balance.
            Returns None if balance data is unavailable.
        """
        # Determine required currency
        required_currency = self._get_required_currency(trading_pair, signal_type)
        
        # Get latest balance for required currency
        balance_data = await self.balance_repo.get_latest_balance(required_currency)
        
        if not balance_data:
            logger.warning(
                "Balance data unavailable, cannot calculate affordable amount",
                trading_pair=trading_pair,
                signal_type=signal_type,
                required_currency=required_currency,
            )
            return None
        
        available_balance = float(balance_data["available_balance"])
        
        # Apply safety margin
        usable_balance = available_balance * self.safety_margin
        
        # For buy orders, amount is already in quote currency (USDT)
        # For sell orders, we need to check if we have enough base currency
        # But since requested_amount is in quote currency, we need to convert
        # For simplicity, we'll assume requested_amount is always in quote currency
        # and check if we have enough quote currency for buy, or enough base currency for sell
        
        if signal_type.lower() == "buy":
            # Buy: requested_amount is in quote currency (USDT)
            # Check if we have enough quote currency
            if usable_balance >= requested_amount:
                # We have enough, return requested amount
                logger.debug(
                    "Sufficient balance for buy order",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    available_balance=available_balance,
                    usable_balance=usable_balance,
                )
                return requested_amount
            elif usable_balance > 0:
                # Adapt amount to available balance
                adapted_amount = usable_balance
                logger.info(
                    "Adapting buy amount to available balance",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    adapted_amount=adapted_amount,
                    available_balance=available_balance,
                    usable_balance=usable_balance,
                )
                return round(adapted_amount, 2)
            else:
                # Insufficient balance
                logger.warning(
                    "Insufficient balance for buy order",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    available_balance=available_balance,
                    usable_balance=usable_balance,
                )
                return None
        else:  # sell
            # Sell: requested_amount is in quote currency (USDT), but we need base currency
            # This is a simplification - in reality, we'd need current price to convert
            # For now, we'll check if we have any base currency available
            # If we have base currency, we assume we can sell it
            base_currency, _ = self._extract_currencies(trading_pair)
            base_balance_data = await self.balance_repo.get_latest_balance(base_currency)
            
            if not base_balance_data:
                logger.warning(
                    "Base currency balance unavailable for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    requested_amount=requested_amount,
                )
                return None
            
            base_available = float(base_balance_data["available_balance"])
            
            # For sell orders, if we have base currency, we can proceed
            # The amount check is more complex (needs price conversion), but for now
            # we'll return requested_amount if we have any base currency
            if base_available > 0:
                logger.debug(
                    "Sufficient base currency for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    base_available=base_available,
                    requested_amount=requested_amount,
                )
                return requested_amount
            else:
                logger.warning(
                    "Insufficient base currency for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    base_available=base_available,
                    requested_amount=requested_amount,
                )
                return None

    async def check_balance_sufficient(
        self,
        trading_pair: str,
        signal_type: str,
        amount: float,
    ) -> bool:
        """
        Check if balance is sufficient for the requested amount.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            signal_type: Signal type ('buy' or 'sell')
            amount: Requested amount in quote currency

        Returns:
            True if balance is sufficient, False otherwise
        """
        affordable = await self.calculate_affordable_amount(trading_pair, signal_type, amount)
        return affordable is not None and affordable >= amount * 0.99  # Allow 1% tolerance


# Global balance calculator instance
balance_calculator = BalanceCalculator(safety_margin=0.95)

