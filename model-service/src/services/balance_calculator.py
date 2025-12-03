"""
Balance-aware signal amount calculator.

Calculates maximum affordable amount for trading signals based on available balance.
When balance snapshots from the database are stale or missing, this calculator can
optionally trigger an on-demand balance sync via ws-gateway's REST API and then
re-read the latest snapshot, similar in spirit to how order-manager refreshes
balance directly from Bybit.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import asyncio

import httpx

from ..database.repositories.account_balance_repo import AccountBalanceRepository
from ..config.logging import get_logger
from ..config.settings import settings
from ..services.position_manager_client import position_manager_client

logger = get_logger(__name__)

# Global lock to avoid spamming ws-gateway with concurrent balance sync requests
_balance_sync_lock = asyncio.Lock()


class BalanceCalculator:
    """Calculates signal amounts based on available balance."""

    def __init__(self, safety_margin: Optional[float] = None):
        """
        Initialize balance calculator.

        Args:
            safety_margin: Safety margin to leave (0.95 = use 95% of available balance)
        """
        # Allow overriding safety margin for tests, otherwise use configuration
        if safety_margin is not None:
            self.safety_margin = safety_margin
        else:
            self.safety_margin = settings.balance_adaptation_safety_margin
        self.balance_repo = AccountBalanceRepository()
        # Track last successful sync trigger time to enforce min interval
        self._last_sync_at: Optional[datetime] = None

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

    async def _trigger_balance_sync(self, context: Dict[str, Any]) -> bool:
        """
        Trigger on-demand balance sync via ws-gateway REST API.

        This method respects BALANCE_SYNC_MIN_INTERVAL_SECONDS to avoid
        overloading ws-gateway/Bybit, and logs all outcomes but does not raise
        to the caller (it returns False on failure).
        """
        if not settings.balance_sync_enabled:
            logger.info(
                "Balance sync via ws-gateway is disabled by configuration",
                context=context,
            )
            return False

        now = datetime.now(timezone.utc)

        # Fast path check before acquiring lock
        if self._last_sync_at is not None:
            elapsed = (now - self._last_sync_at).total_seconds()
            if elapsed < settings.balance_sync_min_interval_seconds:
                logger.info(
                    "Skipping balance sync, minimum interval not elapsed",
                    elapsed_seconds=elapsed,
                    min_interval_seconds=settings.balance_sync_min_interval_seconds,
                    context=context,
                )
                return False

        async with _balance_sync_lock:
            # Re-check inside the lock to avoid races
            now = datetime.now(timezone.utc)
            if self._last_sync_at is not None:
                elapsed = (now - self._last_sync_at).total_seconds()
                if elapsed < settings.balance_sync_min_interval_seconds:
                    logger.info(
                        "Skipping balance sync inside lock, minimum interval not elapsed",
                        elapsed_seconds=elapsed,
                        min_interval_seconds=settings.balance_sync_min_interval_seconds,
                        context=context,
                    )
                    return False

            ws_url = settings.ws_gateway_url.rstrip("/")
            sync_endpoint = f"{ws_url}/api/v1/balances/sync"

            headers = {
                "X-API-Key": settings.ws_gateway_api_key,
                "Content-Type": "application/json",
            }

            try:
                timeout = settings.balance_sync_timeout_seconds
            except Exception:
                timeout = 5.0

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(sync_endpoint, headers=headers)
                    status_code = response.status_code
                    # Try to parse JSON but don't fail if it's not JSON
                    try:
                        payload = response.json()
                    except Exception:
                        payload = {"raw_text": response.text}

                if status_code >= 200 and status_code < 300:
                    self._last_sync_at = now
                    logger.info(
                        "Balance sync via ws-gateway completed successfully",
                        status_code=status_code,
                        updated_coins=payload.get("updated_coins"),
                        updated_count=payload.get("updated_count"),
                        context=context,
                    )
                    return True

                logger.warning(
                    "Balance sync via ws-gateway failed with non-2xx status",
                    status_code=status_code,
                    response=payload,
                    context=context,
                )
                return False
            except httpx.RequestError as e:
                logger.error(
                    "Balance sync via ws-gateway request error",
                    error=str(e),
                    context=context,
                )
                return False
            except Exception as e:
                logger.error(
                    "Balance sync via ws-gateway unexpected error",
                    error=str(e),
                    context=context,
                )
                return False

    async def _get_fresh_balance(
        self,
        coin: str,
        freshness_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest balance for a coin, optionally triggering ws-gateway sync when stale.

        Returns:
            Fresh balance dict or None if unable to obtain a fresh snapshot.
        """
        balance_data = await self.balance_repo.get_latest_balance(coin)

        if not balance_data:
            logger.warning(
                "Balance data unavailable from database",
                required_currency=coin,
                context=freshness_context,
            )
            # Try sync, then re-read
            synced = await self._trigger_balance_sync(freshness_context)
            if not synced:
                return None
            balance_data = await self.balance_repo.get_latest_balance(coin)
            if not balance_data:
                logger.warning(
                    "Balance data still unavailable after sync",
                    required_currency=coin,
                    context=freshness_context,
                )
                return None

        received_at = balance_data.get("received_at")
        balance_age_seconds: Optional[float] = None
        if received_at:
            try:
                now = datetime.now(timezone.utc)
                if getattr(received_at, "tzinfo", None) is None:
                    received_at = received_at.replace(tzinfo=timezone.utc)
                balance_age_seconds = (now - received_at).total_seconds()
                if balance_age_seconds > settings.balance_data_max_age_seconds:
                    logger.warning(
                        "Balance data is stale before sync attempt",
                        required_currency=coin,
                        balance_received_at=received_at.isoformat(),
                        balance_age_seconds=balance_age_seconds,
                        max_age_seconds=settings.balance_data_max_age_seconds,
                        context=freshness_context,
                    )
                    # Try to sync and re-check
                    synced = await self._trigger_balance_sync(freshness_context)
                    if not synced:
                        return None
                    balance_data = await self.balance_repo.get_latest_balance(coin)
                    if not balance_data:
                        logger.warning(
                            "Balance data still unavailable after sync for stale snapshot",
                            required_currency=coin,
                            context=freshness_context,
                        )
                        return None
                    # Recalculate age with new snapshot
                    received_at = balance_data.get("received_at")
                    if received_at:
                        now = datetime.now(timezone.utc)
                        if getattr(received_at, "tzinfo", None) is None:
                            received_at = received_at.replace(tzinfo=timezone.utc)
                        balance_age_seconds = (now - received_at).total_seconds()
                        if balance_age_seconds > settings.balance_data_max_age_seconds:
                            logger.warning(
                                "Balance data remains stale after sync",
                                required_currency=coin,
                                balance_received_at=received_at.isoformat(),
                                balance_age_seconds=balance_age_seconds,
                                max_age_seconds=settings.balance_data_max_age_seconds,
                                context=freshness_context,
                            )
                            return None
                    else:
                        logger.warning(
                            "Balance data after sync missing received_at timestamp",
                            required_currency=coin,
                            context=freshness_context,
                        )
                        return None

            except Exception as e:
                logger.warning(
                    "Failed to evaluate balance data freshness",
                    required_currency=coin,
                    error=str(e),
                    context=freshness_context,
                )
                return None
        else:
            logger.warning(
                "Balance data missing received_at timestamp, cannot verify freshness",
                required_currency=coin,
                context=freshness_context,
            )
            return None

        # Attach age info for callers
        balance_data["_age_seconds"] = balance_age_seconds
        return balance_data

    async def calculate_affordable_amount(
        self,
        trading_pair: str,
        signal_type: str,
        requested_amount: float,
        current_price: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate maximum affordable amount based on available balance.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            signal_type: Signal type ('buy' or 'sell')
            requested_amount: Requested signal amount in quote currency
            current_price: Current market price (optional, used for SELL signal conversion)

        Returns:
            Adapted amount that fits available balance, or None if insufficient balance.
            Returns None if balance data is unavailable.
        """
        # Determine required currency
        required_currency = self._get_required_currency(trading_pair, signal_type)

        context = {
            "trading_pair": trading_pair,
            "signal_type": signal_type,
            "required_currency": required_currency,
        }

        if signal_type.lower() == "buy":
            # Buy: requested_amount is in quote currency (USDT)
            # Get fresh balance for quote currency (with optional sync)
            balance_data = await self._get_fresh_balance(required_currency, context)
            if not balance_data:
                logger.warning(
                    "Balance data unavailable or stale after sync, cannot calculate affordable amount",
                    **context,
                )
                return None

            balance_age_seconds: Optional[float] = balance_data.get("_age_seconds")
            available_balance = float(balance_data["available_balance"])

            # Apply safety margin
            usable_balance = available_balance * self.safety_margin

            # Check if we have enough quote currency
            if usable_balance >= requested_amount:
                # We have enough, return requested amount
                logger.debug(
                    "Sufficient balance for buy order",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    available_balance=available_balance,
                    usable_balance=usable_balance,
                    balance_age_seconds=balance_age_seconds,
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
                    balance_age_seconds=balance_age_seconds,
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
                    balance_age_seconds=balance_age_seconds,
                )
                return None
        else:  # sell
            # Sell: requested_amount is in quote currency (USDT), but we need base currency
            # For SELL signals, use position size instead of balance, as balance may be 0
            # when assets are locked in open positions
            base_currency, _ = self._extract_currencies(trading_pair)
            
            logger.debug(
                "Checking position size for sell order",
                trading_pair=trading_pair,
                base_currency=base_currency,
                requested_amount=requested_amount,
                current_price=current_price,
            )
            
            # Get position size from position-manager
            position_size = await position_manager_client.get_position_size(trading_pair)
            
            logger.debug(
                "Position size retrieved",
                trading_pair=trading_pair,
                position_size=position_size,
            )
            
            if position_size is None or position_size <= 0:
                # No open position - cannot sell
                logger.warning(
                    "No open position for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    requested_amount=requested_amount,
                )
                return None

            # Convert requested_amount (in quote currency) to base currency
            # If current_price is not provided, we'll use a conservative approach
            if current_price is None or current_price <= 0:
                logger.warning(
                    "Current price not provided for sell order conversion, using position size directly",
                    trading_pair=trading_pair,
                    position_size=position_size,
                    requested_amount=requested_amount,
                )
                # Without price, we can't convert accurately, but we know we have position_size
                # Return position_size in base currency (conservative - use all available position)
                logger.warning(
                    "Current price not available, using full position size",
                    trading_pair=trading_pair,
                    position_size=position_size,
                    requested_amount=requested_amount,
                )
                return round(position_size * self.safety_margin, 6)

            # Convert requested_amount (USDT) to base currency quantity
            requested_quantity_base = requested_amount / current_price

            # Apply safety margin to position size
            usable_position_size = position_size * self.safety_margin

            if usable_position_size >= requested_quantity_base:
                # We have enough position, return requested amount in base currency
                logger.debug(
                    "Sufficient position size for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size=position_size,
                    usable_position_size=usable_position_size,
                    requested_amount=requested_amount,
                    requested_quantity_base=requested_quantity_base,
                    current_price=current_price,
                )
                # Return in base currency (ETH), not USDT
                return round(requested_quantity_base, 6)
            elif usable_position_size > 0:
                # Adapt amount to available position size
                adapted_quantity_base = usable_position_size
                logger.info(
                    "Adapting sell amount to available position size",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size=position_size,
                    usable_position_size=usable_position_size,
                    requested_amount=requested_amount,
                    requested_quantity_base=requested_quantity_base,
                    adapted_quantity_base=adapted_quantity_base,
                    current_price=current_price,
                )
                # Return in base currency (ETH), not USDT
                return round(adapted_quantity_base, 6)
            else:
                # Insufficient position
                logger.warning(
                    "Insufficient position size for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size=position_size,
                    usable_position_size=usable_position_size,
                    requested_amount=requested_amount,
                    requested_quantity_base=requested_quantity_base,
                    current_price=current_price,
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

