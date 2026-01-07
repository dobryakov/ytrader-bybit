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

    async def _trigger_balance_sync(
        self, 
        context: Dict[str, Any], 
        force: bool = False
    ) -> bool:
        """
        Trigger on-demand balance sync via ws-gateway REST API.

        This method respects BALANCE_SYNC_MIN_INTERVAL_SECONDS to avoid
        overloading ws-gateway/Bybit, unless force=True (e.g., when data is stale).
        Logs all outcomes but does not raise to the caller (it returns False on failure).

        Args:
            context: Context dictionary for logging
            force: If True, bypass minimum interval check (for stale data)
        """
        if not settings.balance_sync_enabled:
            logger.info(
                "Balance sync via ws-gateway is disabled by configuration",
                context=context,
            )
            return False

        now = datetime.now(timezone.utc)

        # Fast path check before acquiring lock (skip if force=True)
        if not force and self._last_sync_at is not None:
            elapsed = (now - self._last_sync_at).total_seconds()
            if elapsed < settings.balance_sync_min_interval_seconds:
                logger.info(
                    "Skipping balance sync, minimum interval not elapsed",
                    elapsed_seconds=elapsed,
                    min_interval_seconds=settings.balance_sync_min_interval_seconds,
                    context=context,
                    force=force,
                )
                return False

        async with _balance_sync_lock:
            # Re-check inside the lock to avoid races (skip if force=True)
            now = datetime.now(timezone.utc)
            if not force and self._last_sync_at is not None:
                elapsed = (now - self._last_sync_at).total_seconds()
                if elapsed < settings.balance_sync_min_interval_seconds:
                    logger.info(
                        "Skipping balance sync inside lock, minimum interval not elapsed",
                        elapsed_seconds=elapsed,
                        min_interval_seconds=settings.balance_sync_min_interval_seconds,
                        context=context,
                        force=force,
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
                    # Return sync result with updated coins info
                    return {
                        "success": True,
                        "updated_coins": payload.get("updated_coins", []),
                        "updated_count": payload.get("updated_count", 0),
                    }

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
            sync_result = await self._trigger_balance_sync(freshness_context)
            if not sync_result or (isinstance(sync_result, dict) and not sync_result.get("success")):
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
                    logger.info(
                        "Balance data is stale, triggering sync",
                        required_currency=coin,
                        balance_received_at=received_at.isoformat(),
                        balance_age_seconds=balance_age_seconds,
                        max_age_seconds=settings.balance_data_max_age_seconds,
                        context=freshness_context,
                    )
                    # Try to sync and re-check
                    # Force sync when data is stale (bypass min interval)
                    sync_result = await self._trigger_balance_sync(freshness_context, force=True)
                    if not sync_result or (isinstance(sync_result, dict) and not sync_result.get("success")):
                        logger.warning(
                            "Balance data is stale and sync failed",
                            required_currency=coin,
                            balance_age_seconds=balance_age_seconds,
                            max_age_seconds=settings.balance_data_max_age_seconds,
                            context=freshness_context,
                        )
                        return None
                    
                    # Check if required currency was updated
                    updated_coins = sync_result.get("updated_coins", []) if isinstance(sync_result, dict) else []
                    if updated_coins and coin not in updated_coins:
                        logger.warning(
                            "Required currency was not updated during sync",
                            required_currency=coin,
                            updated_coins=updated_coins,
                            context=freshness_context,
                            note="Sync completed but required currency not in updated list. This may indicate the currency has zero balance or validation failed.",
                        )
                        # Continue anyway - maybe the currency was updated but not reported
                    
                    # Wait a bit for ws-gateway to persist data to database
                    # BalanceService.sync_from_rest() is async and may take time to write to DB
                    import asyncio
                    await asyncio.sleep(1.0)  # Increased delay to allow DB write to complete
                    
                    # Retry reading balance with exponential backoff
                    max_retries = 3
                    for attempt in range(max_retries):
                        balance_data = await self.balance_repo.get_latest_balance(coin)
                        if balance_data:
                            # Check if data is fresh
                            received_at = balance_data.get("received_at")
                            if received_at:
                                now = datetime.now(timezone.utc)
                                if getattr(received_at, "tzinfo", None) is None:
                                    received_at = received_at.replace(tzinfo=timezone.utc)
                                new_balance_age_seconds = (now - received_at).total_seconds()
                                if new_balance_age_seconds <= settings.balance_data_max_age_seconds:
                                    # Data is fresh, return it
                                    balance_data["_age_seconds"] = new_balance_age_seconds
                                    return balance_data
                                elif attempt < max_retries - 1:
                                    # Data still stale, wait and retry
                                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff: 0.5s, 1s, 2s
                                    logger.debug(
                                        "Balance data still stale after sync, retrying",
                                        required_currency=coin,
                                        attempt=attempt + 1,
                                        max_retries=max_retries,
                                        balance_age_seconds=new_balance_age_seconds,
                                        wait_time=wait_time,
                                        context=freshness_context,
                                    )
                                    await asyncio.sleep(wait_time)
                                    continue
                    
                    # If we get here, data is still stale or unavailable
                    if balance_data:
                        received_at = balance_data.get("received_at")
                        if received_at:
                            now = datetime.now(timezone.utc)
                            if getattr(received_at, "tzinfo", None) is None:
                                received_at = received_at.replace(tzinfo=timezone.utc)
                            balance_age_seconds = (now - received_at).total_seconds()
                            logger.warning(
                                "Balance data remains stale after sync and retries",
                                required_currency=coin,
                                balance_received_at=received_at.isoformat(),
                                balance_age_seconds=balance_age_seconds,
                                max_age_seconds=settings.balance_data_max_age_seconds,
                                updated_coins=updated_coins,
                                context=freshness_context,
                            )
                    else:
                        logger.warning(
                            "Balance data still unavailable after sync for stale snapshot",
                            required_currency=coin,
                            updated_coins=updated_coins,
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
            # For unified accounts, try account-level balance first (same as get_available_balance_for_buy)
            # This ensures consistency between the two methods
            account_balance = await self._get_account_level_balance()
            
            if account_balance is not None and account_balance > 0:
                # Use account-level balance if available and positive (same logic as get_available_balance_for_buy)
                usable_balance = account_balance * self.safety_margin
                
                logger.debug(
                    "Using account-level balance for affordable amount check",
                    trading_pair=trading_pair,
                    account_available_balance=account_balance,
                    usable_balance=usable_balance,
                    safety_margin=self.safety_margin,
                    requested_amount=requested_amount,
                )
            else:
                # Fallback to coin-level balance if account-level is not available
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
                
                logger.debug(
                    "Using coin-level balance for affordable amount check",
                    trading_pair=trading_pair,
                    available_balance=available_balance,
                    usable_balance=usable_balance,
                    safety_margin=self.safety_margin,
                    balance_age_seconds=balance_age_seconds,
                    requested_amount=requested_amount,
                )

            # Check if we have enough quote currency
            balance_source = "account-level" if account_balance is not None and account_balance > 0 else "coin-level"
            
            if usable_balance >= requested_amount:
                # We have enough, return requested amount
                logger.debug(
                    "Sufficient balance for buy order",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    usable_balance=usable_balance,
                    safety_margin=self.safety_margin,
                    balance_source=balance_source,
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
                    usable_balance=usable_balance,
                    safety_margin=self.safety_margin,
                    balance_source=balance_source,
                )
                return round(adapted_amount, 2)
            else:
                # Insufficient balance
                shortfall = requested_amount - usable_balance if usable_balance > 0 else requested_amount
                logger.warning(
                    "Insufficient balance for buy order",
                    trading_pair=trading_pair,
                    requested_amount=requested_amount,
                    usable_balance=usable_balance,
                    safety_margin=self.safety_margin,
                    shortfall=shortfall,
                    balance_source=balance_source,
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
            
            # Get full position data to check sign and closed_at status
            position_data = await position_manager_client.get_position(trading_pair)
            
            if position_data is None:
                # No position found - cannot sell
                logger.warning(
                    "No position found for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    requested_amount=requested_amount,
                )
                return None
            
            # Check if position is closed (closed_at is set)
            closed_at = position_data.get("closed_at")
            if closed_at is not None:
                # Position is marked as closed - cannot sell
                logger.warning(
                    "Position is closed, cannot sell",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    requested_amount=requested_amount,
                    closed_at=closed_at,
                )
                return None
            
            # Get position size with sign (positive = long, negative = short)
            size_str = position_data.get("size")
            if size_str is None:
                logger.warning(
                    "Position data missing size",
                    trading_pair=trading_pair,
                    position_data=position_data,
                )
                return None
            
            position_size_signed = float(size_str)
            position_size_abs = abs(position_size_signed)
            
            logger.info(
                "Position size retrieved for sell order check",
                trading_pair=trading_pair,
                base_currency=base_currency,
                position_size_signed=position_size_signed,
                position_size_abs=position_size_abs,
                requested_amount_usdt=requested_amount,
                current_price=current_price,
            )
            
            # For short positions (negative size), SELL increases the short position
            # For long positions (positive size), SELL closes/reduces the long position
            # We only allow SELL for long positions (to close them)
            if position_size_signed <= 0:
                # Short position or zero - SELL would increase short position, not close it
                # This is not what we want for a SELL signal
                logger.warning(
                    "Cannot sell: position is short or zero, SELL would increase short position",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size_signed=position_size_signed,
                    requested_amount=requested_amount,
                )
                return None
            
            # Use absolute position size for calculations
            position_size = position_size_abs

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
                # Convert position_size (base currency) to USDT using a conservative estimate
                # Use requested_amount as a proxy for price if available, otherwise return None
                if requested_amount > 0:
                    # Estimate price from requested_amount / some reasonable quantity
                    # This is a fallback - ideally we should always have current_price
                    estimated_price = requested_amount / max(position_size, 0.001)
                    adapted_amount_usdt = (position_size * self.safety_margin) * estimated_price
                    logger.warning(
                        "Current price not available, estimating from requested amount",
                        trading_pair=trading_pair,
                        position_size=position_size,
                        requested_amount=requested_amount,
                        estimated_price=estimated_price,
                        adapted_amount_usdt=adapted_amount_usdt,
                    )
                    return round(adapted_amount_usdt, 2)
                else:
                    logger.error(
                        "Cannot convert position size to USDT without price and requested_amount",
                        trading_pair=trading_pair,
                        position_size=position_size,
                        requested_amount=requested_amount,
                    )
                    return None

            # Convert requested_amount (USDT) to base currency quantity
            requested_quantity_base = requested_amount / current_price

            # Apply safety margin to position size
            usable_position_size = position_size * self.safety_margin
            
            logger.info(
                "Calculating sell order affordability",
                trading_pair=trading_pair,
                base_currency=base_currency,
                position_size=position_size,
                safety_margin=self.safety_margin,
                usable_position_size=usable_position_size,
                requested_amount_usdt=requested_amount,
                requested_quantity_base=requested_quantity_base,
                current_price=current_price,
            )

            if usable_position_size >= requested_quantity_base:
                # We have enough position, return requested amount in USDT
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
                # Return requested amount in USDT (already in correct currency)
                return round(requested_amount, 2)
            elif usable_position_size > 0:
                # Adapt amount to available position size
                adapted_quantity_base = usable_position_size
                # Convert back to USDT
                adapted_amount_usdt = adapted_quantity_base * current_price
                logger.info(
                    "Adapting sell amount to available position size",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size=position_size,
                    usable_position_size=usable_position_size,
                    requested_amount=requested_amount,
                    requested_quantity_base=requested_quantity_base,
                    adapted_quantity_base=adapted_quantity_base,
                    adapted_amount_usdt=adapted_amount_usdt,
                    current_price=current_price,
                )
                # Return adapted amount in USDT
                return round(adapted_amount_usdt, 2)
            else:
                # Insufficient position
                # Calculate what we could actually sell
                max_sellable_amount_usdt = usable_position_size * current_price if usable_position_size > 0 else 0.0
                logger.warning(
                    "Insufficient position size for sell order",
                    trading_pair=trading_pair,
                    base_currency=base_currency,
                    position_size=position_size,
                    usable_position_size=usable_position_size,
                    safety_margin=self.safety_margin,
                    requested_amount=requested_amount,
                    requested_quantity_base=requested_quantity_base,
                    max_sellable_amount_usdt=max_sellable_amount_usdt,
                    current_price=current_price,
                    reason=f"Position size ({position_size} {base_currency}) * safety_margin ({self.safety_margin}) = {usable_position_size} {base_currency} is less than required {requested_quantity_base} {base_currency}",
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

    async def _get_account_level_balance(self) -> Optional[float]:
        """
        Get account-level total available balance from account_margin_balances table.
        
        For unified accounts, this represents the total available margin for trading,
        which is more accurate than coin-level balances when there are borrowed funds.
        
        Returns:
            Total available balance in base currency (usually USDT), or None if unavailable
        """
        query = """
            SELECT total_available_balance, base_currency, received_at
            FROM account_margin_balances
            ORDER BY received_at DESC
            LIMIT 1
        """
        try:
            record = await self.balance_repo._fetchrow(query)
            if not record:
                return None
            
            total_available_balance = float(record["total_available_balance"])
            base_currency = record["base_currency"]
            received_at = record["received_at"]
            
            # Check if data is fresh
            if received_at:
                # Normalize datetime to timezone-aware UTC (asyncpg returns TIMESTAMP as timezone-naive)
                now = datetime.now(timezone.utc)
                if getattr(received_at, "tzinfo", None) is None:
                    received_at = received_at.replace(tzinfo=timezone.utc)
                age_seconds = (now - received_at).total_seconds()
                if age_seconds > settings.balance_data_max_age_seconds:
                    logger.warning(
                        "Account-level balance data is stale",
                        age_seconds=age_seconds,
                        max_age_seconds=settings.balance_data_max_age_seconds,
                    )
            
            logger.debug(
                "Retrieved account-level balance",
                total_available_balance=total_available_balance,
                base_currency=base_currency,
            )
            
            return total_available_balance
        except Exception as e:
            logger.warning(
                "Failed to fetch account-level balance",
                error=str(e),
            )
            return None

    async def get_available_balance_for_buy(
        self,
        trading_pair: str,
        current_price: Optional[float] = None,
    ) -> Optional[float]:
        """
        Get available balance in quote currency for BUY orders.

        For unified accounts, uses account-level totalAvailableBalance instead of
        coin-level available_balance, as coin-level balance can be negative when
        there are borrowed funds, while account-level balance shows the actual
        available margin for trading.

        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            current_price: Current market price (optional, not used for BUY)

        Returns:
            Available balance in quote currency after applying safety margin, or None if unavailable
        """
        base_currency, quote_currency = self._extract_currencies(trading_pair)
        
        context = {
            "trading_pair": trading_pair,
            "signal_type": "buy",
            "required_currency": quote_currency,
        }
        
        # For unified accounts, try account-level balance first (more accurate)
        # This handles cases where coin-level balance is negative due to borrowed funds
        account_balance = await self._get_account_level_balance()
        
        if account_balance is not None and account_balance > 0:
            # Use account-level balance if available and positive
            usable_balance = account_balance * self.safety_margin
            
            logger.info(
                "Using account-level balance for buy order",
                trading_pair=trading_pair,
                quote_currency=quote_currency,
                account_available_balance=account_balance,
                usable_balance=usable_balance,
                safety_margin=self.safety_margin,
            )
            
            return round(usable_balance, 2)
        
        # Fallback to coin-level balance if account-level is not available
        balance_data = await self._get_fresh_balance(quote_currency, context)
        if not balance_data:
            logger.warning(
                "Balance data unavailable for buy order calculation",
                **context,
            )
            return None
        
        available_balance = float(balance_data["available_balance"])
        
        # If coin-level balance is negative or zero, return None
        # (this shouldn't happen if account-level balance was used, but handle it anyway)
        if available_balance <= 0:
            logger.warning(
                "Coin-level balance is non-positive, cannot calculate buy order amount",
                trading_pair=trading_pair,
                quote_currency=quote_currency,
                available_balance=available_balance,
            )
            return None
        
        usable_balance = available_balance * self.safety_margin
        
        balance_age_seconds = balance_data.get("_age_seconds")
        
        logger.debug(
            "Using coin-level balance for buy order",
            trading_pair=trading_pair,
            quote_currency=quote_currency,
            available_balance=available_balance,
            usable_balance=usable_balance,
            balance_age_seconds=balance_age_seconds,
            safety_margin=self.safety_margin,
        )
        
        return round(usable_balance, 2)

    async def get_available_position_for_sell(
        self,
        trading_pair: str,
        current_price: float,
    ) -> Optional[float]:
        """
        Get available position size in quote currency (USDT) for SELL orders.

        Args:
            trading_pair: Trading pair symbol (e.g., 'ETHUSDT')
            current_price: Current market price (required for conversion)

        Returns:
            Available position size in quote currency after applying safety margin, or None if unavailable/closed/short
        """
        base_currency, _ = self._extract_currencies(trading_pair)
        
        context = {
            "trading_pair": trading_pair,
            "signal_type": "sell",
            "base_currency": base_currency,
        }
        
        # Get position data
        position_data = await position_manager_client.get_position(trading_pair)
        
        if position_data is None:
            logger.debug(
                "No position found for sell order calculation",
                **context,
            )
            return None
        
        # Check if position is closed
        closed_at = position_data.get("closed_at")
        if closed_at is not None:
            logger.debug(
                "Position is closed, cannot calculate sell amount",
                **context,
                closed_at=closed_at,
            )
            return None
        
        # Get position size with sign
        size_str = position_data.get("size")
        if size_str is None:
            logger.warning(
                "Position data missing size",
                trading_pair=trading_pair,
                position_data=position_data,
            )
            return None
        
        position_size_signed = float(size_str)
        
        # Only allow SELL for long positions (positive size)
        if position_size_signed <= 0:
            logger.debug(
                "Position is short or zero, cannot calculate sell amount",
                trading_pair=trading_pair,
                position_size_signed=position_size_signed,
            )
            return None
        
        position_size_abs = abs(position_size_signed)
        
        # Apply safety margin
        usable_position_size = position_size_abs * self.safety_margin
        
        # Convert to quote currency (USDT)
        if current_price is None or current_price <= 0:
            logger.warning(
                "Current price not provided for sell order calculation",
                trading_pair=trading_pair,
                position_size=position_size_abs,
            )
            return None
        
        available_amount_usdt = usable_position_size * current_price
        
        logger.debug(
            "Available position for sell order",
            trading_pair=trading_pair,
            base_currency=base_currency,
            position_size=position_size_abs,
            usable_position_size=usable_position_size,
            safety_margin=self.safety_margin,
            current_price=current_price,
            available_amount_usdt=available_amount_usdt,
        )
        
        return round(available_amount_usdt, 2)


# Global balance calculator instance
balance_calculator = BalanceCalculator(safety_margin=0.95)

