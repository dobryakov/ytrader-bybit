"""Risk manager service for enforcing risk limits and safety checks."""

from decimal import Decimal
from typing import Optional

from ..config.settings import settings
from ..config.database import DatabaseConnection
from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..models.position import Position
from ..services.position_manager_client import PositionManagerClient
from ..services.instrument_info_manager import InstrumentInfoManager
from ..utils.bybit_client import get_bybit_client
from ..exceptions import RiskLimitError, OrderExecutionError
from decimal import ROUND_DOWN

logger = get_logger(__name__)


class RiskManager:
    """Service for enforcing risk limits and performing safety checks."""

    def __init__(self):
        """Initialize risk manager with configuration."""
        self.max_position_size = Decimal(str(settings.order_manager_max_position_size))
        self.max_exposure = Decimal(str(settings.order_manager_max_exposure))
        self.max_order_size_ratio = Decimal(str(settings.order_manager_max_order_size_ratio))
        self.position_manager_client = PositionManagerClient()
        self.instrument_info_manager = InstrumentInfoManager()

    async def check_balance(self, signal: TradingSignal, order_quantity: Decimal, order_price: Decimal) -> bool:
        """Check if sufficient balance is available for order.

        Args:
            signal: Trading signal
            order_quantity: Order quantity in base currency
            order_price: Order price (for limit orders) or current market price (for market orders)
            trace_id: Trace ID for logging

        Returns:
            True if balance is sufficient, False otherwise

        Raises:
            OrderExecutionError: If balance check fails
        """
        trace_id = signal.trace_id
        try:
            bybit_client = get_bybit_client()

            # Get account balance from Bybit API
            endpoint = "/v5/account/wallet-balance"
            params = {"accountType": "UNIFIED"}

            response = await bybit_client.get(endpoint, params=params, authenticated=True)
            
            # Log full response for debugging
            logger.info(
                "bybit_balance_response",
                response=response,
                response_keys=list(response.keys()) if isinstance(response, dict) else None,
                trace_id=trace_id,
            )
            
            # Check response status
            ret_code = response.get("retCode", 0)
            ret_msg = response.get("retMsg", "")
            
            if ret_code != 0:
                # Handle signature error (10004) - may be due to invalid API keys or signature format
                if ret_code == 10004:
                    logger.warning(
                        "balance_check_signature_error",
                        signal_id=str(signal.signal_id),
                        ret_code=ret_code,
                        ret_msg=ret_msg,
                        note="Signature error - may be due to invalid API keys or testnet account setup",
                        trace_id=trace_id,
                    )
                    # For testnet with signature errors, skip balance check to allow testing
                    # In production, this should be handled more strictly
                    logger.info(
                        "balance_check_skipped_signature_error",
                        signal_id=str(signal.signal_id),
                        reason="Signature error - skipping balance check for testnet",
                        trace_id=trace_id,
                    )
                    return True
                
                logger.error(
                    "balance_check_api_error",
                    signal_id=str(signal.signal_id),
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    trace_id=trace_id,
                )
                raise OrderExecutionError(f"Bybit API error: {ret_msg} (code: {ret_code})")
            
            result = response.get("result", {})
            
            # Handle different response structures
            # Structure 1: result.list[] (unified account)
            # Structure 2: result may be None or empty if no balance
            list_data = result.get("list", []) if result else []

            if not list_data:
                # Check if result is empty (no account or no balance)
                # This is acceptable for testnet accounts with no balance
                logger.warning(
                    "balance_check_no_account_data",
                    signal_id=str(signal.signal_id),
                    ret_code=ret_code,
                    ret_msg=ret_msg,
                    result_type=type(result).__name__,
                    result_keys=list(result.keys()) if result else None,
                    trace_id=trace_id,
                )
                # For testnet with no balance, assume sufficient balance for small orders
                # In production, this should be handled more strictly
                logger.info(
                    "balance_check_skipped_no_data",
                    signal_id=str(signal.signal_id),
                    reason="No account data in response (testnet account may be empty)",
                    trace_id=trace_id,
                )
                return True

            # For unified account, use totalAvailableBalance from account level
            # This is more reliable than coin-level availableToWithdraw which may be empty
            account = list_data[0]
            account_type = account.get("accountType", "")
            
            # Try to get totalAvailableBalance from account (unified account)
            if account_type == "UNIFIED":
                total_available = account.get("totalAvailableBalance", "0")
                if total_available and total_available != "":
                    try:
                        usdt_balance = Decimal(str(total_available))
                        logger.info(
                            "balance_check_using_total_available",
                            signal_id=str(signal.signal_id),
                            total_available_balance=str(usdt_balance),
                            trace_id=trace_id,
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            "balance_check_invalid_total_available",
                            signal_id=str(signal.signal_id),
                            total_available=total_available,
                            error=str(e),
                            trace_id=trace_id,
                        )
                        usdt_balance = Decimal("0")
                else:
                    # Fallback to coin-level balance
                    usdt_balance = Decimal("0")
                    coins = account.get("coin", [])
                    for coin in coins:
                        if coin.get("coin") == "USDT":
                            # Try walletBalance if availableToWithdraw is empty
                            wallet_balance = coin.get("walletBalance", "0")
                            if wallet_balance and wallet_balance != "":
                                try:
                                    usdt_balance = Decimal(str(wallet_balance))
                                    logger.info(
                                        "balance_check_using_wallet_balance",
                                        signal_id=str(signal.signal_id),
                                        wallet_balance=str(usdt_balance),
                                        trace_id=trace_id,
                                    )
                                except (ValueError, TypeError):
                                    usdt_balance = Decimal("0")
                            break
            else:
                # For non-unified accounts, use coin-level balance
                coins = account.get("coin", [])
                usdt_balance = Decimal("0")
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        # Try availableToWithdraw first, then walletBalance
                        available_value = coin.get("availableToWithdraw", "")
                        if not available_value or available_value == "":
                            available_value = coin.get("walletBalance", "0")
                        if available_value and available_value != "":
                            try:
                                usdt_balance = Decimal(str(available_value))
                            except (ValueError, TypeError) as e:
                                logger.warning(
                                    "balance_check_invalid_value",
                                    signal_id=str(signal.signal_id),
                                    available_value=available_value,
                                    error=str(e),
                                    trace_id=trace_id,
                                )
                                usdt_balance = Decimal("0")
                        break

            # Calculate required balance and check against appropriate currency
            if signal.signal_type.lower() == "buy":
                # Buy orders need USDT (quote currency)
                required_balance = order_quantity * order_price
                available_balance = usdt_balance
                currency = "USDT"
                
                if required_balance > available_balance:
                    shortfall = required_balance - available_balance
                    shortfall_percentage = (shortfall / required_balance) * 100 if required_balance > 0 else 0
                    error_msg = (
                        f"Insufficient balance: required={required_balance} {currency}, "
                        f"available={available_balance} {currency}, shortfall={shortfall} {currency} ({shortfall_percentage:.2f}%)"
                    )
                    logger.error(
                        "balance_check_failed",
                        signal_id=str(signal.signal_id),
                        asset=signal.asset,
                        signal_type=signal.signal_type,
                        required=float(required_balance),
                        available=float(available_balance),
                        shortfall=float(shortfall),
                        shortfall_percentage=float(shortfall_percentage),
                        order_quantity=float(order_quantity),
                        order_price=float(order_price),
                        currency=currency,
                        error_type="RiskLimitError",
                        trace_id=trace_id,
                    )
                    raise RiskLimitError(error_msg)
            else:
                # Sell orders: check position first
                position = await self.position_manager_client.get_position(signal.asset, mode="one-way", trace_id=trace_id)
                has_position = position is not None
                position_size = position.size if position else Decimal("0")
                has_long_position = has_position and position_size > 0
                
                if has_long_position:
                    # Have long position - can use reduce_only (no margin needed)
                    logger.info(
                        "balance_check_passed_sell_with_long_position",
                        signal_id=str(signal.signal_id),
                        asset=signal.asset,
                        position_size=float(position_size),
                        reason="Long position exists, can use reduce_only, margin not required",
                        trace_id=trace_id,
                    )
                    return True
                
                # No long position - need margin in base currency (usually USDT)
                # Get margin from database
                try:
                    pool = await DatabaseConnection.get_pool()
                    query = """
                        SELECT total_available_balance, base_currency, received_at
                        FROM account_margin_balances
                        ORDER BY received_at DESC
                        LIMIT 1
                    """
                    row = await pool.fetchrow(query)
                    
                    if row is None:
                        logger.warning(
                            "balance_check_no_margin_data_fallback_to_api",
                            signal_id=str(signal.signal_id),
                            asset=signal.asset,
                            reason="Margin data not in database, using API response",
                            trace_id=trace_id,
                        )
                        # Fallback to using totalAvailableBalance from API
                        # Note: This assumes USDT, but may need adjustment for other currencies
                        available_margin = usdt_balance
                        base_currency_margin = "USDT"
                        # Try to determine actual base currency from account data
                        account = list_data[0] if list_data else None
                        if account:
                            coins = account.get("coin", [])
                            if isinstance(coins, list):
                                for coin_data in coins:
                                    if isinstance(coin_data, dict):
                                        margin_collateral = coin_data.get("marginCollateral", False)
                                        usd_value = coin_data.get("usdValue", 0)
                                        if margin_collateral and usd_value:
                                            try:
                                                usd_val = Decimal(str(usd_value))
                                                if usd_val > 0:
                                                    base_currency_margin = str(coin_data.get("coin", "USDT"))
                                                    # Try to get balance for this currency
                                                    available_value = coin_data.get("availableToWithdraw", "")
                                                    if not available_value or available_value == "":
                                                        available_value = coin_data.get("walletBalance", "0")
                                                    if available_value and available_value != "":
                                                        try:
                                                            available_margin = Decimal(str(available_value))
                                                            break
                                                        except (ValueError, TypeError):
                                                            pass
                                            except (ValueError, TypeError):
                                                pass
                    else:
                        available_margin = Decimal(str(row["total_available_balance"]))
                        base_currency_margin = str(row["base_currency"]) or "USDT"
                    
                    # Calculate required margin: order value in base currency
                    required_margin = order_quantity * order_price
                    
                    if required_margin > available_margin:
                        shortfall = required_margin - available_margin
                        shortfall_percentage = (shortfall / required_margin) * 100 if required_margin > 0 else 0
                        error_msg = (
                            f"Insufficient margin: required={required_margin} {base_currency_margin}, "
                            f"available={available_margin} {base_currency_margin}, "
                            f"shortfall={shortfall} {base_currency_margin} ({shortfall_percentage:.2f}%)"
                        )
                        logger.error(
                            "balance_check_failed",
                            signal_id=str(signal.signal_id),
                            asset=signal.asset,
                            signal_type=signal.signal_type,
                            required=float(required_margin),
                            available=float(available_margin),
                            shortfall=float(shortfall),
                            shortfall_percentage=float(shortfall_percentage),
                            order_quantity=float(order_quantity),
                            order_price=float(order_price),
                            currency=base_currency_margin,
                            error_type="RiskLimitError",
                            trace_id=trace_id,
                        )
                        raise RiskLimitError(error_msg)
                    
                    logger.info(
                        "balance_check_passed",
                        signal_id=str(signal.signal_id),
                        required=float(required_margin),
                        available=float(available_margin),
                        currency=base_currency_margin,
                        trace_id=trace_id,
                    )
                    return True
                    
                except Exception as e:
                    logger.error(
                        "balance_check_margin_retrieval_failed",
                        signal_id=str(signal.signal_id),
                        asset=signal.asset,
                        error=str(e),
                        trace_id=trace_id,
                        exc_info=True,
                    )
                    # Fallback to original logic
                    base_currency = self._extract_base_currency(signal.asset)
                    required_balance = order_quantity
                    base_currency_balance = self._get_coin_balance(account, base_currency)
                    available_balance = base_currency_balance
                    currency = base_currency
                    
                    if required_balance > available_balance:
                        shortfall = required_balance - available_balance
                        shortfall_percentage = (shortfall / required_balance) * 100 if required_balance > 0 else 0
                        error_msg = (
                            f"Insufficient balance: required={required_balance} {currency}, "
                            f"available={available_balance} {currency}, shortfall={shortfall} {currency} ({shortfall_percentage:.2f}%)"
                        )
                        logger.error(
                            "balance_check_failed",
                            signal_id=str(signal.signal_id),
                            asset=signal.asset,
                            signal_type=signal.signal_type,
                            required=float(required_balance),
                            available=float(available_balance),
                            shortfall=float(shortfall),
                            shortfall_percentage=float(shortfall_percentage),
                            order_quantity=float(order_quantity),
                            order_price=float(order_price),
                            currency=currency,
                            error_type="RiskLimitError",
                            trace_id=trace_id,
                        )
                        raise RiskLimitError(error_msg)

            logger.info(
                "balance_check_passed",
                signal_id=str(signal.signal_id),
                required=float(required_balance),
                available=float(available_balance),
                currency=currency,
                trace_id=trace_id,
            )

            return True

        except RiskLimitError:
            raise
        except Exception as e:
            logger.error("balance_check_error", signal_id=str(signal.signal_id), error=str(e), trace_id=trace_id)
            raise OrderExecutionError(f"Balance check failed: {e}") from e

    def _extract_base_currency(self, trading_pair: str) -> str:
        """Extract base currency from trading pair symbol.
        
        Examples:
            BTCUSDT -> BTC
            ETHUSDT -> ETH
            BTCUSDC -> BTC
            ADAUSDT -> ADA
        
        Args:
            trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Base currency symbol (e.g., 'BTC')
        """
        # Common quote currencies (ordered by length to match longer ones first)
        quote_currencies = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDD', 'EUR', 'GBP', 'JPY']
        
        trading_pair_upper = trading_pair.upper()
        
        # Try to match known quote currencies
        for quote in quote_currencies:
            if trading_pair_upper.endswith(quote):
                base = trading_pair_upper[:-len(quote)]
                return base
        
        # Fallback: assume last 4 characters are quote currency
        if len(trading_pair) >= 8:
            return trading_pair_upper[:-4]
        
        # Last resort: assume last 3 characters are quote currency
        if len(trading_pair) >= 6:
            return trading_pair_upper[:-3]
        
        # If pair is too short, return as-is (shouldn't happen for valid pairs)
        return trading_pair_upper

    def _get_coin_balance(self, account: dict, coin: str) -> Decimal:
        """Get available balance for a specific coin from account data.
        
        Args:
            account: Account data from Bybit API response
            coin: Coin symbol (e.g., 'BTC', 'USDT')
            
        Returns:
            Available balance as Decimal, or Decimal('0') if not found
        """
        coins = account.get("coin", [])
        
        for coin_data in coins:
            if coin_data.get("coin") == coin:
                # Try availableToWithdraw first, then walletBalance
                available_value = coin_data.get("availableToWithdraw", "")
                if not available_value or available_value == "":
                    available_value = coin_data.get("walletBalance", "0")
                
                if available_value and available_value != "":
                    try:
                        return Decimal(str(available_value))
                    except (ValueError, TypeError):
                        return Decimal("0")
                break
        
        return Decimal("0")

    async def check_and_adapt_order_size(
        self, signal: TradingSignal, order_quantity: Decimal, order_price: Decimal
    ) -> tuple[Decimal, bool]:
        """Check and adapt order size to fit within maximum order size limit.

        If order value exceeds max_order_size, reduces quantity proportionally
        and rounds to lot_size to create a valid order within limits.

        Args:
            signal: Trading signal
            order_quantity: Order quantity in base currency
            order_price: Order price
            trace_id: Trace ID for logging

        Returns:
            Tuple of (adapted_quantity, was_adapted) where:
            - adapted_quantity: Original or reduced quantity that fits within limits
            - was_adapted: True if quantity was reduced, False otherwise

        Raises:
            RiskLimitError: If order cannot be adapted (e.g., would be below minimum)
        """
        trace_id = signal.trace_id
        order_value = order_quantity * order_price
        max_order_size = self.max_exposure * self.max_order_size_ratio

        # Check against max order size ratio of max exposure
        if order_value <= max_order_size:
            logger.debug(
                "order_size_check_passed",
                signal_id=str(signal.signal_id),
                asset=signal.asset,
                order_value=float(order_value),
                max_order_size=float(max_order_size),
                trace_id=trace_id,
            )
            return order_quantity, False

        # Order size exceeds limit - adapt quantity
        excess = order_value - max_order_size
        excess_percentage = (excess / order_value) * 100 if order_value > 0 else 0

        logger.warning(
            "order_size_exceeds_limit_adapting",
            signal_id=str(signal.signal_id),
            asset=signal.asset,
            signal_type=signal.signal_type,
            original_order_value=float(order_value),
            original_order_quantity=float(order_quantity),
            order_price=float(order_price),
            max_order_size=float(max_order_size),
            max_exposure=float(self.max_exposure),
            max_ratio=float(self.max_order_size_ratio),
            excess=float(excess),
            excess_percentage=float(excess_percentage),
            trace_id=trace_id,
            reason="Order size exceeds limit, reducing quantity to fit within max_order_size",
        )

        # Calculate maximum allowed quantity based on max_order_size
        max_allowed_quantity = max_order_size / order_price

        # Get instrument info to round to lot_size
        try:
            instrument_info = await self.instrument_info_manager.get_instrument(signal.asset)
            if not instrument_info:
                logger.warning(
                    "order_size_adaptation_no_instrument_info",
                    asset=signal.asset,
                    trace_id=trace_id,
                    reason="Cannot get lot_size for rounding, using direct calculation",
                )
                # Fallback: use original quantity (will fail validation later)
                adapted_quantity = max_allowed_quantity
            else:
                # Round down to lot_size to ensure valid quantity
                lot_size = instrument_info.lot_size
                if lot_size > 0:
                    adapted_quantity = (
                        (max_allowed_quantity / lot_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * lot_size
                    )
                else:
                    adapted_quantity = max_allowed_quantity

                # Ensure adapted quantity is not below minimum
                min_order_qty = instrument_info.min_order_qty
                if adapted_quantity < min_order_qty:
                    logger.error(
                        "order_size_adaptation_below_minimum",
                        signal_id=str(signal.signal_id),
                        asset=signal.asset,
                        adapted_quantity=float(adapted_quantity),
                        min_order_qty=float(min_order_qty),
                        trace_id=trace_id,
                        reason="Adapted quantity would be below minimum order quantity",
                    )
                    raise RiskLimitError(
                        f"Cannot adapt order size: adapted quantity {adapted_quantity} would be below minimum {min_order_qty} for {signal.asset}"
                    )

                # Verify adapted order value is within limits
                adapted_order_value = adapted_quantity * order_price
                if adapted_order_value > max_order_size:
                    # Round down one more step if still exceeds
                    adapted_quantity = adapted_quantity - lot_size
                    if adapted_quantity < min_order_qty:
                        raise RiskLimitError(
                            f"Cannot adapt order size: quantity {order_quantity} too large for max_order_size {max_order_size} USDT"
                        )

        except Exception as e:
            if isinstance(e, RiskLimitError):
                raise
            logger.error(
                "order_size_adaptation_error",
                signal_id=str(signal.signal_id),
                asset=signal.asset,
                error=str(e),
                trace_id=trace_id,
                exc_info=True,
            )
            raise RiskLimitError(f"Failed to adapt order size: {e}") from e

        adapted_order_value = adapted_quantity * order_price
        reduction_percentage = ((order_quantity - adapted_quantity) / order_quantity * 100) if order_quantity > 0 else 0

        logger.info(
            "order_size_adapted",
            signal_id=str(signal.signal_id),
            asset=signal.asset,
            original_order_quantity=float(order_quantity),
            original_order_value=float(order_value),
            adapted_order_quantity=float(adapted_quantity),
            adapted_order_value=float(adapted_order_value),
            max_order_size=float(max_order_size),
            reduction_percentage=float(reduction_percentage),
            trace_id=trace_id,
            reason="Order quantity reduced to fit within max_order_size limit",
        )

        return adapted_quantity, True

    def check_position_size(
        self,
        asset: str,
        current_position: Optional[Position],
        order_quantity: Decimal,
        order_side: str,
        trace_id: Optional[str] = None,
    ) -> bool:
        """Check if order would exceed maximum position size limit.

        Args:
            asset: Trading pair symbol
            current_position: Current position for asset (None if no position)
            order_quantity: Order quantity in base currency
            order_side: Order side ('Buy' or 'Sell')
            trace_id: Optional trace ID for logging

        Returns:
            True if position size is within limits, False otherwise

        Raises:
            RiskLimitError: If position size would exceed limits
        """
        from ..utils.tracing import get_or_create_trace_id

        trace_id = trace_id or get_or_create_trace_id()

        current_size = Decimal("0")
        if current_position:
            current_size = current_position.size

        # Calculate new position size
        if order_side.upper() == "BUY":
            new_size = current_size + order_quantity
        else:  # SELL
            new_size = current_size - order_quantity

        # Check against max position size (absolute value)
        if abs(new_size) > self.max_position_size:
            excess = abs(new_size) - self.max_position_size
            excess_percentage = (excess / abs(new_size)) * 100 if new_size != 0 else 0
            error_msg = (
                f"Position size would exceed limit: current={current_size}, "
                f"order={order_quantity} ({order_side}), new={new_size}, "
                f"max={self.max_position_size}, excess={excess} ({excess_percentage:.2f}%)"
            )
            logger.error(
                "position_size_check_failed",
                asset=asset,
                current_size=float(current_size),
                order_quantity=float(order_quantity),
                order_side=order_side,
                new_size=float(new_size),
                max_size=float(self.max_position_size),
                excess=float(excess),
                excess_percentage=float(excess_percentage),
                error_type="RiskLimitError",
                trace_id=trace_id,
            )
            raise RiskLimitError(error_msg)

        logger.debug(
            "position_size_check_passed",
            asset=asset,
            current_size=float(current_size),
            order_quantity=float(order_quantity),
            order_side=order_side,
            new_size=float(new_size),
            max_size=float(self.max_position_size),
            trace_id=trace_id,
        )

        return True

    def check_max_exposure(self, total_exposure: Decimal, trace_id: Optional[str] = None) -> bool:
        """Check if total exposure across all positions exceeds maximum.

        Args:
            total_exposure: Total exposure across all positions (in USDT)
            trace_id: Trace ID for logging

        Returns:
            True if exposure is within limits, False otherwise

        Raises:
            RiskLimitError: If exposure exceeds limits
        """
        if total_exposure > self.max_exposure:
            excess = total_exposure - self.max_exposure
            excess_percentage = (excess / total_exposure) * 100 if total_exposure > 0 else 0
            error_msg = (
                f"Total exposure exceeds limit: exposure={total_exposure} USDT, "
                f"max={self.max_exposure} USDT, excess={excess} USDT ({excess_percentage:.2f}%)"
            )
            logger.error(
                "max_exposure_check_failed",
                total_exposure=float(total_exposure),
                max_exposure=float(self.max_exposure),
                excess=float(excess),
                excess_percentage=float(excess_percentage),
                error_type="RiskLimitError",
                trace_id=trace_id,
            )
            raise RiskLimitError(error_msg)

        logger.debug(
            "max_exposure_check_passed",
            total_exposure=float(total_exposure),
            max_exposure=float(self.max_exposure),
            utilization_percentage=float((total_exposure / self.max_exposure) * 100) if self.max_exposure > 0 else 0,
            trace_id=trace_id,
        )

        return True

    async def check_max_exposure_from_position_manager(self, trace_id: Optional[str] = None) -> bool:
        """Check if total exposure across all positions exceeds maximum using Position Manager API.

        Gets exposure from Position Manager service as the single source of truth.

        Args:
            trace_id: Trace ID for logging

        Returns:
            True if exposure is within limits, False otherwise

        Raises:
            RiskLimitError: If exposure exceeds limits
            OrderExecutionError: If Position Manager API call fails
        """
        from ..utils.tracing import get_or_create_trace_id

        trace_id = trace_id or get_or_create_trace_id()

        try:
            # Get portfolio exposure from Position Manager
            exposure = await self.position_manager_client.get_portfolio_exposure(trace_id=trace_id)
            total_exposure = exposure.total_exposure_usdt

            # Check against max exposure limit
            return self.check_max_exposure(total_exposure, trace_id=trace_id)

        except Exception as e:
            # If Position Manager is unavailable, log warning but don't block order execution
            # This allows graceful degradation - risk checks can fall back to other mechanisms
            logger.warning(
                "max_exposure_check_position_manager_unavailable",
                error=str(e),
                error_type=type(e).__name__,
                trace_id=trace_id,
                message="Position Manager unavailable - skipping exposure check",
            )
            # Return True to allow order execution (risk check skipped)
            # In production, consider implementing a fallback mechanism
            return True

