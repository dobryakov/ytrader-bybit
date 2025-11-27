"""Risk manager service for enforcing risk limits and safety checks."""

from decimal import Decimal
from typing import Optional

from ..config.settings import settings
from ..config.logging import get_logger
from ..models.trading_signal import TradingSignal
from ..models.position import Position
from ..utils.bybit_client import get_bybit_client
from ..exceptions import RiskLimitError, OrderExecutionError

logger = get_logger(__name__)


class RiskManager:
    """Service for enforcing risk limits and performing safety checks."""

    def __init__(self):
        """Initialize risk manager with configuration."""
        self.max_position_size = Decimal(str(settings.order_manager_max_position_size))
        self.max_exposure = Decimal(str(settings.order_manager_max_exposure))
        self.max_order_size_ratio = Decimal(str(settings.order_manager_max_order_size_ratio))

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
            result = response.get("result", {})
            list_data = result.get("list", [])

            if not list_data:
                raise OrderExecutionError("Failed to retrieve account balance from Bybit")

            # Find USDT balance (quote currency)
            account = list_data[0]
            coins = account.get("coin", [])

            usdt_balance = Decimal("0")
            for coin in coins:
                if coin.get("coin") == "USDT":
                    usdt_balance = Decimal(str(coin.get("availableToWithdraw", "0")))
                    break

            # Calculate required balance
            if signal.signal_type.lower() == "buy":
                required_balance = order_quantity * order_price
            else:
                # For sell orders, check if we have the asset
                # This is a simplified check - in reality, we'd check base currency balance
                required_balance = Decimal("0")

            # Check if balance is sufficient
            if signal.signal_type.lower() == "buy" and required_balance > usdt_balance:
                error_msg = (
                    f"Insufficient balance: required={required_balance}, "
                    f"available={usdt_balance}"
                )
                logger.error(
                    "balance_check_failed",
                    signal_id=str(signal.signal_id),
                    required=float(required_balance),
                    available=float(usdt_balance),
                    trace_id=trace_id,
                )
                raise RiskLimitError(error_msg)

            logger.info(
                "balance_check_passed",
                signal_id=str(signal.signal_id),
                required=float(required_balance) if signal.signal_type.lower() == "buy" else 0,
                available=float(usdt_balance),
                trace_id=trace_id,
            )

            return True

        except RiskLimitError:
            raise
        except Exception as e:
            logger.error("balance_check_error", signal_id=str(signal.signal_id), error=str(e), trace_id=trace_id)
            raise OrderExecutionError(f"Balance check failed: {e}") from e

    def check_order_size(self, signal: TradingSignal, order_quantity: Decimal, order_price: Decimal) -> bool:
        """Check if order size exceeds maximum order size limit.

        Args:
            signal: Trading signal
            order_quantity: Order quantity in base currency
            order_price: Order price
            trace_id: Trace ID for logging

        Returns:
            True if order size is within limits, False otherwise

        Raises:
            RiskLimitError: If order size exceeds limits
        """
        trace_id = signal.trace_id
        order_value = order_quantity * order_price

        # Check against max order size ratio of available balance
        # This is a simplified check - in reality, we'd check against actual balance
        if order_value > self.max_exposure * self.max_order_size_ratio:
            error_msg = (
                f"Order size exceeds limit: order_value={order_value}, "
                f"max_ratio={self.max_order_size_ratio}, max_exposure={self.max_exposure}"
            )
            logger.error(
                "order_size_check_failed",
                signal_id=str(signal.signal_id),
                order_value=float(order_value),
                max_ratio=float(self.max_order_size_ratio),
                trace_id=trace_id,
            )
            raise RiskLimitError(error_msg)

        logger.debug(
            "order_size_check_passed",
            signal_id=str(signal.signal_id),
            order_value=float(order_value),
            trace_id=trace_id,
        )

        return True

    def check_position_size(self, asset: str, current_position: Optional[Position], order_quantity: Decimal, order_side: str) -> bool:
        """Check if order would exceed maximum position size limit.

        Args:
            asset: Trading pair symbol
            current_position: Current position for asset (None if no position)
            order_quantity: Order quantity in base currency
            order_side: Order side ('Buy' or 'Sell')
            trace_id: Trace ID for logging

        Returns:
            True if position size is within limits, False otherwise

        Raises:
            RiskLimitError: If position size would exceed limits
        """
        trace_id = None  # TODO: Get from context

        current_size = Decimal("0")
        if current_position:
            current_size = current_position.size

        # Calculate new position size
        if order_side.upper() == "BUY":
            new_size = current_size + order_quantity
        else:  # SELL
            new_size = current_size - order_quantity

        # Check against max position size
        if abs(new_size) > self.max_position_size:
            error_msg = (
                f"Position size would exceed limit: current={current_size}, "
                f"order={order_quantity}, new={new_size}, max={self.max_position_size}"
            )
            logger.error(
                "position_size_check_failed",
                asset=asset,
                current_size=float(current_size),
                order_quantity=float(order_quantity),
                new_size=float(new_size),
                max_size=float(self.max_position_size),
                trace_id=trace_id,
            )
            raise RiskLimitError(error_msg)

        logger.debug(
            "position_size_check_passed",
            asset=asset,
            current_size=float(current_size),
            new_size=float(new_size),
            trace_id=trace_id,
        )

        return True

    def check_max_exposure(self, total_exposure: Decimal) -> bool:
        """Check if total exposure across all positions exceeds maximum.

        Args:
            total_exposure: Total exposure across all positions (in USDT)

        Returns:
            True if exposure is within limits, False otherwise

        Raises:
            RiskLimitError: If exposure exceeds limits
        """
        if total_exposure > self.max_exposure:
            error_msg = (
                f"Total exposure exceeds limit: exposure={total_exposure}, "
                f"max={self.max_exposure}"
            )
            logger.error(
                "max_exposure_check_failed",
                total_exposure=float(total_exposure),
                max_exposure=float(self.max_exposure),
            )
            raise RiskLimitError(error_msg)

        logger.debug(
            "max_exposure_check_passed",
            total_exposure=float(total_exposure),
            max_exposure=float(self.max_exposure),
        )

        return True

