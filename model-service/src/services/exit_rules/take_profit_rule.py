"""
Take profit exit rule.

Exits position when unrealized profit exceeds configured threshold.
Supports partial exit (e.g., close 50% at 3% profit, rest at 5%).
"""

from typing import Optional, Dict, Any
from decimal import Decimal

from ...models.exit_decision import ExitDecision
from ...models.position_state_tracker import PositionState
from ...config.settings import settings
from ...config.logging import get_logger
from .base import ExitRule

logger = get_logger(__name__)


class TakeProfitRule(ExitRule):
    """
    Take profit exit rule.

    Generates exit signal when unrealized PnL percentage exceeds threshold.
    Supports partial exit configuration.
    """

    def __init__(
        self,
        threshold_pct: Optional[float] = None,
        partial_exit: bool = False,
        partial_amount_pct: Optional[float] = None,
        enabled: bool = True,
        priority: int = 10,
    ):
        """
        Initialize take profit rule.

        Args:
            threshold_pct: Profit threshold percentage (default from settings)
            partial_exit: Whether to support partial exit
            partial_amount_pct: Percentage of position to close on first threshold (default 50%)
            enabled: Whether rule is enabled
            priority: Rule priority (default 10 - high priority for profit protection)
        """
        super().__init__(enabled=enabled, priority=priority)
        # Use unified take profit threshold (MODEL_SERVICE_TAKE_PROFIT_PCT)
        # If threshold_pct is explicitly provided, use it; otherwise use unified setting
        if threshold_pct is not None:
            self.threshold_pct = threshold_pct
        else:
            # Use unified MODEL_SERVICE_TAKE_PROFIT_PCT (default 3.0 if not set)
            self.threshold_pct = getattr(settings, 'model_service_take_profit_pct', 3.0)
        self.partial_exit = partial_exit or settings.take_profit_partial_exit
        self.partial_amount_pct = partial_amount_pct or settings.take_profit_partial_amount_pct
        self._partial_exit_triggered = False  # Track if partial exit was already triggered

    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate take profit rule.

        Args:
            position_data: Position data from update event
            position_state: Optional position state tracking

        Returns:
            ExitDecision if take profit threshold exceeded, None otherwise
        """
        if not self.enabled:
            return None

        # Extract unrealized PnL percentage
        unrealized_pnl_pct = position_data.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is None:
            logger.debug("Take profit rule: missing unrealized_pnl_pct", asset=position_data.get("asset"))
            return None

        unrealized_pnl_pct = float(unrealized_pnl_pct)

        # Check if threshold exceeded
        if unrealized_pnl_pct <= self.threshold_pct:
            return None

        # Get position size
        position_size = position_data.get("size")
        if position_size is None:
            logger.warning("Take profit rule: missing position size", asset=position_data.get("asset"))
            return None

        position_size = abs(float(position_size))

        # Determine exit amount
        if self.partial_exit and not self._partial_exit_triggered:
            # First threshold hit - partial exit
            exit_amount = position_size * (self.partial_amount_pct / 100.0)
            self._partial_exit_triggered = True
            exit_reason = f"Take profit partial exit: {unrealized_pnl_pct:.2f}% > {self.threshold_pct:.2f}% (closing {self.partial_amount_pct}%)"
        else:
            # Full exit (either partial not enabled, or second threshold)
            exit_amount = position_size
            exit_reason = f"Take profit triggered: {unrealized_pnl_pct:.2f}% > {self.threshold_pct:.2f}%"

        logger.info(
            "Take profit rule triggered",
            asset=position_data.get("asset"),
            unrealized_pnl_pct=unrealized_pnl_pct,
            threshold_pct=self.threshold_pct,
            exit_amount=exit_amount,
            partial_exit=self.partial_exit,
        )

        return ExitDecision(
            should_exit=True,
            exit_reason=exit_reason,
            exit_amount=exit_amount,
            priority=self.priority,
            rule_triggered="take_profit",
            metadata={
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "threshold_pct": self.threshold_pct,
                "partial_exit": self.partial_exit,
                "partial_exit_triggered": self._partial_exit_triggered,
            },
        )

    def reset_partial_exit(self) -> None:
        """Reset partial exit flag (call when position is closed)."""
        self._partial_exit_triggered = False

