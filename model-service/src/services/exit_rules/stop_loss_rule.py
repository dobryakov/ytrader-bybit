"""
Stop loss exit rule.

Exits position when unrealized loss exceeds configured threshold.
"""

from typing import Optional, Dict, Any

from ...models.exit_decision import ExitDecision
from ...models.position_state_tracker import PositionState
from ...config.settings import settings
from ...config.logging import get_logger
from .base import ExitRule

logger = get_logger(__name__)


class StopLossRule(ExitRule):
    """
    Stop loss exit rule.

    Generates exit signal when unrealized loss percentage exceeds threshold.
    """

    def __init__(
        self,
        threshold_pct: Optional[float] = None,
        enabled: bool = True,
        priority: int = 20,  # Higher priority than take profit (loss protection is critical)
    ):
        """
        Initialize stop loss rule.

        Args:
            threshold_pct: Loss threshold percentage (negative value, default from settings)
            enabled: Whether rule is enabled
            priority: Rule priority (default 20 - highest priority for loss protection)
        """
        super().__init__(enabled=enabled, priority=priority)
        self.threshold_pct = threshold_pct or settings.stop_loss_threshold_pct

        # Ensure threshold is negative (loss)
        if self.threshold_pct > 0:
            self.threshold_pct = -abs(self.threshold_pct)
            logger.warning(
                "Stop loss threshold was positive, converted to negative",
                original=threshold_pct,
                converted=self.threshold_pct,
            )

    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate stop loss rule.

        Args:
            position_data: Position data from update event
            position_state: Optional position state tracking

        Returns:
            ExitDecision if stop loss threshold exceeded, None otherwise
        """
        if not self.enabled:
            return None

        # Extract unrealized PnL percentage
        unrealized_pnl_pct = position_data.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is None:
            logger.debug("Stop loss rule: missing unrealized_pnl_pct", asset=position_data.get("asset"))
            return None

        unrealized_pnl_pct = float(unrealized_pnl_pct)

        # Check if loss threshold exceeded (unrealized_pnl_pct is more negative than threshold)
        if unrealized_pnl_pct >= self.threshold_pct:
            return None

        # Get position size
        position_size = position_data.get("size")
        if position_size is None:
            logger.warning("Stop loss rule: missing position size", asset=position_data.get("asset"))
            return None

        position_size = abs(float(position_size))

        logger.warning(
            "Stop loss rule triggered",
            asset=position_data.get("asset"),
            unrealized_pnl_pct=unrealized_pnl_pct,
            threshold_pct=self.threshold_pct,
            position_size=position_size,
        )

        return ExitDecision(
            should_exit=True,
            exit_reason=f"Stop loss triggered: {unrealized_pnl_pct:.2f}% < {self.threshold_pct:.2f}%",
            exit_amount=position_size,  # Always full exit for stop loss
            priority=self.priority,
            rule_triggered="stop_loss",
            metadata={
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "threshold_pct": self.threshold_pct,
            },
        )

