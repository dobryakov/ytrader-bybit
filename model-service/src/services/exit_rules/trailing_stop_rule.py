"""
Trailing stop exit rule.

Dynamic stop loss that follows price upward, locking in profits.
Activates after profit threshold is reached, then maintains distance below peak price.
"""

from typing import Optional, Dict, Any

from ...models.exit_decision import ExitDecision
from ...models.position_state_tracker import PositionState
from ...config.settings import settings
from ...config.logging import get_logger
from .base import ExitRule

logger = get_logger(__name__)


class TrailingStopRule(ExitRule):
    """
    Trailing stop exit rule.

    Activates after profit threshold is reached, then maintains a trailing distance
    below the peak price. Exits if price drops by trailing distance from peak.
    """

    def __init__(
        self,
        activation_pct: Optional[float] = None,
        distance_pct: Optional[float] = None,
        enabled: bool = True,
        priority: int = 15,  # Medium-high priority (profit protection)
    ):
        """
        Initialize trailing stop rule.

        Args:
            activation_pct: Profit percentage required to activate trailing stop (default from settings)
            distance_pct: Trailing distance percentage below peak (default from settings)
            enabled: Whether rule is enabled
            priority: Rule priority (default 15)
        """
        super().__init__(enabled=enabled, priority=priority)
        self.activation_pct = activation_pct or settings.trailing_stop_activation_pct
        self.distance_pct = distance_pct or settings.trailing_stop_distance_pct

    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate trailing stop rule.

        Args:
            position_data: Position data from update event
            position_state: Position state tracking (required for peak price)

        Returns:
            ExitDecision if trailing stop triggered, None otherwise
        """
        if not self.enabled:
            return None

        # Trailing stop requires position state tracking
        if position_state is None:
            logger.debug("Trailing stop rule: position state not available", asset=position_data.get("asset"))
            return None

        # Extract unrealized PnL percentage
        unrealized_pnl_pct = position_data.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is None:
            logger.debug("Trailing stop rule: missing unrealized_pnl_pct", asset=position_data.get("asset"))
            return None

        unrealized_pnl_pct = float(unrealized_pnl_pct)

        # Check if activation threshold reached
        if unrealized_pnl_pct < self.activation_pct:
            # Not activated yet - update highest PnL but don't exit
            position_state.update_highest_pnl(unrealized_pnl_pct)
            return None

        # Trailing stop is active - check if price dropped by trailing distance from peak
        # Calculate current price from entry price and PnL
        entry_price = position_state.entry_price
        current_price = entry_price * (1 + unrealized_pnl_pct / 100.0)

        # Update peak price if current price is higher
        peak_updated = position_state.update_peak_price(current_price)
        if peak_updated:
            position_state.update_highest_pnl(unrealized_pnl_pct)
            logger.debug(
                "Trailing stop: peak price updated",
                asset=position_data.get("asset"),
                peak_price=position_state.peak_price,
                current_price=current_price,
            )
            return None  # Price moved up - no exit

        # Calculate trailing stop price (peak price - distance)
        trailing_stop_price = position_state.peak_price * (1 - self.distance_pct / 100.0)

        # Check if current price dropped below trailing stop
        if current_price >= trailing_stop_price:
            return None  # Still above trailing stop

        # Trailing stop triggered
        position_size = position_data.get("size")
        if position_size is None:
            logger.warning("Trailing stop rule: missing position size", asset=position_data.get("asset"))
            return None

        position_size = abs(float(position_size))

        logger.info(
            "Trailing stop rule triggered",
            asset=position_data.get("asset"),
            current_price=current_price,
            peak_price=position_state.peak_price,
            trailing_stop_price=trailing_stop_price,
            distance_pct=self.distance_pct,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )

        return ExitDecision(
            should_exit=True,
            exit_reason=(
                f"Trailing stop triggered: price {current_price:.2f} dropped below "
                f"trailing stop {trailing_stop_price:.2f} (peak: {position_state.peak_price:.2f}, "
                f"distance: {self.distance_pct}%)"
            ),
            exit_amount=position_size,  # Always full exit for trailing stop
            priority=self.priority,
            rule_triggered="trailing_stop",
            metadata={
                "current_price": current_price,
                "peak_price": position_state.peak_price,
                "trailing_stop_price": trailing_stop_price,
                "distance_pct": self.distance_pct,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "activation_pct": self.activation_pct,
            },
        )

