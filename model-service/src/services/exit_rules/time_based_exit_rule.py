"""
Time-based exit rule.

Exits position based on holding time, with optional profit targets that decay over time.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ...models.exit_decision import ExitDecision
from ...models.position_state_tracker import PositionState
from ...config.settings import settings
from ...config.logging import get_logger
from .base import ExitRule

logger = get_logger(__name__)


class TimeBasedExitRule(ExitRule):
    """
    Time-based exit rule.

    Exits position when maximum holding time is reached, or when profit target
    is met after minimum holding time. Supports decay function for profit targets.
    """

    def __init__(
        self,
        max_hours: Optional[int] = None,
        profit_target_pct: Optional[float] = None,
        enabled: bool = True,
        priority: int = 5,  # Lower priority (time-based is less urgent)
    ):
        """
        Initialize time-based exit rule.

        Args:
            max_hours: Maximum holding time in hours (default from settings)
            profit_target_pct: Profit target percentage to exit after minimum time (default from settings)
            enabled: Whether rule is enabled
            priority: Rule priority (default 5 - lower priority)
        """
        super().__init__(enabled=enabled, priority=priority)
        self.max_hours = max_hours or settings.time_based_exit_max_hours
        self.profit_target_pct = profit_target_pct or settings.time_based_exit_profit_target_pct

    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate time-based exit rule.

        Args:
            position_data: Position data from update event
            position_state: Position state tracking (required for entry time)

        Returns:
            ExitDecision if time-based exit triggered, None otherwise
        """
        if not self.enabled:
            return None

        # Time-based exit requires position state tracking
        if position_state is None:
            logger.debug("Time-based exit rule: position state not available", asset=position_data.get("asset"))
            return None

        # Get time held
        time_held_minutes = position_state.get_time_held_minutes()
        if time_held_minutes is None:
            logger.debug("Time-based exit rule: cannot calculate time held", asset=position_data.get("asset"))
            return None

        time_held_hours = time_held_minutes / 60.0

        # Check maximum holding time
        if time_held_hours >= self.max_hours:
            position_size = position_data.get("size")
            if position_size is None:
                logger.warning("Time-based exit rule: missing position size", asset=position_data.get("asset"))
                return None

            position_size = abs(float(position_size))

            logger.info(
                "Time-based exit: maximum holding time reached",
                asset=position_data.get("asset"),
                time_held_hours=time_held_hours,
                max_hours=self.max_hours,
            )

            return ExitDecision(
                should_exit=True,
                exit_reason=f"Maximum holding time reached: {time_held_hours:.2f}h >= {self.max_hours}h",
                exit_amount=position_size,
                priority=self.priority,
                rule_triggered="time_based_max_time",
                metadata={
                    "time_held_hours": time_held_hours,
                    "max_hours": self.max_hours,
                },
            )

        # Check profit target (if configured and minimum time has passed)
        if self.profit_target_pct and time_held_hours >= 1.0:  # At least 1 hour held
            unrealized_pnl_pct = position_data.get("unrealized_pnl_pct")
            if unrealized_pnl_pct is not None:
                unrealized_pnl_pct = float(unrealized_pnl_pct)

                # Apply decay function: profit target decreases over time
                # Linear decay: target = original_target * (1 - time_held / max_hours)
                decay_factor = 1.0 - (time_held_hours / self.max_hours)
                adjusted_target = self.profit_target_pct * max(decay_factor, 0.1)  # Minimum 10% of original target

                if unrealized_pnl_pct >= adjusted_target:
                    position_size = position_data.get("size")
                    if position_size is None:
                        logger.warning(
                            "Time-based exit rule: missing position size", asset=position_data.get("asset")
                        )
                        return None

                    position_size = abs(float(position_size))

                    logger.info(
                        "Time-based exit: profit target met",
                        asset=position_data.get("asset"),
                        unrealized_pnl_pct=unrealized_pnl_pct,
                        adjusted_target=adjusted_target,
                        original_target=self.profit_target_pct,
                        time_held_hours=time_held_hours,
                    )

                    return ExitDecision(
                        should_exit=True,
                        exit_reason=(
                            f"Time-based profit target met: {unrealized_pnl_pct:.2f}% >= "
                            f"{adjusted_target:.2f}% (adjusted from {self.profit_target_pct:.2f}%, "
                            f"held {time_held_hours:.2f}h)"
                        ),
                        exit_amount=position_size,
                        priority=self.priority,
                        rule_triggered="time_based_profit_target",
                        metadata={
                            "unrealized_pnl_pct": unrealized_pnl_pct,
                            "adjusted_target": adjusted_target,
                            "original_target": self.profit_target_pct,
                            "time_held_hours": time_held_hours,
                        },
                    )

        return None

