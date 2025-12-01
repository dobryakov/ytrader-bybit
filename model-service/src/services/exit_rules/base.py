"""
Base exit rule abstract class.

Defines the interface for all exit rules used in position-based exit strategy evaluation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ...models.exit_decision import ExitDecision
from ...models.position_state_tracker import PositionState


class ExitRule(ABC):
    """
    Abstract base class for exit rules.

    All exit rules must implement the evaluate method which takes position data
    and position state and returns an ExitDecision if the rule is triggered.
    """

    def __init__(self, enabled: bool = True, priority: int = 0):
        """
        Initialize exit rule.

        Args:
            enabled: Whether this rule is enabled
            priority: Priority of this rule (higher = more urgent, evaluated first)
        """
        self.enabled = enabled
        self.priority = priority

    @abstractmethod
    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate exit rule against position data and state.

        Args:
            position_data: Dictionary with position data from position update event
                Expected fields: asset, size, unrealized_pnl, unrealized_pnl_pct, etc.
            position_state: Optional PositionState tracking entry price, peak price, etc.

        Returns:
            ExitDecision if rule is triggered, None otherwise
        """
        pass

    def is_enabled(self) -> bool:
        """Check if rule is enabled."""
        return self.enabled

    def get_priority(self) -> int:
        """Get rule priority."""
        return self.priority

