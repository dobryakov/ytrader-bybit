"""
Exit decision data model.

Represents the result of exit strategy evaluation, indicating whether a position
should be exited and providing details about the exit decision.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ExitDecision(BaseModel):
    """
    Exit decision data model.

    Represents the result of evaluating exit rules for a position,
    indicating whether the position should be exited and providing
    details about the decision.
    """

    should_exit: bool = Field(..., description="Whether position should be exited")
    exit_reason: str = Field(..., description="Reason for exit decision")
    exit_amount: float = Field(..., description="Amount to exit (partial or full position size)", gt=0)
    priority: int = Field(..., description="Priority of exit decision (higher = more urgent)", ge=0)
    rule_triggered: str = Field(..., description="Name of the exit rule that triggered this decision")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata about the exit decision (thresholds, values, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exit decision to dictionary."""
        return {
            "should_exit": self.should_exit,
            "exit_reason": self.exit_reason,
            "exit_amount": self.exit_amount,
            "priority": self.priority,
            "rule_triggered": self.rule_triggered,
            "metadata": self.metadata,
        }

