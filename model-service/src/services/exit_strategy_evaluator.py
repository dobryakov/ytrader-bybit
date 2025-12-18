"""
Exit strategy evaluator service.

Evaluates all active exit rules and determines if position should be exited.
Applies rules in priority order and returns the highest priority exit decision.
"""

from typing import List, Optional, Dict, Any

from ..models.exit_decision import ExitDecision
from ..models.position_state_tracker import PositionState
from ..config.logging import get_logger
from .exit_rules.base import ExitRule
from .exit_rules.take_profit_rule import TakeProfitRule
from .exit_rules.stop_loss_rule import StopLossRule
from .exit_rules.time_based_exit_rule import TimeBasedExitRule
from ..config.settings import settings

logger = get_logger(__name__)


class ExitStrategyEvaluator:
    """
    Evaluates exit strategies for positions.

    Applies multiple exit rules and determines if position should be closed.
    Rules are evaluated in priority order (higher priority = evaluated first).
    """

    def __init__(self):
        """Initialize exit strategy evaluator with configured rules."""
        self.rules: List[ExitRule] = []

        # Initialize exit rules based on configuration
        if settings.take_profit_enabled:
            self.rules.append(
                TakeProfitRule(
                    threshold_pct=settings.model_service_take_profit_pct,  # Use unified take profit threshold
                    partial_exit=settings.take_profit_partial_exit,
                    partial_amount_pct=settings.take_profit_partial_amount_pct,
                    enabled=settings.take_profit_enabled,
                )
            )

        if settings.stop_loss_enabled:
            self.rules.append(
                StopLossRule(
                    threshold_pct=settings.stop_loss_threshold_pct,
                    enabled=settings.stop_loss_enabled,
                )
            )

        if settings.time_based_exit_enabled:
            self.rules.append(
                TimeBasedExitRule(
                    max_hours=settings.time_based_exit_max_hours,
                    profit_target_pct=settings.time_based_exit_profit_target_pct,
                    enabled=settings.time_based_exit_enabled,
                )
            )

        # Sort rules by priority (higher priority first)
        self.rules.sort(key=lambda r: r.get_priority(), reverse=True)

        logger.info(
            "Exit strategy evaluator initialized",
            enabled_rules_count=len([r for r in self.rules if r.is_enabled()]),
            total_rules_count=len(self.rules),
            rules=[r.__class__.__name__ for r in self.rules],
        )

    async def evaluate(
        self,
        position_data: Dict[str, Any],
        position_state: Optional[PositionState],
    ) -> Optional[ExitDecision]:
        """
        Evaluate exit strategies for a position.

        Evaluates all enabled rules in priority order and returns the first
        exit decision found (highest priority rule that triggers).

        Args:
            position_data: Position data from update event
                Expected fields: asset, size, unrealized_pnl, unrealized_pnl_pct, etc.
            position_state: Optional PositionState tracking entry price, peak price, etc.

        Returns:
            ExitDecision if any rule triggers, None otherwise
        """
        asset = position_data.get("asset", "unknown")

        logger.debug(
            "Evaluating exit strategies",
            asset=asset,
            rules_count=len(self.rules),
            enabled_rules_count=len(self.get_enabled_rules()),
            has_position_state=position_state is not None,
            unrealized_pnl_pct=position_data.get("unrealized_pnl_pct"),
        )

        # Evaluate rules in priority order
        evaluated_rules = 0
        for rule in self.rules:
            if not rule.is_enabled():
                logger.debug("Rule disabled, skipping", asset=asset, rule=rule.__class__.__name__)
                continue

            evaluated_rules += 1
            try:
                logger.debug(
                    "Evaluating exit rule",
                    asset=asset,
                    rule=rule.__class__.__name__,
                    priority=rule.get_priority(),
                )
                decision = await rule.evaluate(position_data, position_state)
                if decision:
                    logger.info(
                        "Exit rule triggered",
                        asset=asset,
                        rule=rule.__class__.__name__,
                        exit_reason=decision.exit_reason,
                        exit_amount=decision.exit_amount,
                        priority=decision.priority,
                        rule_triggered=decision.rule_triggered,
                        metadata=decision.metadata,
                    )
                    return decision
                else:
                    logger.debug("Exit rule not triggered", asset=asset, rule=rule.__class__.__name__)
            except Exception as e:
                logger.error(
                    "Error evaluating exit rule",
                    asset=asset,
                    rule=rule.__class__.__name__,
                    error=str(e),
                    exc_info=True,
                )
                # Continue with other rules even if one fails
                continue

        logger.debug(
            "No exit rules triggered",
            asset=asset,
            evaluated_rules=evaluated_rules,
            total_rules=len(self.rules),
        )
        return None

    def get_rules(self) -> List[ExitRule]:
        """Get list of configured exit rules."""
        return self.rules.copy()

    def get_enabled_rules(self) -> List[ExitRule]:
        """Get list of enabled exit rules."""
        return [r for r in self.rules if r.is_enabled()]


# Global exit strategy evaluator instance
exit_strategy_evaluator = ExitStrategyEvaluator()

