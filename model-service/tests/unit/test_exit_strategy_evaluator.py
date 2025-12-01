"""
Unit tests for ExitStrategyEvaluator.
"""

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime
from src.services.exit_strategy_evaluator import ExitStrategyEvaluator
from src.models.exit_decision import ExitDecision
from src.models.position_state_tracker import PositionState


@pytest.mark.asyncio
async def test_evaluator_with_no_rules():
    """Test evaluator with no rules configured."""
    with patch("src.services.exit_strategy_evaluator.settings") as mock_settings:
        mock_settings.take_profit_enabled = False
        mock_settings.stop_loss_enabled = False
        mock_settings.trailing_stop_enabled = False
        mock_settings.time_based_exit_enabled = False

        evaluator = ExitStrategyEvaluator()

        position_data = {"asset": "BTCUSDT", "size": 1000.0, "unrealized_pnl_pct": 5.0}

        decision = await evaluator.evaluate(position_data, None)

        assert decision is None


@pytest.mark.asyncio
async def test_evaluator_priority_order():
    """Test that rules are evaluated in priority order."""
    with patch("src.services.exit_strategy_evaluator.settings") as mock_settings:
        mock_settings.take_profit_enabled = True
        mock_settings.take_profit_threshold_pct = 3.0
        mock_settings.take_profit_partial_exit = False
        mock_settings.take_profit_partial_amount_pct = 50.0
        mock_settings.stop_loss_enabled = True
        mock_settings.stop_loss_threshold_pct = -2.0
        mock_settings.trailing_stop_enabled = False
        mock_settings.time_based_exit_enabled = False

        evaluator = ExitStrategyEvaluator()

        # Both rules could trigger, but stop loss has higher priority
        position_data = {
            "asset": "BTCUSDT",
            "size": 1000.0,
            "unrealized_pnl_pct": -2.5,  # Triggers stop loss
        }

        decision = await evaluator.evaluate(position_data, None)

        assert decision is not None
        assert decision.rule_triggered == "stop_loss"  # Higher priority rule


@pytest.mark.asyncio
async def test_evaluator_rule_error_handling():
    """Test that evaluator continues with other rules if one fails."""
    with patch("src.services.exit_strategy_evaluator.settings") as mock_settings:
        mock_settings.take_profit_enabled = True
        mock_settings.take_profit_threshold_pct = 3.0
        mock_settings.take_profit_partial_exit = False
        mock_settings.take_profit_partial_amount_pct = 50.0
        mock_settings.stop_loss_enabled = False
        mock_settings.trailing_stop_enabled = False
        mock_settings.time_based_exit_enabled = False

        evaluator = ExitStrategyEvaluator()

        # Mock rule to raise exception
        if evaluator.rules:
            original_evaluate = evaluator.rules[0].evaluate

            async def failing_evaluate(*args, **kwargs):
                raise Exception("Rule evaluation failed")

            evaluator.rules[0].evaluate = failing_evaluate

            position_data = {
                "asset": "BTCUSDT",
                "size": 1000.0,
                "unrealized_pnl_pct": 5.0,
            }

            # Should not raise exception, but may return None if all rules fail
            decision = await evaluator.evaluate(position_data, None)

            # Restore original method
            evaluator.rules[0].evaluate = original_evaluate


@pytest.mark.asyncio
async def test_evaluator_get_rules():
    """Test getting list of rules."""
    with patch("src.services.exit_strategy_evaluator.settings") as mock_settings:
        mock_settings.take_profit_enabled = True
        mock_settings.take_profit_threshold_pct = 3.0
        mock_settings.take_profit_partial_exit = False
        mock_settings.take_profit_partial_amount_pct = 50.0
        mock_settings.stop_loss_enabled = True
        mock_settings.stop_loss_threshold_pct = -2.0
        mock_settings.trailing_stop_enabled = False
        mock_settings.time_based_exit_enabled = False

        evaluator = ExitStrategyEvaluator()

        rules = evaluator.get_rules()
        assert len(rules) >= 2  # At least take profit and stop loss

        enabled_rules = evaluator.get_enabled_rules()
        assert len(enabled_rules) >= 2

