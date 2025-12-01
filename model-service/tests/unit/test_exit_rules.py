"""
Unit tests for exit rules.
"""

import pytest
from datetime import datetime, timedelta
from src.models.exit_decision import ExitDecision
from src.models.position_state_tracker import PositionState
from src.services.exit_rules.take_profit_rule import TakeProfitRule
from src.services.exit_rules.stop_loss_rule import StopLossRule
from src.services.exit_rules.trailing_stop_rule import TrailingStopRule
from src.services.exit_rules.time_based_exit_rule import TimeBasedExitRule


@pytest.mark.asyncio
async def test_take_profit_rule_triggered():
    """Test take profit rule triggers when threshold exceeded."""
    rule = TakeProfitRule(threshold_pct=3.0, enabled=True)

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 3.5,  # Above threshold
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is not None
    assert decision.should_exit is True
    assert decision.rule_triggered == "take_profit"
    assert decision.exit_amount == 1000.0
    assert "3.5" in decision.exit_reason
    assert "3.0" in decision.exit_reason


@pytest.mark.asyncio
async def test_take_profit_rule_not_triggered():
    """Test take profit rule doesn't trigger when below threshold."""
    rule = TakeProfitRule(threshold_pct=3.0, enabled=True)

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,  # Below threshold
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is None


@pytest.mark.asyncio
async def test_take_profit_rule_partial_exit():
    """Test take profit rule with partial exit."""
    rule = TakeProfitRule(
        threshold_pct=3.0,
        partial_exit=True,
        partial_amount_pct=50.0,
        enabled=True,
    )

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 3.5,
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is not None
    assert decision.exit_amount == 500.0  # 50% of 1000.0
    assert "partial" in decision.exit_reason.lower()


@pytest.mark.asyncio
async def test_stop_loss_rule_triggered():
    """Test stop loss rule triggers when loss threshold exceeded."""
    rule = StopLossRule(threshold_pct=-2.0, enabled=True)

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": -2.5,  # Below threshold (more negative)
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is not None
    assert decision.should_exit is True
    assert decision.rule_triggered == "stop_loss"
    assert decision.exit_amount == 1000.0
    assert "-2.5" in decision.exit_reason


@pytest.mark.asyncio
async def test_stop_loss_rule_not_triggered():
    """Test stop loss rule doesn't trigger when above threshold."""
    rule = StopLossRule(threshold_pct=-2.0, enabled=True)

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": -1.5,  # Above threshold (less negative)
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is None


@pytest.mark.asyncio
async def test_trailing_stop_rule_activation():
    """Test trailing stop rule activates after profit threshold."""
    rule = TrailingStopRule(activation_pct=2.0, distance_pct=1.0, enabled=True)

    entry_time = datetime.utcnow()
    position_state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    # First, profit reaches activation threshold
    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,  # Above activation
    }

    decision = await rule.evaluate(position_data, position_state)

    # Should not exit yet (price is rising)
    assert decision is None
    assert position_state.peak_price >= 51000.0  # Peak updated


@pytest.mark.asyncio
async def test_trailing_stop_rule_triggered():
    """Test trailing stop rule triggers when price drops."""
    rule = TrailingStopRule(activation_pct=2.0, distance_pct=1.0, enabled=True)

    entry_time = datetime.utcnow()
    entry_price = 50000.0
    # Set peak price manually to 51500 (3% profit from 50000)
    # This simulates a position that reached 3% profit and is now dropping
    position_state = PositionState(
        asset="BTCUSDT",
        entry_price=entry_price,
        entry_time=entry_time,
        peak_price=51500.0,  # Peak at 3% profit (50000 * 1.03)
        highest_unrealized_pnl=3.0,
    )

    # First, activate trailing stop with profit above activation threshold
    # This ensures trailing stop is active
    position_data_activate = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 2.5,  # Above activation (2.0%)
    }
    await rule.evaluate(position_data_activate, position_state)
    
    # Manually set peak back to 51500 and highest PnL to ensure trailing stop is active
    position_state.peak_price = 51500.0
    position_state.highest_unrealized_pnl = 3.0
    
    # Now price drops below trailing stop
    # Trailing stop price = 51500 * (1 - 0.01) = 50985
    # Current price at 1.0% = 50000 * 1.01 = 50500
    # 50500 < 50985, so trailing stop should trigger
    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 1.0,  # Well below trailing stop threshold
    }

    decision = await rule.evaluate(position_data, position_state)

    assert decision is not None, f"Trailing stop should trigger. Peak: {position_state.peak_price}, Current PnL: 1.5%"
    assert decision.rule_triggered == "trailing_stop"
    assert decision.exit_amount == 1000.0


@pytest.mark.asyncio
async def test_time_based_exit_max_hours():
    """Test time-based exit triggers at max holding time."""
    rule = TimeBasedExitRule(max_hours=24, enabled=True)

    entry_time = datetime.utcnow() - timedelta(hours=25)  # Held for 25 hours
    position_state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 1.0,
    }

    decision = await rule.evaluate(position_data, position_state)

    assert decision is not None
    assert decision.rule_triggered == "time_based_max_time"
    assert "maximum holding time" in decision.exit_reason.lower()


@pytest.mark.asyncio
async def test_time_based_exit_profit_target():
    """Test time-based exit with profit target."""
    rule = TimeBasedExitRule(
        max_hours=24,
        profit_target_pct=1.0,
        enabled=True,
    )

    entry_time = datetime.utcnow() - timedelta(hours=2)  # Held for 2 hours
    position_state = PositionState(
        asset="BTCUSDT",
        entry_price=50000.0,
        entry_time=entry_time,
        peak_price=51000.0,
        highest_unrealized_pnl=2.0,
    )

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 1.5,  # Above profit target
    }

    decision = await rule.evaluate(position_data, position_state)

    assert decision is not None
    assert decision.rule_triggered == "time_based_profit_target"
    assert "profit target" in decision.exit_reason.lower()


@pytest.mark.asyncio
async def test_exit_rule_disabled():
    """Test that disabled rules don't trigger."""
    rule = TakeProfitRule(threshold_pct=3.0, enabled=False)

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 5.0,  # Well above threshold
    }

    decision = await rule.evaluate(position_data, None)

    assert decision is None

