"""
Unit tests for ExitDecision data model.
"""

import pytest
from datetime import datetime
from src.models.exit_decision import ExitDecision


def test_exit_decision_creation():
    """Test creating an ExitDecision."""
    decision = ExitDecision(
        should_exit=True,
        exit_reason="Take profit triggered: 3.5% > 3.0%",
        exit_amount=1000.0,
        priority=10,
        rule_triggered="take_profit",
        metadata={"unrealized_pnl_pct": 3.5, "threshold_pct": 3.0},
    )

    assert decision.should_exit is True
    assert decision.exit_reason == "Take profit triggered: 3.5% > 3.0%"
    assert decision.exit_amount == 1000.0
    assert decision.priority == 10
    assert decision.rule_triggered == "take_profit"
    assert decision.metadata == {"unrealized_pnl_pct": 3.5, "threshold_pct": 3.0}


def test_exit_decision_to_dict():
    """Test converting ExitDecision to dictionary."""
    decision = ExitDecision(
        should_exit=True,
        exit_reason="Stop loss triggered",
        exit_amount=500.0,
        priority=20,
        rule_triggered="stop_loss",
        metadata={"unrealized_pnl_pct": -2.5},
    )

    result = decision.to_dict()

    assert result["should_exit"] is True
    assert result["exit_reason"] == "Stop loss triggered"
    assert result["exit_amount"] == 500.0
    assert result["priority"] == 20
    assert result["rule_triggered"] == "stop_loss"
    assert result["metadata"] == {"unrealized_pnl_pct": -2.5}


def test_exit_decision_without_metadata():
    """Test ExitDecision without metadata."""
    decision = ExitDecision(
        should_exit=False,
        exit_reason="No exit conditions met",
        exit_amount=1.0,  # Must be > 0, even if should_exit=False
        priority=0,
        rule_triggered="none",
    )

    assert decision.metadata is None


def test_exit_decision_validation():
    """Test ExitDecision validation (exit_amount must be > 0)."""
    with pytest.raises(Exception):  # pydantic validation error
        ExitDecision(
            should_exit=True,
            exit_reason="Test",
            exit_amount=-100.0,  # Invalid: negative amount
            priority=10,
            rule_triggered="test",
        )

