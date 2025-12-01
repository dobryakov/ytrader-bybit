"""
Integration tests for PositionBasedSignalGenerator.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.position_based_signal_generator import position_based_signal_generator
from src.models.signal import TradingSignal


@pytest.mark.asyncio
async def test_evaluate_position_exit_disabled():
    """Test that exit evaluation is skipped when disabled."""
    original_enabled = position_based_signal_generator._enabled
    position_based_signal_generator._enabled = False

    try:
        position_data = {
            "asset": "BTCUSDT",
            "size": 1000.0,
            "unrealized_pnl_pct": 5.0,
        }

        signal = await position_based_signal_generator.evaluate_position_exit(
            position_data=position_data,
            strategy_id="test_strategy",
        )

        assert signal is None
    finally:
        position_based_signal_generator._enabled = original_enabled


@pytest.mark.asyncio
async def test_evaluate_position_exit_with_missing_asset():
    """Test that exit evaluation handles missing asset gracefully."""
    position_data = {
        "size": 1000.0,
        "unrealized_pnl_pct": 5.0,
        # Missing asset field
    }

    signal = await position_based_signal_generator.evaluate_position_exit(
        position_data=position_data,
        strategy_id="test_strategy",
    )

    assert signal is None


@pytest.mark.asyncio
async def test_debouncing():
    """Test that debouncing prevents excessive evaluation."""
    position_based_signal_generator._debounce_windows.clear()

    position_data = {
        "asset": "BTCUSDT",
        "size": 1000.0,
        "unrealized_pnl_pct": 5.0,
    }

    # First evaluation should proceed
    should_eval1 = await position_based_signal_generator._should_evaluate("BTCUSDT")
    assert should_eval1 is True

    # Immediate second evaluation should be debounced
    should_eval2 = await position_based_signal_generator._should_evaluate("BTCUSDT")
    assert should_eval2 is False


@pytest.mark.asyncio
async def test_get_metrics():
    """Test getting metrics from signal generator."""
    metrics = position_based_signal_generator.get_metrics()

    assert "exit_signals_generated" in metrics
    assert "rules_triggered" in metrics
    assert "rate_limiting_events" in metrics
    assert "evaluation_count" in metrics
    assert "evaluation_latency_p95_ms" in metrics
    assert "errors" in metrics

    assert isinstance(metrics["exit_signals_generated"], int)
    assert isinstance(metrics["rules_triggered"], dict)
    assert isinstance(metrics["evaluation_latency_p95_ms"], (int, float))

