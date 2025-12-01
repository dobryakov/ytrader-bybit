"""
Position-based signal generator service.

Generates trading signals (SELL) based on position updates in real-time.
Evaluates exit strategies and generates exit signals when rules are triggered.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from ..models.signal import TradingSignal, MarketDataSnapshot
from ..models.exit_decision import ExitDecision
from ..models.position_state_tracker import PositionState
from ..services.exit_strategy_evaluator import exit_strategy_evaluator
from ..services.position_state_tracker import position_state_tracker
from ..services.exit_signal_rate_limiter import exit_signal_rate_limiter
from ..services.model_inference import model_inference
from ..publishers.signal_publisher import signal_publisher
from ..config.settings import settings
from ..config.logging import get_logger, bind_context

logger = get_logger(__name__)


class PositionBasedSignalGenerator:
    """
    Generates trading signals based on position updates.

    Reacts to position changes in real-time and evaluates exit strategies.
    Generates SELL signals when exit rules are triggered.
    """

    def __init__(self):
        """Initialize position-based signal generator."""
        self._enabled = settings.exit_strategy_enabled
        self._debounce_windows: Dict[str, float] = {}  # Asset -> last evaluation time
        self._debounce_window_seconds = 2.0  # 2 second debounce window
        self._fallback_mode = False  # Fallback to periodic evaluation if event-driven fails
        self._fallback_task: Optional[asyncio.Task] = None

        # Metrics tracking
        self._metrics = {
            "exit_signals_generated": 0,
            "rules_triggered": defaultdict(int),  # rule_name -> count
            "rate_limiting_events": 0,
            "evaluation_count": 0,
            "evaluation_latency_ms": [],  # Track latencies for p95 calculation
            "errors": 0,
        }

    async def evaluate_position_exit(
        self,
        position_data: Dict[str, Any],
        strategy_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """
        Evaluate if position should be exited based on exit rules.

        Args:
            position_data: Position data from position update event
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            SELL signal if exit conditions met, None otherwise
        """
        if not self._enabled:
            logger.debug("Exit strategy disabled, skipping evaluation")
            return None

        bind_context(strategy_id=strategy_id, trace_id=trace_id)

        asset = position_data.get("asset")
        if not asset:
            logger.warning("Position data missing asset field", position_data_keys=list(position_data.keys()))
            return None

        # Apply debouncing to prevent excessive evaluation
        if not await self._should_evaluate(asset):
            logger.debug("Position update debounced", asset=asset, trace_id=trace_id)
            return None

        # Track evaluation start time for latency metrics
        evaluation_start = time.time()

        try:
            self._metrics["evaluation_count"] += 1

            # Get or create position state for tracking
            position_state = await position_state_tracker.get_or_create_state(asset, position_data)

            # Evaluate exit strategies
            exit_decision = await exit_strategy_evaluator.evaluate(position_data, position_state)

            # Track evaluation latency
            evaluation_latency_ms = (time.time() - evaluation_start) * 1000
            self._metrics["evaluation_latency_ms"].append(evaluation_latency_ms)
            # Keep only last 1000 latencies for memory efficiency
            if len(self._metrics["evaluation_latency_ms"]) > 1000:
                self._metrics["evaluation_latency_ms"] = self._metrics["evaluation_latency_ms"][-1000:]

            logger.debug(
                "Exit strategy evaluation completed",
                asset=asset,
                latency_ms=round(evaluation_latency_ms, 2),
                exit_decision=exit_decision is not None,
                trace_id=trace_id,
            )

            if not exit_decision or not exit_decision.should_exit:
                # Update position state even if no exit
                await position_state_tracker.update_state(asset, position_data)
                return None

            # Check rate limiting
            allowed, reason = exit_signal_rate_limiter.check_rate_limit(asset)
            if not allowed:
                self._metrics["rate_limiting_events"] += 1
                logger.warning(
                    "Exit signal rate limited",
                    asset=asset,
                    reason=reason,
                    trace_id=trace_id,
                )
                return None

            # Generate SELL signal
            signal = await self._generate_exit_signal(
                exit_decision=exit_decision,
                position_data=position_data,
                position_state=position_state,
                strategy_id=strategy_id,
                trace_id=trace_id,
            )

            if signal:
                # Mark exit signal sent in position state
                await position_state_tracker.mark_exit_signal_sent(asset)

                # Update metrics
                self._metrics["exit_signals_generated"] += 1
                self._metrics["rules_triggered"][exit_decision.rule_triggered] += 1

                logger.info(
                    "Exit signal generated",
                    asset=asset,
                    exit_reason=exit_decision.exit_reason,
                    exit_amount=exit_decision.exit_amount,
                    rule_triggered=exit_decision.rule_triggered,
                    signal_id=signal.signal_id,
                    trace_id=trace_id,
                )

            return signal

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(
                "Error evaluating position exit",
                asset=asset,
                strategy_id=strategy_id,
                error=str(e),
                exc_info=True,
                trace_id=trace_id,
            )
            return None

    async def _should_evaluate(self, asset: str) -> bool:
        """
        Check if position should be evaluated (debouncing).

        Args:
            asset: Trading pair symbol

        Returns:
            True if evaluation should proceed, False if debounced
        """
        import time

        current_time = time.time()
        last_evaluation = self._debounce_windows.get(asset, 0)

        if current_time - last_evaluation < self._debounce_window_seconds:
            return False

        self._debounce_windows[asset] = current_time
        return True

    async def _generate_exit_signal(
        self,
        exit_decision: ExitDecision,
        position_data: Dict[str, Any],
        position_state: PositionState,
        strategy_id: str,
        trace_id: Optional[str],
    ) -> Optional[TradingSignal]:
        """
        Generate SELL signal from exit decision.

        Args:
            exit_decision: Exit decision from rule evaluation
            position_data: Position data from update event
            position_state: Position state tracking
            strategy_id: Trading strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            TradingSignal (SELL) or None if generation fails
        """
        asset = position_data.get("asset")
        if not asset:
            return None

        # Get market data snapshot at signal generation time
        market_data_snapshot = model_inference.get_market_data_snapshot(asset)
        if not market_data_snapshot:
            logger.warning("Market data unavailable for exit signal", asset=asset, trace_id=trace_id)
            return None

        # Create SELL signal
        signal = TradingSignal(
            signal_type="sell",
            asset=asset,
            amount=exit_decision.exit_amount,
            confidence=1.0,  # Maximum confidence for exit rules
            strategy_id=strategy_id,
            model_version=None,  # Not from model
            is_warmup=False,
            market_data_snapshot=market_data_snapshot,
            metadata={
                "reasoning": exit_decision.exit_reason,
                "rule_triggered": exit_decision.rule_triggered,
                "exit_strategy": True,
                "exit_decision_metadata": exit_decision.metadata,
            },
            trace_id=trace_id,
        )

        return signal

    async def start_fallback_mode(self) -> None:
        """
        Start fallback periodic evaluation mode.

        Used when event-driven processing is unavailable.
        Periodically checks positions and evaluates exit strategies.
        """
        if self._fallback_mode:
            logger.warning("Fallback mode already started")
            return

        self._fallback_mode = True
        logger.warning("Starting fallback periodic evaluation mode", message="Event-driven processing unavailable")

        # Fallback mode would periodically check positions
        # This is a placeholder - full implementation would query Position Manager API
        # and evaluate exit strategies for all open positions
        logger.info("Fallback mode: periodic evaluation not yet implemented")

    async def stop_fallback_mode(self) -> None:
        """Stop fallback periodic evaluation mode."""
        if not self._fallback_mode:
            return

        self._fallback_mode = False
        if self._fallback_task:
            self._fallback_task.cancel()
            try:
                await self._fallback_task
            except asyncio.CancelledError:
                pass
            self._fallback_task = None

        logger.info("Fallback mode stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for exit strategy operations.

        Returns:
            Dictionary with metrics:
            - exit_signals_generated: Total exit signals generated
            - rules_triggered: Dict of rule_name -> count
            - rate_limiting_events: Number of rate limiting events
            - evaluation_count: Total evaluations performed
            - evaluation_latency_p95_ms: 95th percentile evaluation latency
            - errors: Number of errors encountered
        """
        latency_ms = self._metrics["evaluation_latency_ms"]
        if latency_ms:
            sorted_latencies = sorted(latency_ms)
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_latency_ms = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
        else:
            p95_latency_ms = 0.0

        return {
            "exit_signals_generated": self._metrics["exit_signals_generated"],
            "rules_triggered": dict(self._metrics["rules_triggered"]),
            "rate_limiting_events": self._metrics["rate_limiting_events"],
            "evaluation_count": self._metrics["evaluation_count"],
            "evaluation_latency_p95_ms": round(p95_latency_ms, 2),
            "errors": self._metrics["errors"],
        }


# Global position-based signal generator instance
position_based_signal_generator = PositionBasedSignalGenerator()

