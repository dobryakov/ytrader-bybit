"""
Intelligent signal generation orchestrator.

Coordinates model-based signal generation, validation, rate limiting, and publishing.
"""

import asyncio
from typing import List, Optional

from ..models.signal import TradingSignal
from ..services.intelligent_signal_generator import intelligent_signal_generator
from ..services.rate_limiter import rate_limiter
from ..services.signal_validator import signal_validator, SignalValidationError
from ..services.mode_transition import mode_transition
from ..publishers.signal_publisher import signal_publisher
from ..config.settings import settings
from ..config.logging import get_logger, bind_context

logger = get_logger(__name__)


class IntelligentOrchestrator:
    """Orchestrates intelligent (model-based) signal generation workflow."""

    def __init__(self):
        """Initialize intelligent orchestrator."""
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start intelligent signal generation loop."""
        if self._running:
            logger.warning("Intelligent orchestrator already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._generation_loop())
        logger.info("Intelligent orchestrator started")

    async def stop(self) -> None:
        """Stop intelligent signal generation."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Intelligent orchestrator stopped")

    async def _generation_loop(self) -> None:
        """Main signal generation loop."""
        logger.info("Starting intelligent signal generation loop")

        # Calculate sleep interval based on intelligent signal frequency (signals per minute).
        # This is independent from warm-up frequency. If INTELLIGENT_SIGNAL_FREQUENCY is not
        # explicitly set in .env, it defaults to 60 (one signal per second).
        freq = settings.intelligent_signal_frequency
        interval_seconds = 60.0 / freq if freq > 0 else 60.0

        # Get trading strategies and assets from configuration
        strategies = settings.trading_strategy_list
        if not strategies:
            logger.warning("No trading strategies configured, intelligent mode cannot generate signals")
            return

        # Default assets if not configured
        assets = ["BTCUSDT", "ETHUSDT"]  # TODO: Make configurable

        while self._running:
            try:
                # Check for mode transitions (warm-up -> model-based)
                for strategy_id in strategies:
                    await mode_transition.check_and_transition(strategy_id=strategy_id)

                # Generate signals for all strategies and assets
                signals = await self._generate_signals_for_all(strategies, assets)

                # Process and publish signals
                if signals:
                    await self._process_and_publish_signals(signals)

                # Wait for next generation cycle
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Intelligent generation loop cancelled")
                break
            except Exception as e:
                logger.error("Error in intelligent generation loop", error=str(e), exc_info=True)
                # Continue loop even on error
                await asyncio.sleep(interval_seconds)

    async def _generate_signals_for_all(
        self, strategies: List[str], assets: List[str]
    ) -> List[TradingSignal]:
        """
        Generate signals for all strategies and assets.

        Args:
            strategies: List of strategy identifiers
            assets: List of trading pair symbols

        Returns:
            List of generated signals
        """
        signals = []
        for strategy_id in strategies:
            for asset in assets:
                try:
                    # Generate signal using intelligent generator
                    signal = await intelligent_signal_generator.generate_signal(
                        asset=asset,
                        strategy_id=strategy_id,
                    )
                    if signal:
                        signals.append(signal)

                except Exception as e:
                    logger.error(
                        "Error generating intelligent signal",
                        strategy_id=strategy_id,
                        asset=asset,
                        error=str(e),
                        exc_info=True,
                    )
                    # Continue with other signals

        return signals

    async def _process_and_publish_signals(self, signals: List[TradingSignal]) -> None:
        """
        Validate and publish signals.

        Args:
            signals: List of signals to process
        """
        valid_signals = []
        invalid_count = 0

        for signal in signals:
            try:
                # Validate signal
                signal_validator.validate_and_raise(signal)
                valid_signals.append(signal)

            except SignalValidationError as e:
                invalid_count += 1
                logger.warning(
                    "Signal validation failed",
                    signal_id=signal.signal_id,
                    error=str(e),
                )
                # Continue with other signals

        # Publish valid signals
        if valid_signals:
            try:
                published_count = await signal_publisher.publish_batch(valid_signals)
                logger.info(
                    "Published intelligent signals",
                    total=len(signals),
                    valid=len(valid_signals),
                    published=published_count,
                    invalid=invalid_count,
                )
            except Exception as e:
                logger.error(
                    "Failed to publish signals",
                    count=len(valid_signals),
                    error=str(e),
                    exc_info=True,
                )
        elif invalid_count > 0:
            logger.warning("All signals failed validation", total=len(signals), invalid=invalid_count)

    async def generate_single_signal(
        self, asset: str, strategy_id: str, trace_id: Optional[str] = None
    ) -> Optional[TradingSignal]:
        """
        Generate a single intelligent signal (for testing or manual triggers).

        Args:
            asset: Trading pair symbol
            strategy_id: Strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            Generated signal or None if generation fails
        """
        bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

        try:
            # Generate signal
            signal = await intelligent_signal_generator.generate_signal(
                asset=asset,
                strategy_id=strategy_id,
                trace_id=trace_id,
            )
            if not signal:
                return None

            # Validate signal
            signal_validator.validate_and_raise(signal)

            # Publish signal
            await signal_publisher.publish(signal)

            logger.info("Generated and published single intelligent signal", signal_id=signal.signal_id)

            return signal

        except SignalValidationError as e:
            logger.error("Signal validation failed", error=str(e))
            return None
        except Exception as e:
            logger.error("Error generating single signal", error=str(e), exc_info=True)
            return None


# Global intelligent orchestrator instance
intelligent_orchestrator = IntelligentOrchestrator()

