"""
Warm-up mode orchestrator.

Coordinates signal generation, validation, rate limiting, and publishing for warm-up mode.
"""

import asyncio
from typing import List, Optional

from ..models.signal import TradingSignal
from ..services.warmup_signal_generator import warmup_signal_generator
from ..services.rate_limiter import rate_limiter
from ..services.signal_validator import signal_validator, SignalValidationError
from ..publishers.signal_publisher import signal_publisher
from ..config.settings import settings
from ..config.logging import get_logger, bind_context

logger = get_logger(__name__)


class WarmUpOrchestrator:
    """Orchestrates warm-up mode signal generation workflow."""

    def __init__(self):
        """Initialize warm-up orchestrator."""
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start warm-up mode signal generation loop."""
        if self._running:
            logger.warning("Warm-up orchestrator already running")
            return

        if not settings.warmup_mode_enabled:
            logger.info("Warm-up mode is disabled in configuration")
            return

        self._running = True
        self._task = asyncio.create_task(self._generation_loop())
        logger.info("Warm-up orchestrator started")

    async def stop(self) -> None:
        """Stop warm-up mode signal generation."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Warm-up orchestrator stopped")

    async def _generation_loop(self) -> None:
        """Main signal generation loop."""
        logger.info("Starting warm-up signal generation loop")

        # Calculate sleep interval based on frequency
        # frequency is signals per minute, so interval = 60 / frequency seconds
        interval_seconds = 60.0 / settings.warmup_signal_frequency if settings.warmup_signal_frequency > 0 else 60.0

        # Get trading strategies and assets from configuration
        strategies = settings.trading_strategy_list
        if not strategies:
            logger.warning("No trading strategies configured, warm-up mode cannot generate signals")
            return

        # Default assets if not configured (can be extended later)
        assets = ["BTCUSDT", "ETHUSDT"]  # TODO: Make configurable

        while self._running:
            try:
                # Generate signals for all strategies and assets
                signals = await self._generate_signals_for_all(strategies, assets)

                # Process and publish signals
                if signals:
                    await self._process_and_publish_signals(signals)

                # Wait for next generation cycle
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Warm-up generation loop cancelled")
                break
            except Exception as e:
                logger.error("Error in warm-up generation loop", error=str(e), exc_info=True)
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
                    # Check rate limit before generating
                    allowed, reason = rate_limiter.check_rate_limit()
                    if not allowed:
                        logger.debug(
                            "Rate limit exceeded, skipping signal generation",
                            strategy_id=strategy_id,
                            asset=asset,
                            reason=reason,
                        )
                        continue

                    # Generate signal
                    generator = get_warmup_generator()
                    signal = generator.generate_signal(asset, strategy_id)
                    if signal:
                        signals.append(signal)

                except Exception as e:
                    logger.error(
                        "Error generating signal",
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
                    "Published warm-up signals",
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
        Generate a single warm-up signal (for testing or manual triggers).

        Args:
            asset: Trading pair symbol
            strategy_id: Strategy identifier
            trace_id: Trace ID for request flow tracking

        Returns:
            Generated signal or None if generation fails
        """
        bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

        try:
            # Check rate limit
            allowed, reason = rate_limiter.check_rate_limit()
            if not allowed:
                logger.warning("Rate limit exceeded for single signal", reason=reason)
                return None

            # Generate signal
            generator = get_warmup_generator()
            signal = generator.generate_signal(asset, strategy_id, trace_id)
            if not signal:
                return None

            # Validate signal
            signal_validator.validate_and_raise(signal)

            # Publish signal
            await signal_publisher.publish(signal)

            logger.info("Generated and published single warm-up signal", signal_id=signal.signal_id)

            return signal

        except SignalValidationError as e:
            logger.error("Signal validation failed", error=str(e))
            return None
        except Exception as e:
            logger.error("Error generating single signal", error=str(e), exc_info=True)
            return None


# Global warm-up orchestrator instance
warmup_orchestrator = WarmUpOrchestrator()

