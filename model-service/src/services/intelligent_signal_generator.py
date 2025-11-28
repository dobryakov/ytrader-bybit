"""
Intelligent signal generation service.

Uses model inference to generate trading signals with confidence scores,
applies quality thresholds, and integrates with rate limiting and validation.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal

from ..models.signal import TradingSignal, MarketDataSnapshot
from ..models.position_state import OrderPositionState
from ..services.model_loader import model_loader
from ..services.model_inference import model_inference
from ..services.signal_validator import signal_validator
from ..services.rate_limiter import rate_limiter
from ..database.repositories.position_state_repo import PositionStateRepository
from ..config.settings import settings
from ..config.logging import get_logger, bind_context
from ..config.exceptions import ModelInferenceError, SignalValidationError

logger = get_logger(__name__)


class IntelligentSignalGenerator:
    """Generates trading signals using trained ML models."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        min_amount: float = 100.0,
        max_amount: float = 1000.0,
    ):
        """
        Initialize intelligent signal generator.

        Args:
            min_confidence_threshold: Minimum confidence score to generate signal (0-1)
            min_amount: Minimum order amount in quote currency
            max_amount: Maximum order amount in quote currency
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.position_state_repo = PositionStateRepository()

    async def generate_signal(
        self,
        asset: str,
        strategy_id: str,
        model_version: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal using model inference.

        Args:
            asset: Trading pair symbol (e.g., 'BTCUSDT')
            strategy_id: Trading strategy identifier
            model_version: Model version to use (None for active model)
            trace_id: Trace ID for request flow tracking

        Returns:
            TradingSignal or None if signal generation fails or confidence too low
        """
        bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

        logger.info("Generating intelligent signal", asset=asset, strategy_id=strategy_id, model_version=model_version)

        # Check rate limit
        allowed, reason = rate_limiter.check_rate_limit()
        if not allowed:
            logger.warning("Signal generation rate limited", asset=asset, strategy_id=strategy_id, reason=reason)
            return None

        try:
            # Load model
            if model_version:
                model = await model_loader.load_model_by_version(model_version)
            else:
                model = await model_loader.load_active_model(strategy_id=strategy_id)

            if not model:
                logger.warning("No active model available", asset=asset, strategy_id=strategy_id, model_version=model_version)
                return None

            # Get order/position state
            order_position_state = await self.position_state_repo.get_order_position_state(
                strategy_id=strategy_id,
                asset=asset,
            )

            # Get market data snapshot at inference time
            market_data_snapshot = model_inference.get_market_data_snapshot(asset)
            if not market_data_snapshot:
                logger.warning("Market data unavailable, skipping signal generation", asset=asset, strategy_id=strategy_id)
                return None

            # Prepare features
            features_df = model_inference.prepare_features(
                asset=asset,
                order_position_state=order_position_state,
                market_data_snapshot=market_data_snapshot,
            )

            # Run model prediction
            prediction_result = model_inference.predict(model, features_df)

            # Check confidence threshold
            confidence = prediction_result.get("confidence", 0.0)
            if confidence < self.min_confidence_threshold:
                logger.debug(
                    "Signal confidence below threshold",
                    asset=asset,
                    strategy_id=strategy_id,
                    confidence=confidence,
                    threshold=self.min_confidence_threshold,
                )
                return None

            # Determine signal type from prediction
            signal_type = self._determine_signal_type(prediction_result)

            # Calculate order amount
            amount = self._calculate_amount(asset, order_position_state, market_data_snapshot, confidence)

            # Get model version string for signal
            if not model_version:
                # Get active model version from database
                from ..database.repositories.model_version_repo import ModelVersionRepository
                model_version_repo = ModelVersionRepository()
                active_model = await model_version_repo.get_active_by_strategy(strategy_id)
                model_version = active_model["version"] if active_model else None

            # Create signal
            signal = TradingSignal(
                signal_type=signal_type,
                asset=asset,
                amount=amount,
                confidence=confidence,
                strategy_id=strategy_id,
                model_version=model_version,
                is_warmup=False,
                market_data_snapshot=market_data_snapshot,
                metadata={
                    "reasoning": f"Model prediction: {prediction_result.get('prediction')}",
                    "prediction_result": {
                        "prediction": prediction_result.get("prediction"),
                        "buy_probability": prediction_result.get("buy_probability"),
                        "sell_probability": prediction_result.get("sell_probability"),
                    },
                    "model_version": model_version,
                    "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                },
                trace_id=trace_id,
            )

            # Validate signal
            is_valid, errors = signal_validator.validate(signal)
            if not is_valid:
                logger.warning(
                    "Generated signal failed validation",
                    asset=asset,
                    strategy_id=strategy_id,
                    errors=errors,
                )
                return None

            logger.info(
                "Generated intelligent signal",
                asset=asset,
                strategy_id=strategy_id,
                signal_type=signal_type,
                amount=amount,
                confidence=confidence,
                model_version=model_version,
                trace_id=trace_id,
            )

            return signal

        except ModelInferenceError as e:
            logger.error("Model inference error during signal generation", asset=asset, strategy_id=strategy_id, error=str(e), exc_info=True)
            return None
        except Exception as e:
            logger.error("Error generating intelligent signal", asset=asset, strategy_id=strategy_id, error=str(e), exc_info=True)
            return None

    def _determine_signal_type(self, prediction_result: Dict[str, Any]) -> str:
        """
        Determine signal type from model prediction.

        Args:
            prediction_result: Prediction result from model inference

        Returns:
            'buy' or 'sell'
        """
        # For classification models, use buy/sell probabilities
        buy_probability = prediction_result.get("buy_probability", 0.0)
        sell_probability = prediction_result.get("sell_probability", 0.0)

        if buy_probability > sell_probability:
            return "buy"
        else:
            return "sell"

    def _calculate_amount(
        self,
        asset: str,
        order_position_state: OrderPositionState,
        market_data_snapshot: MarketDataSnapshot,
        confidence: float,
    ) -> float:
        """
        Calculate order amount based on confidence and position state.

        Args:
            asset: Trading pair symbol
            order_position_state: Current order and position state
            market_data_snapshot: Market data snapshot
            confidence: Signal confidence score

        Returns:
            Order amount in quote currency
        """
        # Base amount
        base_amount = (self.min_amount + self.max_amount) / 2

        # Adjust based on confidence (higher confidence = larger amount, up to max)
        confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
        amount = base_amount * confidence_multiplier

        # Adjust based on position state (reduce if already have large position)
        position = order_position_state.get_position(asset)
        if position and position.size != 0:
            # Reduce amount if we already have a position
            position_size_abs = abs(float(position.size))
            if position_size_abs > 0:
                # Reduce amount proportionally to existing position
                reduction_factor = min(1.0, 0.5 / (position_size_abs / base_amount + 0.5))
                amount = amount * reduction_factor

        # Clamp to valid range
        amount = max(self.min_amount, min(self.max_amount, amount))

        return round(amount, 2)

    async def generate_signals_for_strategies(
        self,
        assets: List[str],
        strategy_ids: List[str],
        trace_id: Optional[str] = None,
    ) -> List[TradingSignal]:
        """
        Generate intelligent signals for multiple strategies and assets.

        Args:
            assets: List of trading pair symbols
            strategy_ids: List of strategy identifiers
            trace_id: Trace ID for request flow tracking

        Returns:
            List of generated signals
        """
        signals = []
        for strategy_id in strategy_ids:
            for asset in assets:
                signal = await self.generate_signal(asset, strategy_id, trace_id=trace_id)
                if signal:
                    signals.append(signal)
        return signals


# Initialize intelligent signal generator with settings
def get_intelligent_generator():
    """Get intelligent signal generator instance with current settings."""
    return IntelligentSignalGenerator(
        min_confidence_threshold=settings.model_quality_threshold_accuracy,  # Use quality threshold as confidence threshold
        min_amount=settings.warmup_min_amount,
        max_amount=settings.warmup_max_amount,
    )


# Global intelligent signal generator instance
intelligent_signal_generator = get_intelligent_generator()

